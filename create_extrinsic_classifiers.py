import os 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from typing import List, Dict
import torch
import torch.nn as nn
import argparse
import gc
import random

from salient_event import patch_lib
from salient_event import classifier as classifier_lib

from attribute.segmentor import Segmentor
import cv2 
import argparse
from model import RewardModel
from dataset.dataset import Dataset
from attribute.attribution import Attribute

from create_subgoal_classifiers import plot_classifier, plot_classifier_comparison, load_all_data, save_classifiers

def create_positive_training_set(data_dir: str):
    all_data = load_all_data(data_dir)
    
    states = []
    reward = []
    
    for data_point in tqdm(all_data):
        if data_point["step_extrinsic_reward"] != 0:
            if np.random.rand() < 0.5:
                states.append(data_point['state'])
                reward.append(data_point['step_extrinsic_reward'])
    
    save_dict = {
        'states': states,
        'rewards': reward
    }
    
    print(f"Found {len(states)} positive states")
    
    # with open(os.path.join(data_dir, 'positive_data.pkl'), 'wb') as f:
    #     pickle.dump(save_dict, f)
    
    return save_dict


def get_dataset(positive_data: dict,
                negative_data_dir: str):
    assert os.path.exists(os.path.join(negative_data_dir, 'negative_data.pkl'))
    
    dataset = Dataset(32)
    
    dataset.add_data(positive_data['states'], positive_data['rewards'])
    
    with open(os.path.join(negative_data_dir, 'negative_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        dataset.add_data(data['states'], data['rewards'])
    
    return dataset

def train_model(dataset: Dataset,
                model: RewardModel,
                device,
                epochs: int,
                lr: float,
                model_save_dir: str):
    os.makedirs(model_save_dir, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss_tracker = []
        for _ in trange(dataset.batch_num):
            x, y = dataset.get_batch()
            x = torch.from_numpy(x/255.0).float().to(device)
            y = torch.from_numpy(y/100.0).float().to(device)
            y = y.unsqueeze(-1)
            
            pred_y = model(x)
            loss = criterion(pred_y, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_tracker.append(loss.item())
        
        print(f"Epoch {epoch}: average mse loss {np.mean(epoch_loss_tracker)}")
    torch.save(model.state_dict(), 
               os.path.join(model_save_dir, "reward.model"))
    return model

def create_classifiers_from_model(model: RewardModel,
                                  positive_data: str,
                                  negative_data_dir: str,
                                  att_type: str,
                                  device,
                                  segmentor: Segmentor,
                                  threshold=None,
                                  base_plotting_dir: str = "classifier_plots"):
    assert att_type in ["random", "states"]
    # Create plotting directories
    os.makedirs(base_plotting_dir, exist_ok=True)
    rejected_dir = os.path.join(base_plotting_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)
    
    classifiers = []
    next_classifier_id = 0
    
    attribute = Attribute(device,
                          model=model,
                          attribution_type="deep_lift_shap")
    all_data = random.sample(positive_data['states'], k=2000)
    with open(os.path.join(negative_data_dir, 'negative_data.pkl'), 'rb') as f:
        baselines = pickle.load(f)
        baselines = random.sample(baselines['states'], k=1000)
        
    
    calculate_threshold = False
    if threshold is None:
        calculate_threshold = True
    
    for state in tqdm(all_data, desc="Creating classifiers"):
        state_seg = state.squeeze(0)
        state_seg = state_seg.astype(np.uint8)
        
        state_seg = cv2.cvtColor(state_seg, cv2.COLOR_GRAY2RGB)
        segments, bboxes = segmentor.segment(state_seg)
        
        if att_type == "random":
            base = torch.rand((5,1,84,84)).float().to(attribute.device)
        elif att_type == "states":
            base = np.stack(random.sample(baselines, k=20))
            base = torch.from_numpy(base).float().to(attribute.device)
        
        attributions = attribute.analyze_state(np.expand_dims(state, axis=0), 
                                               base)
        attributions = attributions.squeeze()
        state = np.squeeze(state)
        ave_att = []
        for m in segments:
            att = attributions[m]
            att[att==0] = np.nan
            ave_att.append(np.nanmean(att))
        
        if calculate_threshold:
            norm_att = ave_att
            threshold = np.mean(norm_att) + np.std(norm_att)
        else:
            norm_att = (ave_att-np.min(ave_att))/(np.max(ave_att)-np.min(ave_att))
        
        keep_mask = norm_att >= threshold
        keep_bboxes = [bbox for bbox, mask in zip(bboxes, keep_mask) if mask]
        
        # create salient patches dictionary
        salient_patches = {}
        for bbox in keep_bboxes:
            bbox = tuple(int(x) for x in bbox)
            
            x, y, w, h = bbox
            
            # skip if bbox width or hieght is <= 1
            if w <= 1 or h <= 1 or w*h <= 4:
                continue
            
            # extract bbox patch
            salient_patch = patch_lib.extract_patch(
                state,
                bbox
            )
            
            salient_patches[bbox] = salient_patch
        
        if len(salient_patches) == 0:
            continue
        
        # create new classifier
        new_classifier = classifier_lib.create_classifier(
            salient_patches=salient_patches,
            prototype_image=state,
            prototype_info_vector=None, # just ignoring for now
            classifier_id=None, # assign later if unique
            redundancy_checking_method='histogram',
            base_plotting_dir=base_plotting_dir
        )
        
        # check if classifier is redundant
        is_redundant = False
        
        for existing_classifier in classifiers:
            if classifier_lib.equals(existing_classifier, new_classifier):
                is_redundant = True
                plot_classifier_comparison(
                    existing_classifier,
                    new_classifier,
                    rejected_dir,
                    "Equals comparison"
                )
                break
        
        
        if not is_redundant:
            new_classifier = classifier_lib.assign_id(new_classifier, next_classifier_id)
            classifiers.append(new_classifier)
            plot_classifier(new_classifier, base_plotting_dir)
            next_classifier_id += 1
    
    print(f"Created {len(classifiers)} unique classifiers")
    
    return classifiers
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--classifier_dir", type=str, default="classifiers_ext")
    parser.add_argument("--plot_dir", type=str, default="classifier_ext_plots")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--attr_type", type=str, choices=["random", "states"])
    parser.add_argument("--use_cpu", action="store_true")
    
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reward_model_dir", type=str, default="reward_model")
    parser.add_argument("--neg_data", type=str, default="resources/negative_data")
    
    print("Creating extrinsic reward classifiers")
    
    args = parser.parse_args()
    
    segmentor = Segmentor()
    reward_model = RewardModel()
    
    device = torch.device('cpu' if args.use_cpu else 'cuda')
    reward_model.to(device)
    
    positive_data = create_positive_training_set(args.data_dir)
    
    print("Positive data collected")
    
    if args.train_model:
        print("training model")
        dataset = get_dataset(positive_data,
                              args.neg_data)
        reward_model = train_model(dataset,
                                   reward_model,
                                   device,
                                   args.epochs,
                                   args.lr,
                                   args.reward_model_dir)
        del dataset
        gc.collect()
    else:
        print("loading model")
        assert os.path.exists(os.path.join(args.reward_model_dir,
                                           "reward.model"))
        reward_model.load_state_dict(torch.load(
            os.path.join(args.reward_model_dir,
                         "reward.model")
        ))
    
    print("creating classifiers")
    classifiers = create_classifiers_from_model(reward_model,
                                                positive_data,
                                                args.neg_data,
                                                args.attr_type,
                                                device,
                                                segmentor,
                                                args.threshold,
                                                args.plot_dir)
    save_classifiers(classifiers, args.classifier_dir)
    
    
    
    
    


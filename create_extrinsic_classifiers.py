import os 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from typing import List, Dict
import torch
import torch.nn as nn

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
        if data_point["attribution_low"] is None:
            states.append(data_point['state'])
            reward.append(data_point['step_extrinsic_reward'])
    
    save_dict = {
        'states': states,
        'rewards': reward
    }
    
    with open(os.path.join(data_dir, 'positive_data.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)


def get_dataset(positive_data_dir: str,
                negative_data_dir: str):
    assert os.path.exists(os.path.join(positive_data_dir, 'positive_data.pkl'))
    assert os.path.exists(os.path.join(negative_data_dir, 'negative_data.pkl'))
    
    dataset = Dataset(32)
    
    with open(os.path.join(positive_data_dir, 'positive_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        dataset.add_data(data['states'], data['rewards'])
    
    with open(os.path.join(negative_data_dir, 'negative_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        dataset.add_data(data['states'], data['rewards'])
    
    return dataset

def train_model(dataset: Dataset,
                model: RewardModel,
                device,
                epochs: int,
                model_save_dir: str):
    os.makedirs(model_save_dir, exist_ok=True)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in epochs:
        epoch_loss_tracker = []
        for _ in trange(dataset.batch_num):
            x, y = dataset.get_batch()
            x = torch.from_numpy(x/255.0).float().to(device)
            y = torch.from_numpy(y/100.0).float().to(device)
            
            pred_y = model(x)
            loss = criterion(pred_y, y).mean(-1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_tracker.append(loss.item())
        
        print(f"Epoch {epoch}: average mse loss {np.mean(epoch_loss_tracker)}")

def create_classifiers_from_model(model,
                                  positive_data_dir,
                                  negative_data_dir,
                                  att_type,
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
    with open(os.path.join(positive_data_dir, 'positive_data.pkl'), 'rb') as f:
        all_data = pickle.load(f)
    with open(os.path.join(negative_data_dir, 'negative_data.pkl'), 'rb') as f:
        baselines = pickle.load(f)
    
    
    
    calculate_threshold = False
    if threshold is None:
        calculate_threshold = True
    
    for state in all_data:
        state_seg = state.astype(np.uint8)
        
        state_seg = cv2.cvtColor(state_seg, cv2.COLOR_GRAY2RGB)
        segments, bboxes = segmentor.segment(state_seg)
        
        if att_type == "random":
            base = torch.rand((5,1,84,84)).float().to(attribute.device)
        elif att_type == "states":
            base = np.random.choice(baselines, size=5, replace=False)
            base = torch.from_numpy(base).float().to(attribute.device)
        
        attributions = attribute.analyze_state(state, base)
        
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
            
        


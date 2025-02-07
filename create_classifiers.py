import numpy as np
import argparse
import glob
import os
import pickle
from model import RNDTarget, RNDPredictor
from attribute.attribution import Attribute
import torch
from salient_event import patch_lib
from salient_event import classifier as cls
import random

def main(data_dir, classifier_dir):
    # TODO
    # load data and check which states are if ontronsic and extrinsic reward
    # load RND model and attribute image
    # learn classifier by extracting most important patch
    # learn model for extrinsic reward
    # attribute and learn classifiers for extrinsic rewards
    
    # QUESTION: what do we use for baseline images?
    
    target = RNDTarget()
    target.load_state_dict(torch.load(os.path.join(data_dir, "MontezumaRevengeNoFrameskip-v4.target")))
    pred = RNDPredictor()
    pred.load_state_dict(torch.load(os.path.join(data_dir, "MontezumaRevengeNoFrameskip-v4.pred")))
    # change this to toggle
    device = torch.device("cuda")
    attribution = Attribute(pred,
                            target,
                            device,
                            type="deep_lift_shape",
                            segmentation_type="sam") 
    
    files = glob.glob(os.path.join(data_dir, '*.pkl'))
    cls_idx = 0
    
    false_data = []
    true_data = []
    
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
        
        for sample in data:
            # this may need to change if using 2 or 3 times std
            if sample.intrinsic_rewards > (sample.intrinsic_reward_mean+sample.intrinsic_reward_std):
                # this is an intrinsic reward goal
                subgoal = sample.states
                # need to find relevant baselines
                _, bbox = attribution.analyze_state(subgoal, 
                                                    baselines=None)
                patch = patch_lib.extract_patch(subgoal, bbox)
                
                classifier = cls.create_classifier(
                    salient_patches={'bbox': bbox,
                                     'patch': patch},
                    prototype_image=subgoal,
                    prototype_info_vector=sample.info,
                    classifier_id=cls_idx
                )
                with open(os.path.join(classifier_dir, f'{cls_idx}.pkl')) as f:
                    pickle.dump(classifier, f)
                
                cls_idx += 1
            
                if random.random() < 0.01:
                    false_data.append(subgoal)
            
            else:
                # this is an extrinsic reward goal
                true_data.append(sample.states)
    
    with open(os.path.join(classifier_dir, "true_data.pkl")) as f:
        pickle.dump(true_data, f)
    
    with open(os.path.join(classifier_dir, "false_data.pkl")) as f:
        pickle.dump(false_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir', type=str)
    parser.add_argument('-c', '--classifier_dir', type=str)
    
    args = parser.parse_args()
    
    main(args.data_dir, args.classifier_dir)
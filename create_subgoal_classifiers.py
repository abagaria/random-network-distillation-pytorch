import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict

from salient_event import patch_lib
from salient_event import classifier as classifier_lib

from attribute.segmentor import Segmentor
import cv2
import argparse
import math


def plot_classifier(classifier: Dict, output_dir: str):
    """Plot the prototype image with bounding box for a classifier."""
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Get prototype image and bbox
    image = classifier['prototype_image']
    bboxes = list(classifier['salient_patches'].keys())
    
    # Plot image
    plt.imshow(image, cmap='gray')
    
    for bbox in bboxes:
        # Add bounding box
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    # Add classifier ID and bbox dimensions as title
    plt.title(f"Classifier {classifier['classifier_id']} - bbox: {w}x{h}")
    
    # Remove axes
    plt.axis('off')
    
    # Save plot
    output_path = os.path.join(output_dir, f"classifier_{classifier['classifier_id']}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()


def plot_classifier_comparison(existing_classifier: Dict, new_classifier: Dict, output_dir: str, reason: str):
    """Plot two classifiers side by side for comparison."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot existing classifier
    image1 = existing_classifier['prototype_image']
    bboxes1 = list(existing_classifier['salient_patches'].keys())
    
    ax1.imshow(image1, cmap='gray')
    for bbox1 in bboxes1:
        x1, y1, w1, h1 = bbox1
        rect1 = plt.Rectangle((x1, y1), w1, h1, fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect1)
    ax1.set_title(f"Existing Classifier {existing_classifier['classifier_id']} - bbox: {w1}x{h1}")
    ax1.axis('off')
    
    # Plot new classifier
    image2 = new_classifier['prototype_image']
    bboxes2 = list(new_classifier['salient_patches'].keys())
    
    ax2.imshow(image2, cmap='gray')
    for bbox2 in bboxes2:
        x2, y2, w2, h2 = bbox2
        rect2 = plt.Rectangle((x2, y2), w2, h2, fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect2)
    ax2.set_title(f"Rejected New Classifier - bbox: {w2}x{h2}")
    ax2.axis('off')
    
    # Add super title indicating reason for rejection
    plt.suptitle(f"Rejection Reason: {reason}", y=0.95)
    
    # Save plot
    output_path = os.path.join(
        output_dir, 
        f"rejected_comparison_existing_{existing_classifier['classifier_id']}.png"
    )
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()


def create_classifiers_from_data(data_dir: str, 
                                 segmentor: Segmentor, 
                                 att_type: str,
                                 threshold = None,
                                 base_plotting_dir: str = "classifier_plots"):
    """Process all data points and create unique classifiers."""
    
        
    
    # Create plotting directories
    os.makedirs(base_plotting_dir, exist_ok=True)
    rejected_dir = os.path.join(base_plotting_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)
    
    # Load all data
    all_data = load_all_data(data_dir)
    
    # Initialize list of classifiers
    classifiers = []
    next_classifier_id = 0
    
    print(f"Processing {len(all_data)} data points...")
    
    calculate_threshold = False
    if threshold is None:
        calculate_threshold = True
    
    for data_point in tqdm(all_data, desc="Creating classifiers"):
        # Skip if bbox is None (this happens for extrinsically rewarding transitions)
        if data_point[f'attribution_{att_type}'] is None:
            continue
        
        state = data_point['state'].squeeze(0)
        state_seg = state.astype(np.uint8)
        
        state_seg = cv2.cvtColor(state_seg, cv2.COLOR_GRAY2RGB)
        segments, bboxes = segmentor.segment(state_seg)
        attribution = data_point[f'attribution_{att_type}'].squeeze()
        ave_att = []
        for m in segments:
            att = attribution[m]
            att_mean = np.mean(att)
            if math.isnan(att_mean):
                att_mean = 0
            ave_att.append(att_mean)
        
        if calculate_threshold:
            norm_att = ave_att
            threshold = np.mean(norm_att) + np.std(norm_att)
        else:
            norm_att = (ave_att-np.min(ave_att))/(np.max(ave_att)-np.min(ave_att))
        
        keep_mask = norm_att >= threshold
        keep_bboxes = [bbox for bbox, mask in zip(bboxes, keep_mask) if mask]
        
        # Create salient patches dictionary
        salient_patches = {}
        for bbox in keep_bboxes:
            bbox = tuple(int(x) for x in bbox)
        
            x, y, w, h = bbox
        
            # Skip if bbox width or height is <= 1
            if w <= 1 or h <= 1 or w*h <= 4:
                continue
        
            # Extract the patch corresponding to the bbox
            salient_patch = patch_lib.extract_patch(
                state,
                bbox
            )
            
            salient_patches[bbox] = salient_patch
        
        if len(salient_patches) == 0:
            continue
        
        # Create new classifier
        new_classifier = classifier_lib.create_classifier(
            salient_patches=salient_patches,
            prototype_image=state,
            prototype_info_vector=data_point['info']['ram'],
            classifier_id=None,  # Will assign later if unique
            redundancy_checking_method='histogram',
            base_plotting_dir=base_plotting_dir
        )
        
        # Check if this classifier is redundant
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
        
        # If not redundant, assign ID, add to list, and plot
        if not is_redundant:
            new_classifier = classifier_lib.assign_id(new_classifier, next_classifier_id)
            classifiers.append(new_classifier)
            plot_classifier(new_classifier, base_plotting_dir)
            next_classifier_id += 1
    
    print(f"Created {len(classifiers)} unique classifiers")
    return classifiers


def load_all_data(data_dir: str) -> List[Dict]:
    """Load all pickle files and return their data points."""
    all_data = []
    pickle_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
    
    for pkl_file in tqdm(pickle_files, desc="Loading pickle files"):
        file_path = os.path.join(data_dir, pkl_file)
        with open(file_path, 'rb') as f:
            data_list = pickle.load(f)
            all_data.extend(data_list)
    
    return all_data


def save_classifiers(classifiers: List[Dict], save_dir: str):
    """Save classifiers to disk."""
    os.makedirs(save_dir, exist_ok=True)
    
    for classifier in classifiers:
        filename = f"classifier_{classifier['classifier_id']}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(classifier, f)
    
    print(f"Saved {len(classifiers)} classifiers to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="rnd_lifetime_data1")
    parser.add_argument("--save_dir", type=str, default="classifiers")
    parser.add_argument("--plot_dir", type=str, default="classifier_plots")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--attr_type", type=str, choices=["init", "low", "rand"], default="low")

    # attr_type options = ["init", "low", "rand"]
    # init uses only initial state
    # low uses all low states from all complete runs and initial state
    # rand uses a sample using torch.rand()
    
    args = parser.parse_args()
    data_dir = args.data_dir
    attr_type = args.attr_type
    threshold = args.threshold
    plot_dir = args.plot_dir
    save_dir = args.save_dir
        
    segmentor = Segmentor()
    
    # Create classifiers
    classifiers = create_classifiers_from_data(data_dir, 
                                               segmentor,
                                               attr_type,
                                               threshold,
                                               plot_dir)
    
    # Save classifiers
    save_classifiers(classifiers, save_dir)

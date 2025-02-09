import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict

from salient_event import patch_lib
from salient_event import classifier as classifier_lib


def plot_classifier(classifier: Dict, output_dir: str):
    """Plot the prototype image with bounding box for a classifier."""
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Get prototype image and bbox
    image = classifier['prototype_image']
    bbox = list(classifier['salient_patches'].keys())[0]  # Get the first (and only) bbox
    x, y, w, h = bbox
    
    # Plot image
    plt.imshow(image, cmap='gray')
    
    # Add bounding box
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
    bbox1 = list(existing_classifier['salient_patches'].keys())[0]
    x1, y1, w1, h1 = bbox1
    
    ax1.imshow(image1, cmap='gray')
    rect1 = plt.Rectangle((x1, y1), w1, h1, fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect1)
    ax1.set_title(f"Existing Classifier {existing_classifier['classifier_id']} - bbox: {w1}x{h1}")
    ax1.axis('off')
    
    # Plot new classifier
    image2 = new_classifier['prototype_image']
    bbox2 = list(new_classifier['salient_patches'].keys())[0]
    x2, y2, w2, h2 = bbox2
    
    ax2.imshow(image2, cmap='gray')
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


def create_classifiers_from_data(data_dir: str, base_plotting_dir: str = "classifier_plots"):
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
    
    for data_point in tqdm(all_data, desc="Creating classifiers"):
        # Skip if bbox is None (this happens for extrinsically rewarding transitions)
        if data_point['bbox'] is None:
            continue
        
        # Create salient patches dictionary
        bbox = tuple(int(x) for x in data_point['bbox'])
        
        x, y, w, h = bbox
        
        # Skip if bbox width or height is <= 1
        if w <= 1 or h <= 1 or w*h <= 4:
            continue
        
        state = data_point['state'].squeeze(0)
        
        # Extract the patch corresponding to the bbox
        salient_patch = patch_lib.extract_patch(
            state,
            bbox
        )
        
        salient_patches = {bbox: salient_patch}
        
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
    data_dir = "rnd_lifetime_data5"
    save_dir = "classifiers"
    plot_dir = "classifier_plots"
    
    # Create classifiers
    classifiers = create_classifiers_from_data(data_dir, plot_dir)
    
    # Save classifiers
    save_classifiers(classifiers, save_dir)

import os 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import glob

from salient_event import classifier as classifier_lib

def plot_classifier_positive(classifier: Dict,
                             states: List,
                             rewards: List,
                             output_dir: str):
    fig, axes = plt.subplot(4,4)
    
    bboxes = list(classifier['salient_patches'].keys())
    
    def add_bboxes(ax, image, reward, bboxes):
        ax.imshow(image, cmap='gray')
        for bbox in bboxes:
            x, y, w, h, = bbox
            rect = plt.Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=2)
            ax.add_patch(rect)
        ax.set_title(f"Reward: {reward}")
        ax.axis('off')
    
    add_bboxes(axes[0], classifier['prototype_image'], "-", bboxes)
    
    ax_idx = 1
    
    for state, reward in zip(states, rewards):
        add_bboxes(axes[ax_idx], state, reward, bboxes)
    
    output_path = os.path.join(
        output_dir,
        f"classifier_{classifier['classifier_id']}.png"
    )
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()

def get_all_classifiers(save_dir):
    classifiers = []
    for file in glob.glob(os.path.join(save_dir,"*.pkl")):
        with open(file, 'rb') as f:
            classifiers.append(pickle.load(f))
    
    return classifiers

def test_classifiers(data_file: str,
                     classifiers: List,
                     plot_dir: str):
    
    with open(data_file, 'rb') as f:
        data_points = pickle.load(f)
    
    for classifier in classifiers:
        states = []
        rewards = []
        for idx in range(len(data_points["states"])):
            if len(states) == 15:
                break
            
            state = data_points["states"][idx]
            reward = data_points["rewards"][idx]
            
            state = np.squeeze(state)
            term = classifier_lib.classify(classifier, state)
            if term:
                states.append(state)
                rewards.append(reward)
            
        plot_classifier_positive(classifier,
                                 states,
                                 rewards,
                                 plot_dir)

if __name__ == "__main__":
    classifier_dir = "classifiers_ext"
    plot_dir = "ext_test"
    
    
    classifiers = get_all_classifiers(classifier_dir)



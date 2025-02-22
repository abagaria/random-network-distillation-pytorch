import os 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import glob

from salient_event import classifier as classifier_lib
from create_subgoal_classifiers import load_all_data
from tqdm import tqdm, trange

def plot_classifier_positive(classifier: Dict,
                             states: List,
                             rewards: List,
                             output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(4,4, figsize=(36,24))
    
    bboxes = list(classifier['salient_patches'].keys())
    
    def add_bboxes(ax, image, reward, bboxes):
        ax.imshow(image, cmap='gray')
        for bbox in bboxes:
            x, y, w, h, = bbox
            rect = plt.Rectangle((x,y),w,h,fill=False,edgecolor='red',linewidth=2)
            ax.add_patch(rect)
        ax.set_title(f"Reward: {reward}")
        ax.axis('off')
        
    add_bboxes(axes[0,0], classifier['prototype_image'], "-", bboxes)
    
    ax_idx = 1
    
    for state, reward in zip(states, rewards):
        add_bboxes(axes[ax_idx//4, ax_idx%4], state, reward, bboxes)
        ax_idx += 1
    
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

def get_data(data_dir: str):
    all_data = load_all_data(data_dir)
    
    states = []
    reward = []
    
    for data_point in tqdm(all_data):
        if data_point["step_extrinsic_reward"] != 0:
            if np.random.rand() < 0.15:
                states.append(data_point['state'])
                reward.append(data_point['step_extrinsic_reward'])
    
    save_dict = {
        'states': states,
        'rewards': reward
    }
    
    print(f"Found {len(states)} positive states")
    
    return save_dict

def test_classifiers(data_points: Dict,
                     classifiers: List,
                     plot_dir: str):
    
    print("testing classifiers")
    for classifier in tqdm(classifiers):
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
    print("classifiers tested")

if __name__ == "__main__":
    
    classifier_dir = "run_6/ext_classifiers6_calcThresh_rand"
    plot_dir = "ext_no_reward_test"
    data_dir = "rnd_lifetime_data6"
    
    
    classifiers = get_all_classifiers(classifier_dir)
    data = get_data(data_dir)
    
    # with open("resources/negative_data/negative_data.pkl", "rb") as f:
    #     data = pickle.load(f)
    
    print(len(classifiers))
    test_classifiers(data,
                     classifiers,
                     plot_dir)
    
    



import os 
import pickle 
import gym
from envs import MaxAndSkipEnv
import random
import numpy as np
from PIL import Image
import cv2
import pickle

def get_all_files(base_dir):
    file_list = []
    
    for root, _, files in os.walk(base_dir):
        
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    
    
    return file_list

def set_state(env, ram_file):
    with open(ram_file, 'rb') as f:
        ram = pickle.load(f)
    
    env.unwrapped.ale.restoreState(ram)
    env.step(0)
    
    return env

def pre_proc(x):
    x = np.array(Image.fromarray(x).convert('L')).astype('float32')
    x = cv2.resize(x, (84, 84))
    return x

def get_data(ram_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    ram_files = get_all_files(ram_dir)
    states = []
    rewards = []
    
    env = MaxAndSkipEnv(gym.make("MontezumaRevengeNoFrameskip-v4"), False)
    env.reset()
    
    for file in ram_files:
        
        for _ in range(50):
            env.reset()
            env = set_state(env, file)
            
            for _ in range(100):
                # 18 actions
                action = random.randint(0, 17)
                obs, reward, _, _ = env.step(action)
                obs = pre_proc(obs)
                obs = np.expand_dims(obs, axis=0)
                states.append(obs)
                rewards.append(reward)

    print(len(states))
    print(len(rewards))
    
    save_dict = {
        'states': states,
        'rewards': rewards
    }
    
    with open(os.path.join(save_dir, 'negative_data.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)
    
        

if __name__ == '__main__':
    get_data('resources/monte_state', "resources/negative_data")
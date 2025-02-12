
import os
import time
import pickle
import numpy as np
import json

from collections import deque


class DataLogger:
    def __init__(self, save_dir, buffer_size=1000):
        self._get_save_dir(save_dir)
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.current_file_count = 0
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _get_save_dir(self, base_save_dir):
        # make sure we dont overwrite data if we run on same machine
        i = 0
        while os.path.exists(base_save_dir+str(i)):
            i += 1
        
        self.save_dir = base_save_dir+str(i)
        
    
    def add_steps(self, 
                  state, 
                  intrinsic_reward, 
                  extrinsic_reward, 
                  info, 
                  done,
                  intrin_rew_mean,
                  intrin_rew_std,
                  attribution_init,
                  attribution_low,
                  attribution_rand):
        batch_data = {
            'state': state,
            'intrinsic_reward': intrinsic_reward,
            'extrinsic_reward': extrinsic_reward,
            'info': info,
            'done': done,
            'intrinsic_reward_mean': intrin_rew_mean,
            'intrinsic_reward_std': intrin_rew_std,
            'attribution_init': attribution_init,
            'attribution_low': attribution_low,
            'attribution_rand': attribution_rand,
            'timestamp': time.time()
        }
        
        self.buffer.append(batch_data)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        if len(self.buffer) == 0:
            return
            
        filename = os.path.join(self.save_dir, f'data_{self.current_file_count}.pkl')
        
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        
        self.current_file_count += 1
        self.buffer.clear()


def pickle_to_memmap(pickle_dir, memmap_dir):
    """Convert pickle files to memmap format for efficient processing."""
    os.makedirs(memmap_dir, exist_ok=True)
    
    # First pass: count total samples and get shapes
    total_samples = 0
    shapes = {}
    dtypes = {}
    
    pickle_files = sorted([f for f in os.listdir(pickle_dir) if f.endswith('.pkl')])
    
    print('Fetching metadata..')
    
    for filename in pickle_files:
        with open(os.path.join(pickle_dir, filename), 'rb') as f:
            data_batches = pickle.load(f)
            import ipdb; ipdb.set_trace()
            for batch in data_batches:
                for key, value in batch.items():
                    if isinstance(value, np.ndarray):
                        if key not in shapes:
                            shapes[key] = value.shape[1:]  # Store shape excluding batch dimension
                            dtypes[key] = value.dtype
                total_samples += len(batch['states'])
    
    print('Creating memmap..')
    
    # Create memmap files
    memmaps = {}
    for key in shapes:
        mmap_path = os.path.join(memmap_dir, f'{key}.npy')
        memmaps[key] = np.memmap(mmap_path, dtype=dtypes[key], mode='w+',
                                shape=(total_samples, *shapes[key]))
        
    print('Iterating over pickle files..')
    
    # Second pass: fill memmap files
    current_idx = 0
    for filename in pickle_files:
        with open(os.path.join(pickle_dir, filename), 'rb') as f:
            data_batches = pickle.load(f)
            for batch in data_batches:
                batch_size = len(batch['states'])
                for key, value in batch.items():
                    if isinstance(value, np.ndarray):
                        memmaps[key][current_idx:current_idx + batch_size] = value
                current_idx += batch_size
    
    # Save metadata
    metadata = {
        'total_samples': total_samples,
        'shapes': {k: list(v) for k, v in shapes.items()},
        'dtypes': {k: str(v) for k, v in dtypes.items()}
    }
    
    with open(os.path.join(memmap_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    print('Created metadata.json..')
    
    # Flush all memmaps
    for mmap in memmaps.values():
        mmap.flush()
    
    return metadata


def load_memmap_data(memmap_dir, start_idx=0, end_idx=None):
    """Load data from memmap files"""
    with open(os.path.join(memmap_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    data = {}
    for key in metadata['shapes']:
        mmap_path = os.path.join(memmap_dir, f'{key}.npy')
        shape = metadata['shapes'][key]
        dtype = np.dtype(metadata['dtypes'][key])
        
        fp = np.memmap(mmap_path, dtype=dtype, mode='r',
                      shape=(metadata['total_samples'], *shape))
        
        if end_idx is None:
            end_idx = len(fp)
        
        data[key] = fp[start_idx:end_idx]
    
    return data

class StatTracker():
    def __init__(self):
        self.count = 0
        self.running_mean = 0
        self.sum_sq = 0
    
    def update(self, rewards):
        self.count += len(rewards)
        old_mean = self.running_mean
        self.running_mean += np.sum(rewards - self.running_mean)/self.count
        self.sum_sq += np.sum((rewards - old_mean)*(rewards - self.running_mean))
    
    def mean(self):
        return self.running_mean
    
    def std(self):
        if self.count < 2:
            return 0
        return np.sqrt(self.sum_sq/(self.count-1))
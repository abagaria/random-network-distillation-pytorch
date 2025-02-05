from dataset.dataset import Dataset
from attribute.attribution import Attribute
from rnd_only.agent import RNDAgent
import numpy as np
import torch
import matplotlib.pyplot as plt

class Experiment():
    def __init__(self,
                 batch_size,
                 learning_rate,
                 type="deep_lift_shap",
                 segmentor_type="sam",
                 update_proportion=0.25,
                 use_cuda=True,
                 **segment_kwargs):
        self.train_dataset = Dataset(batch_size=batch_size)
        self.test_dataset = Dataset(batch_size=batch_size)
        self.agent = RNDAgent(learning_rate,
                              update_proportion,
                              use_cuda)
        self.attribution = Attribute(self.agent.rnd.predictor,
                                     self.agent.rnd.target,
                                     self.agent.device,
                                     type,
                                     segmentation_type=segmentor_type,
                                     **segment_kwargs) 
    
    def load_train_data(self, folder):
        self.train_dataset.load(folder)
    
    def load_test_data(self, folder):
        self.test_dataset.load(folder)
    
    def train(self, epochs):
        for epoch in range(epochs):
            for _ in range(self.train_dataset.batch_num):
                x = self.train_dataset.get_batch()
                x = torch.from_numpy(x/255).float().to(self.agent.device)
                x = torch.permute(x, (0, 3, 1, 2))
                self.agent.train_step(x)
            
            print(f"Epoch {epoch}/{epochs}")
    
    def get_classifier(self):
        data = self.test_dataset.memory
                
        intrinsic_reward = []
        for state in data:
            x = torch.from_numpy(state/255).float().to(self.agent.device)
            x = torch.permute(x, (2,0,1))
            x = torch.unsqueeze(x, 0)
            intrinsic_reward.append(self.agent.compute_intrinsic_reward(x))
        
        intrinsic_reward = np.squeeze(np.array(intrinsic_reward))
                
        max_state_idx = np.argmax(intrinsic_reward)
        max_state = data[max_state_idx]
        
        # plt.imshow(max_state.squeeze())
        # plt.title(f"reward: {intrinsic_reward[max_state_idx]}")
        # plt.savefig("max_state.png")
        
        # plt.clf()
        
        sorted_idxs = np.argsort(intrinsic_reward)
        min_state = data[sorted_idxs[:5]]
        
        # for idx in range(len(min_state)):
        #     state = min_state[idx].squeeze()
        #     plt.imshow(state)
        #     plt.title(f"reward: {intrinsic_reward[int(sorted_idxs[idx])]}")
        #     plt.savefig(f"min_state_{idx}.png")
            
        #     plt.clf()
        
        max_state = torch.from_numpy(max_state/255).float().to(self.agent.device)
        max_state = torch.permute(max_state, (2,0,1))
        max_state = torch.unsqueeze(max_state, 0)
        
        min_state = min_state.squeeze()
        min_state = torch.from_numpy(min_state/255).float().to(self.agent.device)
        # min_state = torch.permute(min_state, (0,3,1,2))
        min_state = torch.unsqueeze(min_state, 1)

        
        
        self.attribution.analyze_state(max_state, min_state, plot=True)
        
        
    
    
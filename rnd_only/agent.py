import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn 
import torch.optim as optim

from torch.distributions.categorical import Categorical

from rnd_only.model import RNDModel
from rnd_only.utils import global_grad_norm_

class RNDAgent():
    def __init__(self, 
                 learning_rate,
                 update_proportion=0.25,
                 use_cuda=True):
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.update_proportion = update_proportion
        
        self.rnd = RNDModel().to(self.device)
        self.optimizer = optim.Adam(list(self.rnd.predictor.parameters()),
                                    lr=learning_rate)
        
        self.criterion = nn.MSELoss(reduction='none')
    
    def compute_intrinsic_reward(self, next_obs):
        
        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1)/2
        
        return intrinsic_reward.data.cpu().numpy()
    
    def target(self, obs):
        return self.rnd.target(obs)
    
    def predict(self, obs):
        return self.rnd.predictor(obs)
    
    def train_step(self, batch_obs):
        predict_next_obs_feature, target_next_state_feature = self.rnd(batch_obs)
        forward_loss = self.criterion(predict_next_obs_feature, target_next_state_feature.detach()).mean(-1)
        
        mask = torch.rand(len(forward_loss)).to(self.device)
        mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
        
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), 
                                                               torch.Tensor([1]).to(self.device))
        
        self.optimizer.zero_grad()
        forward_loss.backward()
        global_grad_norm_(list(self.rnd.predictor.parameters()))
        self.optimizer.step()
        
        


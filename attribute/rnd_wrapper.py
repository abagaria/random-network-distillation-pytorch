import torch.nn as nn

class RNDModelWrapper(nn.Module):
    def __init__(self, predict_model, target_model):
        super().__init__()
        self.predict_model = predict_model
        self.target_model = target_model
    
    def forward(self, input):
        predict_out = self.predict_model(input)
        target_out = self.target_model(input)
        reward = (predict_out-target_out).pow(2).sum(dim=1)
        return reward
from typing import Tuple
import numpy as np
from PIL import Image
import gym
from gym.core import Env, Wrapper, ObservationWrapper
import cv2 

class ResizeObsWrapper(ObservationWrapper):
    def __init__(self, env, image_size: Tuple):
        super().__init__(env)
        self.image_size = image_size
    
    """Resize the observation image."""
    def observation(self, observation):
        observation = np.squeeze(observation)
        img = Image.fromarray(observation)
        return np.asarray(img.resize(self.image_size, Image.BILINEAR))

class TransposeWrapper(ObservationWrapper):
    def observation(self, observation):
        observation = observation.transpose(2,0,1)
        return observation


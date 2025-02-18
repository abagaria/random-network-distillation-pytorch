import numpy as np
from collections import deque
import glob
import math
import os
import random

class Dataset():
    # very basic dataset
    def __init__(self,
                 batch_size):
        self.batch_size = batch_size
        self.random_idxs = None
        self.memory = deque()
        self.labels = deque()
        self.batch_num = 0
        self.counter = 0
    
    def _get_random_sample(self):
        self.random_idxs = np.arange(0, len(self.memory))
        np.random.shuffle(self.random_idxs)
    
    def add_data(self, data, labels):
        assert len(data) == len(labels)
        self.memory = self.concatenate(self.memory, data)
        self.labels = self.concatenate(self.labels, labels)
        self._set_batch_num()
        self.counter = 0
        self.random_idxs = np.random.permutation(len(self.memory))
        
    def _set_batch_num(self):
        self.batch_num = math.ceil(
            len(self.memory)/self.batch_size
        )
    
    def get_batch(self):
        if (self.index() + self.batch_size) > len((self.memory)):
            data = self.memory[self.random_idxs[self.index():]]
            labels = self.labels[self.random_idxs[self.index():]]
        else:
            data = self.memory[self.random_idxs[self.index():self.index()+self.batch_size]]
            labels = self.labels[self.random_idxs[self.index():self.index()+self.batch_size]]
        
        self.counter += 1
        return data, labels
    
    @staticmethod
    def concatenate(arr1, arr2):
        if len(arr1) == 0:
            return arr2
        else:
            return np.concatenate((arr1, arr2), axis=0)
    
    def index(self):
        return (self.counter*self.batch_size)%len(self.memory)
        
import matplotlib.pyplot as plt
import numpy as np
import torch 
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from skimage.segmentation import slic, felzenszwalb, clear_border
from skimage.feature import canny 
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage import morphology
from skimage.measure import label
from skimage.filters import threshold_otsu

import time

TYPES = [
    "sam",
    "slic",
    "felzenszwalb",
    "canny_edge",
    "threshold_otsu"
]

class Segmentor():
    def __init__(self,
                 segment_type="sam",
                 device="cuda",
                 n_segments=100,
                 compactness=1,
                 scale=1,
                 sigma=0.8,
                 min_size=20):
        assert segment_type in TYPES
        
        self.type = segment_type
        self.device = device
        
        # slic parameters
        self.n_segments = n_segments
        self.compactness = compactness
        
        # falzenswalb parameters
        self.scale = scale 
        self.sigma = sigma 
        self.min_size = min_size
        
        if self.type == "sam":
            model = sam_model_registry["vit_h"]("sam_vit_h_4b8939.pth")
            model.to(self.device)
            self.mask_generator = SamAutomaticMaskGenerator(model,
                                                            points_per_side=24)
        
    def segment(self, image, plot=False):
        if type(image) == torch.Tensor:
            image = image.numpy()
        
        if np.max(image) < 1:
            image = image*255
        
        image = np.squeeze(image)
        
        if image.shape[0] == 3:
            image = np.transpose(image, (1,2,0))
        
        lst_mask = None
        bbox = None
        if self.type == "sam":
            lst_mask, bbox = self.sam(image)
        if self.type == "slic":
            lst_mask = self.slic(image)
        if self.type == "felzenszwalb":
            lst_mask = self.felzenszwalb(image)
        if self.type == "canny_edge":
            lst_mask = self.canny(image)
        if self.type == "threshold_otsu":
            lst_mask = self.threshold_otsu(image)
        
        if plot:
            self.plot_masks(lst_mask)
            plt.imshow(image[:,:,0])
            plt.savefig("original")
            plt.cla()
        
        return lst_mask, bbox
    
    def plot_masks(self, masks):
        if len(masks) == 0:
            return
        
        sorted_masks = sorted(masks, key=(lambda x: np.sum(x)), reverse=True)
        
        ax = plt.gca()
        img = np.ones((sorted_masks[0].shape[0], 
                       sorted_masks[0].shape[1],
                       4))
        img[:,:,3] = 0
        for mask in sorted_masks:
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[mask] = color_mask
        ax.imshow(img)
        
        plt.savefig('segmentations.png')
        plt.close()
    
    def slic(self, image):
        segments = slic(image,
                        n_segments=self.n_segments,
                        compactness=self.compactness)
        lst_masks = []
        mask_ids = np.unique(segments)
        for m in mask_ids:
            mask = np.zeros_like(segments)
            mask[segments==m] = 1
            lst_masks.append(mask.astype(bool))
        
        return lst_masks
    
    def felzenszwalb(self, image):
        segments = felzenszwalb(image,
                                scale=self.scale,
                                sigma=self.sigma,
                                min_size=self.min_size)
        lst_masks = []
        mask_ids = np.unique(segments)
        for m in mask_ids:
            mask = np.zeros_like(segments)
            mask[segments==m] = 1
            lst_masks.append(mask.astype(bool))
        
        return lst_masks
    
    def sam(self, image):
        masks = self.mask_generator.generate(image)
        
        lst_masks = [m['segmentation'] for m in masks]
        
        bbox = [m['bbox'] for m in masks]
        
        return lst_masks, bbox
    
    def canny(self, image):
        image = rgb2gray(image)
        edges = canny(image, sigma=4)
        
        
        filled_edges = morphology.binary_closing(edges, morphology.disk(2))
        cleared_edges = clear_border(filled_edges)
        
        labels = label(cleared_edges)
        
        label_idxs = np.unique(labels)
        lst_masks = []
        for idx in label_idxs:
            mask = np.zeros_like(labels)
            mask[labels==idx] = 1
            lst_masks.append(mask.astype(bool))
        
        return lst_masks
        
    def threshold_otsu(self, image):
        image = rgb2gray(image)
        
        value = threshold_otsu(image)
        binary_im = image>value
        
        filled_im = morphology.remove_small_holes(binary_im, area_threshold=5)
        
        clean_im = morphology.remove_small_objects(filled_im, min_size=5)
        
        labels = label(clean_im)
        
        label_idxs = np.unique(labels)
        lst_masks = []
        for idx in label_idxs:
            mask = np.zeros_like(labels)
            mask[labels==idx] = 1
            lst_masks.append(mask.astype(bool))
        
        return lst_masks
        
        
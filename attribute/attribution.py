import captum.attr as attr
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
from attribute.rnd_wrapper import RNDModelWrapper
import torch
from attribute.segmentor import Segmentor
from skimage.measure import regionprops
import cv2

TYPES = [
    "deep_lift_shap",
    "deep_lift",
    "ig",
    "saliency",
    "gradient_shap",
]

class Attribute():
    def __init__(self,
                 predict_model,
                 target_model,
                 device,
                 attribution_type="deep_lift_shap",
                 segmentation_type="sam",
                 **kwargs):
        model_wrapper = RNDModelWrapper(predict_model, target_model)
        self.ave_baselines = None
        self.attribution = self._get_attribution(attribution_type, model_wrapper)
        self.device = device
        self.type = attribution_type
        self.segmentor = Segmentor(segmentation_type, **kwargs)
        
    def _get_attribution(self, type, model_wrapper):
        assert type in TYPES
        if type == "deep_lift_shap":
            self.ave_baselines = False
            return attr.DeepLiftShap(model_wrapper)
        if type == "deep_lift":
            self.ave_baselines = True
            return attr.DeepLift(model_wrapper)
        if type == "ig":
            self.ave_baselines = True
            return attr.IntegratedGradients(model_wrapper)
        if type == "saliency":
            self.ave_baselines = False
            return attr.Saliency(model_wrapper)
        if type == "gradient_shap":
            self.ave_baselines = False
            return attr.GradientShap(model_wrapper)

    def analyze_state(self, state, baselines, plot=False):
        if self.ave_baselines:
            baselines = torch.mean(baselines, dim=0, keepdim=True)
        
        if self.type == "saliency":
            attribution_map = self.attribution.attribute(state).detach().cpu().numpy()
        else:
            attribution_map = self.attribution.attribute(state,
                                                         baselines=baselines).detach().cpu().numpy()
        
        attribution_img = (attribution_map-np.min(attribution_map))/(np.max(attribution_map)-np.min(attribution_map))
        
        attribution_img = np.squeeze(attribution_img)
        
        attribution_map = np.squeeze(attribution_map)
        
        state_seg = cv2.cvtColor(state.squeeze().cpu().numpy(), cv2.COLOR_GRAY2RGB)
        
        segments, bbox = self.segmentor.segment(state_seg, plot=True)
        
        ave_att = []
        segment_img = np.zeros_like(segments[0], dtype=int)
        for idx, m in enumerate(segments):            
            segment_img[m] = idx+1
            ave_att.append(np.mean(attribution_map[m]))
        
        max_mask = segments[np.argmax(ave_att)]
        # bbox in XYWH format
        max_bbox = bbox[np.argmax(ave_att)]
        
        print(max_bbox)
        
        plt.imshow(max_mask)
        plt.colorbar()
        plt.savefig("max_mask.png")
        plt.clf()
        
        if plot:
            
            plt_state = state.cpu().numpy()
            plt_state = np.squeeze(plt_state)
            plt_state = np.expand_dims(plt_state, axis=-1)
            attribution_map = np.expand_dims(attribution_map, axis=-1)
            
            fig,_ = viz.visualize_image_attr(attribution_map, 
                                            plt_state, 
                                            method="blended_heat_map",
                                            sign="all")
            
            fig.savefig("attribution.png")
            
            
            ave_min = np.min(ave_att)
            ave_max = np.max(ave_att)
            normalized_ave = (ave_att-ave_min)/(ave_max-ave_min)
            
            colormap = plt.cm.get_cmap('viridis')
            coloured_segments = np.zeros((84,84,3))
            
            for region in regionprops(segment_img):
                color = colormap(normalized_ave[region.label - 1])
                for coords in region.coords:
                    coloured_segments[coords[0], coords[1], :] = color[:3]
            
            
            
            plt.imshow(coloured_segments)
            plt.colorbar()
            plt.savefig("ave_att.png")
            plt.clf()

        
        return max_mask
        
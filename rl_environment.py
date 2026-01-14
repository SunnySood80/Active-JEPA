import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
import gym
from gym import spaces


class MaskingEnv(gym.Env):
    def __init__(self, fi1_shape: tuple, mask_ratio: float = 0.5, patch_size: int = 8, feature_dim=None, device='cuda'):
        super(MaskingEnv, self).__init__()

        B, D, H8, W8 = fi1_shape  # e.g., [B, D, 28, 28]

        self.B = B
        self.D = D
        self.H8 = H8
        self.W8 = W8
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.device = device

        self.n_patches_h = H8 // patch_size
        self.n_patches_w = W8 // patch_size
        self.total_patches = self.n_patches_h * self.n_patches_w
        self.num_masked = int(mask_ratio * self.total_patches)
        
        self.action_space = spaces.Discrete(self.total_patches)

        self.feature_dim = feature_dim

        # per patch features
        self.observation_space = spaces.Dict({
            'patch_mask': spaces.Box(low=0.0, high=1.0, shape=(self.total_patches,), dtype=np.float32),
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_patches, self.feature_dim), dtype=np.float32)
        })

        self.current_mask = None
        self.masked_count = 0
        self.state = None
        self.masked_patches = None
        
        self.step_count = 0
        self.max_steps = self.num_masked
    
    def reset(self, image_features=None):
        self.current_mask = torch.zeros(self.H8 * self.W8, dtype=torch.bool, device=self.device)
        self.masked_count = 0
        self.masked_patches = set()
        self.step_count = 0
        
        if image_features is not None:
            self.features = image_features  # [64, 768]
        else:
            self.features = torch.zeros(self.ptotal_patches, self.feature_dim, device=self.device)
        
        # Initialize mask tensor
        self.mask = torch.zeros(self.total_patches, dtype=torch.float32, device=self.device)
        
        # Return dict state
        state = {
            'mask': self.mask,
            'features': self.features
        }
        
        return state


    def step(self, action):
        self.step_count += 1

        ph = action // self.n_patches_w
        pw = action % self.n_patches_w
        
        h_start = ph * self.patch_size
        h_end = min(h_start + self.patch_size, self.H8)
        w_start = pw * self.patch_size  
        w_end = min(w_start + self.patch_size, self.W8)
        
        for h in range(h_start, h_end):
            for w in range(w_start, w_end):
                self.current_mask[h * self.W8 + w] = True
                
        self.masked_count += 1
        self.masked_patches.add(action)
        
        # Update the mask tensor
        self.mask[action] = 1.0  # ← Mark this patch as masked
        
        done = (self.masked_count >= self.num_masked) or (self.step_count >= self.max_steps)
        
        if done and self.masked_count >= self.num_masked:
            steps_saved = self.max_steps - self.step_count
            efficiency_bonus = (steps_saved / self.max_steps) * 10.0
            reward = efficiency_bonus
        else:
            reward = 0.0
        
        # Return dict state instead of flat
        state = {
            'mask': self.mask,           # [64]
            'features': self.features    # [64, 768]
        }
        
        return state, reward, done, {}

    def get_final_mask(self):
        return self.current_mask


    def get_available_actions(self):

        available = []
        for action in range(self.total_patches):
            if action not in self.masked_patches:
                available.append(action)
        return available
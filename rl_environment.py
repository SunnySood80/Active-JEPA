import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import math
import gym
from gym import spaces


class MaskingEnv(gym.Env):
    def __init__(self, fi1_shape: tuple, mask_ratio: float = 0.5, patch_size: int = 8, feature_dim = 768, device='cuda'):
        super(MaskingEnv, self).__init__()
        
        self.feature_dim = feature_dim
        self.global_features = None

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

        print(f"here is the type of self.total_patches: {type(self.total_patches)}")
        print(f"here is the type of self.feature_dim: {type(self.feature_dim)}")


        state_size = self.total_patches + int(self.feature_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

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
        
        # Store global features
        if image_features is not None:
            self.global_features = image_features.mean(dim=0)
        else:
            self.global_features = torch.zeros(self.feature_dim, device=self.device)
        
        # Convert masked_patches set to tensor (all zeros at start)
        patch_mask = torch.zeros(self.total_patches, dtype=torch.float32, device=self.device)
        
        state = torch.cat([patch_mask, self.global_features])

        self.state = state.cpu().numpy()

        return self.state


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
        done = (self.masked_count >= self.num_masked) or (self.step_count >= self.max_steps)

        if done and self.masked_count >= self.num_masked:

            actual_pixels = self.current_mask.sum().item()
            expected = self.masked_count * self.patch_size * self.patch_size
                
            steps_saved = self.max_steps - self.step_count
            efficiency_bonus = (steps_saved / self.max_steps) * 10.0
            reward = efficiency_bonus
        else:
            reward = 0.0

        # Convert masked_patches set to tensor
        patch_mask = torch.zeros(self.total_patches, dtype=torch.float32, device=self.device)
        if len(self.masked_patches) > 0:
            indices = torch.tensor(list(self.masked_patches), device=self.device)
            patch_mask[indices] = 1.0
        
        state = torch.cat([patch_mask, self.global_features])

        self.state = state.cpu().numpy()
    
        return self.state, reward, done, {}

    def get_final_mask(self):
        return self.current_mask


    def get_available_actions(self):

        available = []
        for action in range(self.total_patches):
            if action not in self.masked_patches:
                available.append(action)
        return available

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict
from rl_environment import MaskingEnv


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=None, action_dim=None, hidden_dim=None):  # Changed obs_shape to input_dim
        super().__init__()

        if input_dim is not None and action_dim is not None and hidden_dim is not None:
            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
        else:
            raise ValueError("input_dim, action_dim, and hidden_dim must be provided")

        self.shared = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        elif state.dim() == 2:
            squeeze_output = False
        else:
            raise ValueError(f"Invalid state dimension: {state.dim()}")
            
        batch_size = state.shape[0]
        state_flat = state.reshape(batch_size, -1)
        
        features = self.shared(state_flat)
        
        action_logits = self.policy_head(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        value = self.value_head(features)
        
        if squeeze_output:
            action_probs = action_probs.squeeze(0)
            value = value.squeeze(0)
            
        return action_probs, value
    
    def get_action(self, state, deterministic=False, generator=None):
        with torch.no_grad():
            action_probs, value = self.forward(state)
        return action_probs, value.item()  # Don't sample here, just return probs
    

class SmartPolicyNetwork(nn.Module):
    def __init__(self, feature_dim=768, action_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # Score each patch using BOTH features AND mask state
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim + 1, 256),  # +1 for mask input
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Value head (needs to see all features + mask state)
        self.value_head = nn.Sequential(
            nn.Linear((feature_dim + 1) * action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state_dict):
        # state_dict: {'mask': [B, 64], 'features': [B, 64, 768]}
        features = state_dict['features']  # [B, 64, 768]
        mask = state_dict['mask']  # [B, 64]
        
        # Concatenate mask as extra feature per patch
        mask_expanded = mask.unsqueeze(-1)  # [B, 64, 1]
        features_with_mask = torch.cat([features, mask_expanded], dim=-1)  # [B, 64, 769]
        
        # Score each patch (now considering mask state)
        action_logits = self.scorer(features_with_mask).squeeze(-1)  # [B, 64]
        
        # CRITICAL: Mask out already-selected patches (set logits to -inf)
        action_logits = action_logits.masked_fill(mask.bool(), float('-inf'))
        
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Value estimate (also uses mask info)
        flat_features = features_with_mask.reshape(features.shape[0], -1)  # [B, 64*769]
        value = self.value_head(flat_features)  # [B, 1]
        
        return action_probs, value
    
    def get_action(self, state_dict, deterministic=False, generator=None):
        with torch.no_grad():
            action_probs, value = self.forward(state_dict)
        return action_probs, value

class PPO:
    def __init__(
        self,
        action_dim = 64,
        feature_dim = 768,
        lr: float = 1e-4, # was 3e-4
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.3,
        value_coef: float = 0.5,
        entropy_coef: float = 0.1, # was 0.01
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = 'cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.policy = SmartPolicyNetwork(feature_dim=feature_dim, action_dim=action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Create generators for reproducible operations (seeded by utils.set_seed())
        from utils import get_seed
        # CUDA generator for multinomial sampling (on GPU)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(get_seed())
        # CPU generator for torch.randperm (requires CPU device)
        self.cpu_generator = torch.Generator(device='cpu')
        self.cpu_generator.manual_seed(get_seed())

        
        
    def select_action(self, state, available_actions, deterministic=False):
        # state is dict: {'mask': tensor[64], 'features': tensor[64, 768]}
        # They're already tensors from the environment!
        
        # Just ensure they're on the right device and have batch dim
        if isinstance(state['mask'], torch.Tensor):
            mask = state['mask'].unsqueeze(0).to(self.device)
            features = state['features'].unsqueeze(0).to(self.device)
        else:
            # If they're numpy arrays (shouldn't be with new env)
            mask = torch.FloatTensor(state['mask']).unsqueeze(0).to(self.device)
            features = torch.FloatTensor(state['features']).unsqueeze(0).to(self.device)
        
        state_dict = {
            'mask': mask,      # [1, 64]
            'features': features  # [1, 64, 768]
        }
        
        # Get probabilities from policy
        action_probs, value = self.policy.get_action(state_dict, deterministic, generator=self.generator)
        
        # Remove batch dimension
        action_probs = action_probs.squeeze(0)  # [64]
        value = value.squeeze(0).item()  # scalar
        
        # Clamp to prevent extreme values
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        
        # Normalize
        action_probs = action_probs / (action_probs.sum() + 1e-10)
        action_probs = torch.clamp(action_probs, min=1e-8)
        action_probs = action_probs / action_probs.sum()
        
        # Create distribution
        dist = Categorical(action_probs)
        
        # Sample action
        if deterministic:
            action = torch.argmax(action_probs).item()
        else:
            action = dist.sample().item()
        
        log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
        
        return action, log_prob, value
    
    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        advantages = []
        gae = 0
        
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                
            advantages.insert(0, gae)
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values[:-1], dtype=np.float32)
        
        return advantages, returns
    
    def train_on_batch(
        self,
        states: List[Dict],
        actions: List[int],
        old_log_probs: List[float],
        advantages: np.ndarray,
        returns: np.ndarray
    ):
        masks = torch.stack([state['mask'] for state in states]).to(self.device)  # [N, 64]
        features = torch.stack([state['features'] for state in states]).to(self.device)
        states_dict = {
            'mask': masks,
            'features': features
        }
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # ADD: Check for NaN in inputs
        if torch.isnan(advantages_tensor).any() or torch.isnan(returns_tensor).any():
            print("WARNING: NaN in advantages or returns! Skipping batch")
            return
        
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Metrics tracking
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'kl_divergence': 0.0,
            'clip_fraction': 0.0,
            'approx_kl': 0.0,
            'grad_norm_policy': 0.0,
            'grad_norm_value': 0.0,
        }
            
        dataset_size = len(states)
        for epoch in range(self.n_epochs):
            indices = torch.randperm(dataset_size, generator=self.cpu_generator)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = {
                    'mask': states_dict['mask'][batch_indices],
                    'features': states_dict['features'][batch_indices]
                }
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                action_probs, values = self.policy(batch_states)
                
                # Clamp action_probs to prevent NaN
                action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                ratio = torch.clamp(ratio, 0.01, 100.0)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                values_flat = values.squeeze(-1)
                if values_flat.shape != batch_returns.shape:
                    batch_returns = batch_returns.view_as(values_flat)
                value_loss = F.mse_loss(values_flat, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Check for NaN in loss
                if torch.isnan(loss).any():
                    print(f"WARNING: NaN detected in loss!")
                    continue
                
                # ========== INSERT THESE CALCULATIONS HERE ==========
                # Calculate KL metrics BEFORE accumulating
                with torch.no_grad():
                    log_ratio = log_probs - batch_old_log_probs
                    kl_divergence = log_ratio.mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                # ====================================================
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN in gradients
                has_nan_grad = False
                for name, param in self.policy.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"WARNING: NaN gradient in {name}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    self.optimizer.zero_grad()
                    continue
                
                # ========== INSERT GRADIENT NORM CALCULATIONS HERE ==========
                # Calculate grad norms AFTER backward, BEFORE optimizer.step()
                policy_params = [p for n, p in self.policy.named_parameters() if 'scorer' in n or 'policy' in n.lower()]
                value_params = [p for n, p in self.policy.named_parameters() if 'value' in n.lower()]
                
                if policy_params:
                    grad_norm_policy = torch.nn.utils.clip_grad_norm_(policy_params, self.max_grad_norm)
                else:
                    grad_norm_policy = torch.tensor(0.0)
                
                if value_params:
                    grad_norm_value = torch.nn.utils.clip_grad_norm_(value_params, self.max_grad_norm)
                else:
                    grad_norm_value = torch.tensor(0.0)
                
                # Also clip all params together
                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # ================================================================
                
                self.optimizer.step()
                
                # NOW accumulate metrics (all variables exist now!)
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['total_loss'] += loss.item()
                metrics['kl_divergence'] += kl_divergence.item()
                metrics['approx_kl'] += approx_kl.item()
                metrics['clip_fraction'] += clip_fraction.item()
                metrics['grad_norm_policy'] += grad_norm_policy.item() if isinstance(grad_norm_policy, torch.Tensor) else grad_norm_policy
                metrics['grad_norm_value'] += grad_norm_value.item() if isinstance(grad_norm_value, torch.Tensor) else grad_norm_value
       
        # Average metrics over all updates
        num_updates = (dataset_size // self.batch_size) * self.n_epochs
        
        for key in metrics:
            metrics[key] /= max(1, num_updates)
        
        # Add additional stats
        metrics['advantage_mean'] = advantages_tensor.mean().item()
        metrics['advantage_std'] = advantages_tensor.std().item()
        metrics['return_mean'] = returns_tensor.mean().item()
        metrics['return_std'] = returns_tensor.std().item()
        
        # Explained variance (using final batch values)
        values_final = values_flat.detach().cpu().numpy()
        returns_final = batch_returns.cpu().numpy()
        var_y = np.var(returns_final)
        if var_y > 1e-8:
            explained_var = 1 - np.var(returns_final - values_final) / var_y
        else:
            explained_var = 0.0
        metrics['explained_variance'] = explained_var
        
        return metrics
    
    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def get(self):
        return {
            'states': self.states,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'values': self.values,
            'rewards': self.rewards,
            'dones': self.dones
        }
    
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def __len__(self):
        return len(self.states)
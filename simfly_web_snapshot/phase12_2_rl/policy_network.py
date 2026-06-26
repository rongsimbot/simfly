#!/usr/bin/env python3
"""
PPO Policy Network — lightweight MLP for per-joint torque scaling.

State:  36 torque values + 7 scalar metrics -> 43 dims
Action: 36 per-joint multipliers in [0.5, 2.0] (Gaussian policy)
Value:  scalar (critic)

The policy does NOT control the fly — it only tunes the motor decoder interface.
The connectome provides ALL neural signals.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOPolicySmall(nn.Module):
    """Lightweight PPO policy: 128 hidden, Tanh activations."""
    
    def __init__(self, state_dim=43, action_dim=36, hidden=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.shared = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(hidden, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden, hidden // 2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden // 2, 1), std=1.0),
        )
    
    def get_value(self, state):
        return self.critic(self.shared(state))
    
    def get_action_and_value(self, state, action=None):
        features = self.shared(state)
        value = self.critic(features)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd.clamp(-3, 2))
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value
    
    def get_action(self, state):
        with torch.no_grad():
            features = self.shared(state)
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_logstd.clamp(-3, 2))
            dist = Normal(action_mean, action_std)
            return dist.sample()


def create_policy(network_type='small', state_dim=43, action_dim=36):
    if network_type == 'small':
        return PPOPolicySmall(state_dim=state_dim, action_dim=action_dim)
    else:
        return PPOPolicySmall(state_dim=state_dim, action_dim=action_dim, hidden=256)


if __name__ == '__main__':
    policy = create_policy('small')
    state = torch.randn(1, 43)
    action, log_prob, entropy, value = policy.get_action_and_value(state)
    n_params = sum(p.numel() for p in policy.parameters())
    print("Policy: %d params, Action shape: %s, Value: %.3f, Entropy: %.3f" % (
        n_params, str(action.shape), value.item(), entropy.item()))
    print("Smoke test PASSED")

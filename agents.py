# built-in modules
from typing import Optional

# 3rd-party modules
import torch
import torch.nn as nn
from torch.distributions import Normal


class SACAgent:
    backup_entropy: bool = True # TODO: check if this is correct
    def __init__(
        self,
        model: nn.Module,
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """
        print(f'SACAgent init... ', end='', flush=True)
        self.model = model

        self.target_entropy = target_entropy
        if target_entropy is None:
            self.target_entropy = -self.model.action_dim / 2

        self.discount = discount
        self.tau = tau

        print('Done!')

    def update_actor(self, batch: dict):
        mean, std = self.model.actor_forward(batch['observations'])
        
        dist = Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        qs = self.model.critic_forward(batch['observations'], actions)
        q = torch.min(qs, dim=0).values # TODO: check if this is correct
        
        actor_loss = (
            log_probs * self.model.temperature() - q
        ).mean()

        self.model.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.model.actor_optimizer.step()

        return actor_loss, -log_probs.mean()

    def update_temperature(self, entropy: float):
        temperature = self.model.temperature()
        temp_loss = temperature * (entropy - self.target_entropy).mean()
        return temp_loss
        
    def update_critic(self, batch: dict):
        mean, std = self.model.actor_forward(batch['next_observations'])
        dist = Normal(mean, std)
        next_actions = dist.sample()

        next_qs = self.model.target_critic_forward(batch['next_observations'], next_actions)
        next_q = torch.min(next_qs, dim=0).values # TODO: check if this is correct

        target_q = batch['rewards'] + self.discount * (1 - batch['dones']) * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * (1 - batch['dones'])
                * self.model.temperature()
                * next_log_probs
            )
        
        qs = self.model.critic_forward(batch['observations'], batch['actions'])
        critic_loss = ((qs - target_q) ** 2).mean() # TODO: MAE vs. MSE

        self.model.critic_optimizer.zero_grad()
        critic_loss.backward()

        self.model.critic_optimizer.step()
        for target_param, param in zip(self.model.target_critic.parameters(), self.model.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss

    def update(self, batch: dict):
        _ = self.update_critic(batch)
        _, entropy = self.update_actor(batch)
        _ = self.update_temperature(entropy)

    def __call__(self, obs: dict):
        mean, std = self.model.actor_forward(obs)
        dist = Normal(mean, std)
        return dist.sample()
        
# built-in modules


# 3rd-party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.layer_norm = nn.LayerNorm(64) # RLPD 써야됨
        self.mean_head = nn.Linear(64, action_dim)
        self.std_head = nn.Linear(64, action_dim)

    def forward(self, embedding):
        x = F.relu(self.fc1(embedding))
        x = F.relu(self.fc2(x))
        x = self.layer_norm(x)
        mean = self.mean_head(x)
        std = self.std_head(x)
        std = F.softplus(std) + 1e-5
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, output_dim: int):
        super(Critic, self).__init__()
        self.action_hidden_dim = 16
        self.action_embedder = nn.Linear(action_dim, self.action_hidden_dim)

        self.fc1 = nn.Linear(state_dim + self.action_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, output_dim)

    def forward(self, embedding, actions):
        action_embedding = F.relu(self.action_embedder(actions))
        
        x = torch.cat((embedding, action_embedding), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x

class Temperature(nn.Module):
    def __init__(self, init_temperature: float=1.0):
        super(Temperature, self).__init__()

        self.temperature = nn.Parameter(torch.tensor(init_temperature))
    
    def forward(self):
        x = torch.log(self.temperature)
        return torch.exp(x)

class Model(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, output_dim: int):
        super(Model, self).__init__()
        print(f'Model init... ', end='', flush=True)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.embed_dim = 128

        self.vision_embedder = torchvision.models.resnet18(pretrained=True)
        self.vision_embedder.fc = nn.Linear(self.vision_embedder.fc.in_features, self.embed_dim)
        self.state_embedder = nn.Linear(state_dim, self.embed_dim)

        self.actor = Actor(state_dim=(self.embed_dim * 2), action_dim=action_dim)
        self.critic = Critic(state_dim=(self.embed_dim * 2), action_dim=action_dim, output_dim=output_dim)
        self.target_critic = Critic(state_dim=(self.embed_dim * 2), action_dim=action_dim, output_dim=output_dim)
        self.temperature = Temperature()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.temperature_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=3e-4)
        print('Done!')

    def embed_observations(self, observations: dict):
        images = observations['images']
        states = observations['states']

        image_embedding = self.vision_embedder(images)
        state_embedding = self.state_embedder(states)
        embedding = torch.cat([image_embedding.squeeze(), state_embedding], dim=-1)

        return embedding

    def actor_forward(self, observations: dict):
        embedding = self.embed_observations(observations)

        return self.actor(embedding)
    
    def critic_forward(self, observations: dict, actions: torch.Tensor):
        embedding = self.embed_observations(observations)

        return self.critic(embedding, actions)
    
    def target_critic_forward(self, observations: dict, actions: torch.Tensor):
        embedding = self.embed_observations(observations)

        return self.target_critic(embedding, actions)

    def forward(self, batch: dict):
        pass
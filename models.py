# built-in modules


# 3rd-party modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms

class Actor(nn.Module):
    def __init__(self, state_embed_dim: int, action_dim: int):
        super(Actor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_embed_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
        )

        self.mean_head = nn.Linear(64, action_dim)
        self.std_head = nn.Linear(64, action_dim)

    def forward(self, embedding):
        x = self.mlp(embedding)

        mean = self.mean_head(x)

        std = self.std_head(x)
        std = F.softplus(std) + 1e-5

        return mean, std

class Critic(nn.Module):
    def __init__(self, state_embed_dim: int, action_dim: int, output_dim: int):
        super(Critic, self).__init__()
        
        self.action_embed_dim = 128
        self.action_embedder = nn.Sequential(
            nn.Linear(action_dim, self.action_embed_dim),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(state_embed_dim + self.action_embed_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            
            nn.Linear(64, output_dim)
        )

    def forward(self, state_embedding, actions):
        action_embedding = self.action_embedder(actions)
        
        x = torch.cat((state_embedding, action_embedding), dim=-1)
        
        x = self.mlp(x)

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

        self.vision_embedder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.vision_embedder.fc = nn.Linear(self.vision_embedder.fc.in_features, self.embed_dim)
        self.state_embedder = nn.Linear(state_dim, self.embed_dim)

        self.actor = Actor(state_embed_dim=(self.embed_dim * 2), action_dim=action_dim)
        self.critic = Critic(state_embed_dim=(self.embed_dim * 2), action_dim=action_dim, output_dim=output_dim)
        self.target_critic = Critic(state_embed_dim=(self.embed_dim * 2), action_dim=action_dim, output_dim=output_dim)
        self.temperature = Temperature()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.temperature_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=3e-4)

        self.img_preprocessor = transforms.Compose([
            transforms.Normalize(                    # ImageNet의 평균과 표준편차로 정규화
                mean=[0.485, 0.456, 0.406],          # ImageNet 평균 (RGB 순)
                std=[0.229, 0.224, 0.225]            # ImageNet 표준편차 (RGB 순)
            ),
        ])

        print('Done!')

    def img_preprocess(self, img_batch):
        return self.img_preprocessor(img_batch)


    def embed_observations(self, observations: dict):
        images = observations['images']
        states = observations['states']
        
        images = self.img_preprocess(images)
        states = states.to(torch.float32)

        image_embedding = self.vision_embedder(images).to(torch.float32)
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
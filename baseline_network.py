import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch
from general import batch_iterator

class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """
    def __init__(self, env, config):
        """
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate

        self.network = build_mlp(self.env.observation_space.shape[0],1,self.config.n_layers,self.config.layer_size)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters() ,lr=self.lr)


    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        """
       

        output = self.network(observations)
        output = torch.squeeze(output)
       
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)] input
        Returns:
            advantages: np.array of shape [batch size]

        """
        observations = np2torch(observations)
        
        with torch.no_grad():
            self.baseline = self(observations)

        baseline = self.baseline.to('cpu').numpy()
        advantages = returns - baseline

        
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]

        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        
        '''
        mini_batch = batch_iterator(observations, returns)

        for mb in mini_batch:
            self.optimizer.zero_grad()
            self.baseline = self(mb[0])
            loss = self.mse_loss(self.baseline, mb[1])
            loss.backward()
            self.optimizer.step()
        '''
        self.optimizer.zero_grad()
        self.baseline = self(observations)
        loss = self.mse_loss(self.baseline, returns)
        loss.backward()
        self.optimizer.step()
        

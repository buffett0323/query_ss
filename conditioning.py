import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FiLM(nn.Module):
    def __init__(
        self,
        cond_embedding_dim: int,
        channels: int,
        additive: bool = True,
        multiplicative: bool = False,
        depth: int = 1,
        activation: str = "ELU",
    ):
        super().__init__()

        self.additive = additive
        self.multiplicative = multiplicative
        self.depth = depth

        # Define activation function
        Activation = getattr(nn, activation)

        if self.multiplicative:
            if depth == 1:
                self.gamma = nn.Linear(cond_embedding_dim, channels)
            else:
                layers = [nn.Linear(cond_embedding_dim, channels)]
                for _ in range(depth - 1):
                    layers += [Activation(), nn.Linear(channels, channels)]
                self.gamma = nn.Sequential(*layers)
        else:
            self.gamma = None

        if self.additive:
            if depth == 1:
                self.beta = nn.Linear(cond_embedding_dim, channels)
            else:
                layers = [nn.Linear(cond_embedding_dim, channels)]
                for _ in range(depth - 1):
                    layers += [Activation(), nn.Linear(channels, channels)]
                self.beta = nn.Sequential(*layers)
        else:
            self.beta = None

    def forward(self, x, w):
        # Apply gamma (multiplicative) modulation
        if self.multiplicative:
            gamma = self.gamma(w)
            gamma = gamma[:, :, None, None]  # Reshape for broadcasting over spatial dimensions
            x = gamma * x

        # Apply beta (additive) modulation
        if self.additive:
            beta = self.beta(w)
            beta = beta[:, :, None, None]  # Reshape for broadcasting over spatial dimensions
            x = x + beta

        return x
    

# Define the Hypernetwork for generating FiLM parameters
class FiLMHyperNetwork(nn.Module):
    def __init__(self, query_dim, channels, depth=1, activation='ELU'):
        super(FiLMHyperNetwork, self).__init__()
        self.depth = depth
        self.activation = getattr(nn, activation)
        
        # Defining a multi-layer perceptron to generate gamma and beta
        layers = [nn.Linear(query_dim, channels)]
        for _ in range(depth - 1):
            layers += [self.activation(), nn.Linear(channels, channels)]
        
        self.fc_gamma = nn.Sequential(*layers)
        self.fc_beta = nn.Sequential(*layers)

    def forward(self, query):
        # Generate gamma and beta from the query
        gamma = self.fc_gamma(query)
        beta = self.fc_beta(query)
        return gamma, beta



# Modified FiLM class to use hypernetwork-generated parameters
class Hyper_FiLM(nn.Module):
    def __init__(
        self,
        cond_embedding_dim: int,
        channels: int,
        hypernetwork: FiLMHyperNetwork,
        additive: bool = True,
        multiplicative: bool = False,
    ):
        super().__init__()
        
        self.additive = additive
        self.multiplicative = multiplicative
        self.hypernetwork = hypernetwork

    def forward(self, x, query):
        # Get dynamic gamma and beta from hypernetwork
        gamma, beta = self.hypernetwork(query)
        
        # Apply gamma (multiplicative) modulation
        if self.multiplicative:
            gamma = gamma[:, :, None, None]  # Reshape for broadcasting
            x = gamma * x

        # Apply beta (additive) modulation
        if self.additive:
            beta = beta[:, :, None, None]  # Reshape for broadcasting
            x = x + beta

        return x





if __name__ == "__main__":
    B = 8            # Batch size
    T = 36           # Temporal dimension of mixture_input
    F = 64           # Frequency dimension of mixture_input
    channels = 512   # Number of channels/features in mixture_input
    query_dim = 768  # Dimension of the query input (condition)

    # Instantiate the hypernetwork and FiLM model
    hypernet = FiLMHyperNetwork(query_dim=query_dim, channels=channels, depth=2, activation="ELU")
    film_layer = Hyper_FiLM(cond_embedding_dim=query_dim, channels=channels, hypernetwork=hypernet, additive=True, multiplicative=True)

    # Dummy inputs
    mixture_input = torch.randn(B, channels, F, T)  # Processed mixture audio input with shape (BS, 512, 64, 36)
    query_input = torch.randn(B, query_dim)         # Query stem input with shape (BS, 512)

    # Apply the FiLM layer
    output = film_layer(mixture_input, query_input)

    print("Output shape:", output.shape)  # Expected shape: (8, 512, 64, 36)
    
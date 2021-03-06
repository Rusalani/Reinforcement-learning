import torch
import torch.nn as nn


def build_mlp(
          input_size,
          output_size,
          n_layers,
          size):
    """
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    """

    
    model = []
    
    model.append(nn.Linear(input_size,size))
    model.append(nn.ReLU())
    for i in range(n_layers-1):
        model.append(nn.Linear(size,size))
        model.append(nn.ReLU())
        
    model.append(nn.Linear(size,output_size))
    model = nn.Sequential(*model)
    return model



device = torch.device( 'cpu')

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x

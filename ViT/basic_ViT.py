 """
    Implementation of the basic ViT from Tim Nguyen's blog

    Taken from https://github.com/tintn/vision-transformer-from-scratch/blob/main/vit.py
    """

import math
import torch import nn 


class NewGELUActivation(nn.Module):

  def forward(self,input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):

  def __init__(self,config):
    super().__init__()
    self.image_size = config["image_size"]
    self.patch_size = config["patch_size"]
    self.num_channels = config["num_channels"]
    self.hidden_size = config["hidden_size"]

    self.num_patches = (self.image_size // self.patch_size) ** 2
    self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size = self.patch_size, stride = self.patch_size)

  def forward(self,x):
    x = self.projection(x)
    x = x.flatten(2).transpose(1,2)
    return x

class Embeddings(nn.Module):

  def __init__(self,config):
    super().__init__()
    self.config = config
    self.patch_embeddings = PatchEmbeddings(config)
    
    
  
  

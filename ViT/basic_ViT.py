 """
    Implementation of the basic ViT from Tim Nguyen's blog

    Taken from https://github.com/tintn/vision-transformer-from-scratch/blob/main/vit.py
    """

import math
import torch
from torch import nn
from torch import einsum


class NewGeluActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1 + torch.tanh(math.sqrt(2 / math.pi) * (input + 0.044715 * input ** 3)))
    

class PatchEmbedding(nn.Module):
    """
    Convert Input Image to Patch Embeddings

    Essentially break down images into smaller patches and we feed them (embed them) into a vector space
    """
  
    def __init__(self,config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["path_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        self.num_patches = (self.image_size // self.patch_size) ** 2

        #Layer to project the broken down patches into vectors of length hidden_size to put into the embedding space
        self.projection = nn.Conv2d(self.nim_channels, self.hidden_size, kernel_size = self.patch_size, stride=self.patch_size)

    def forward(self,x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1,2)
        return x


class AttentionHead(nn.Module):
    """
    Single Attention Head, will do Multi Attention Head as well 
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        #For the Attention Head we need to have the query, key and value proj layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        #refered to as omega in papers
        attention_scores  = torch.matmul(query, key.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)
    

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Implementation
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #will not be implementing qkv bias

        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_droupout_prob"],
            )
            self.heads.append(head)

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions = False):
        attention_outputs = [head(x) for head in self.heads]
        #concatenate outputs from the different attention heads
        attention_outputs = torch.cat([attention_output for attention_output, _ in attention_outputs], dim = -1)

        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_provs in attention_outputs], dim = 1)
            return (attention_output, attention_probs)
        
    
    
class SimplicalAttention(nn.Module):
    """
    Following James Clift's paper on 2-Simplical Attention

    This is not on Tim Nguyen's blog, but coded by me.
    Full credit goes to James Clift's Team.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        #For the Attention Head we need to have the query, key and value proj layers
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key_prime = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value_prime = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        key_prime = self.key_prime(x)
        value_prime = self.value_prime(x)

        #2-simplical Transformer use
        attention_scores = torch.einsum("bid,bjd,bkd->bijk", query, key, key_prime) / math.sqrt(self.attention_head_size)
        attention_scores = nn.functional.softmax(attention_scores, dim = -1)

        # Symmetry: attention_scores = (attention_scores + attention_scores.transpose(-2,-1)) / 2

        attention_scores = self.dropout(attention_scores)


        attention_output = torch.einsum("bijk,bjd,bkd->bid", attention_scores, value, value_prime)

        return (attention_output)
    

class MultiSimplicalAttention(nn.Module):

    def __init__(self, config):
         super().__init__()
         self.hidden_size = config["hidden_size"]
         self.num_attention_heads = config["num_attention_heads"]

         self.attention_head_size = self.hidden_size // self.num_attention_heads
         self.all_head_size = self.num_attention_heads * self.attention_head_size

         #No Bias
         self.heads = nn.ModuleList([
             SimplicalAttention(
                 self.hidden_size,
                 self.attention_head_size,
                 config["attention_probs_dropout_prob"]
             ) for _ in range(self.num_attention_heads)
         ])

         self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
         self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    def forward (self, x):
        head_outputs = [head(x) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)

        attention_output = self.output_projection(head_outputs)
        attention_output = self.output_dropout(head_outputs)

        return (attention_output)

    
class MLP(nn.Module):
    """
    MLP Layer
    """
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGeluActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)

        return x
    
class Block(nn.Module):
    "Single Transformer Block"

    def __init__(self, config):
        super().__init__()
        self.use_multiSimplical_attention = config.get("use_multiSimplical_attention", False)

        if self.use_multiSimplical_attention:
            self.attention = MultiSimplicalAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        
        self.layernorm1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm2 = nn.LayerNorm(config["hidden_size"])


    def forward(self, x):

        #Self Attention
        attention_output = self.attention(self.layernorm1(x))

        #skip connection
        x = x + attention_output

        #FF Network
        mlp_output = self.mlp(self.layernorm2(x))

        #Skip Connection
        x = x + mlp_output

        return (x)
    
class Encoder(nn.Module):
    "Simple Encoder Block, but not saving attention probs"
    def _init__(self, config):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        return x
    










    
  
  

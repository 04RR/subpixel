import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

# https://github.com/lucidrains/tab-transformer-pytorch


cont_mean_std = torch.randn(10, 2)

## EXPERMIENTAL ##
# If normal ML models don't give good results and if the dataset is big enogh to use TabTransformer.
# Use exisiting train function or make a new train function to train TabTransformer on the custom dataset.
# Try to maybe find methods to find the best parameters for TabTransformer for the given task.


model = TabTransformer(
    categories=(10, 5, 6, 5, 8),  # tuple containing the number of unique values within each category
    num_continuous=10,  # number of continuous values
    dim=32,  # dimension, paper set at 32
    dim_out=1,  # binary prediction, but could be anything
    depth=6,  # depth, paper recommended 6
    heads=8,  # heads, paper recommends 8
    attn_dropout=0.1,  # post-attention dropout
    ff_dropout=0.1,  # feed forward dropout
    mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
    mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
    continuous_mean_std=cont_mean_std,  # (optional) - normalize the continuous values before layer norm
)

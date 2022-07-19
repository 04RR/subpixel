from typing import Union
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import json
from torchinfo import summary
from vision.train import visionTrainer
from utils import findLR, find_batch_size
import numpy as np


class Model(nn.Module):
    '''
    Converts model architecture from JSON to a trainable model and has a fit function that can train the model on the given dataset when called.
    '''
    def __init__(self,model : nn.Module = None,path :str = 'arch.json') -> None:
        super(Model,self).__init__()
        if isinstance(model,nn.Module):
            self.pre_defined_model = True
            self.model = model
            return

        JSON_file = open(path,"r")
        arch = json.load(JSON_file)
        JSON_file.close()
        
        self.details = arch["details"]
        self.arch = arch["arch"]
        self.architecture = nn.ModuleList()

        for i in range(len(self.arch)):
            if self.arch[i]["code"] != i:
                
                raise json.JSONDecodeError("layers not arranged properly!!", "", -1)
            else:
                x = self.arch[i]["layer"]
                y = eval(x)
                self.architecture.append(y)
        
    
    def forward(self,*X):
        if self.pre_defined_model:
            return self.model(*X)

        outputs = []
        
        if len(X)!= self.details["num_inputs"]:

            l  = self.details["num_inputs"]
            raise RuntimeError(f"Expected {l} inputs, got {len(X)}.")
            
            return

        for i in range(len(self.arch)):

            if len(self.arch[i]["inputs"]) == 1:
                outputs.append(self.architecture[i](outputs[self.arch[i]["inputs"][0]] if self.arch[i]["inputs"][0] >= 0 else X[abs(self.arch[i]["inputs"][0]) - 1]))
            
            elif len(self.arch[i]["inputs"])>=1:
                x = torch.cat([outputs[j] if j>=0 else X[abs(j)-1] for j in self.arch[i]["inputs"]], self.arch[i]["cat_dim"])
                outputs.append(self.architecture[i](x))
        
        return [outputs[j] for j in self.details["outputs"]] if len(self.details["outputs"]) > 1 else outputs[self.details["outputs"][0]] 

    def fit(self,trainset : Union[str,nn.Module], loss_fun : nn.Module,optimizer : str , lr :int = None, mode :str = "classification",valset :nn.Module = None):
        '''
        Trains the model on the given trainset.

        trainset : str | nn.Module , if trainset is str should be path to dataset. Check Trainer documentation for more details.

        loss_fn : nn.Module , function to check the loss.

        optimizer : str, lowercase string specifying name of optimizer to be used.

        lr (optional) : int  , initial learning rate. If not specified, ideal initial LR is found and used (refer utild.findLR()).

        mode : str  , default "classification", specifies the mode of operation. takes any of ["classification", "detection", "segmentation"]

        valset (optional): nn.Module | None , default None, provides validation set. Note:- if trainset is str automatically valset is taken from directory structure. 
        '''

        self.trainer = visionTrainer(self, trainset= trainset, epochs= 10, learning_rate= lr, loss_fn= loss_fun, optimizer= optimizer, mode= mode, valset= valset)
        self.history = self.trainer.fit()
        return self.history

    def find_size(self):

        '''Finds the size occupied by the trainable model parameters in CUDA memory.
        
        Returns the total number of trainable parameters and the size occupied. 
        '''
        
        p_total = sum(p.numel() for p in self.parameters() if p.requires_grad) 
        bits = 32.

        mods = list(self.modules())
        for i in range(1,len(mods)):
            m = mods[i]
            p = list(m.parameters())
            sizes = []
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        total_bits = 0
        for i in range(len(sizes)):
            s = sizes[i]
            bits = np.prod(np.array(s))*bits
            total_bits += bits

        return p_total, total_bits

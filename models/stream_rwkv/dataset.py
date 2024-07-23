########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset



class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = []
        for d in os.listdir(args.data_file):
            if "-face.pth" in d:
                self.data.append(torch.load(f"{args.data_file}/{d}"))
                
    def __len__(self):
        return min(self.args.epoch_steps * self.args.micro_bsz, len(self.data))

    def __getitem__(self, idx):
        dix = self.data[idx]
        x = dix[:-1, :]
        y = dix[1:, :]

        return x, y

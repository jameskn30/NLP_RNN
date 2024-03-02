import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.neural_network import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision import transforms, utils
from torchvision.transforms import v2
from sklearn.metrics import *
from tqdm import tqdm

num_classes = 4

model = nn.Sequential(
    nn.LazyConv2d(16, 5),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Dropout(0.2),
    nn.LazyConv2d(32,5),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Dropout(0.2),
    nn.Flatten(),
    nn.LazyLinear(512),
    nn.ReLU(),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(num_classes),
)

import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision
from cyphercat.definitions import DATASETS_DIR, DATASPLITS_DIR
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
from .splitter import splitter, dataset_split


def Cifar10_preload_and_split(path=None, splits=[0.4, 0.1, 0.25, 0.25], transform=None):
    """Index and split CIFAR10 dataset.

    Args:
        path (string): Path to location containing dataset. If left as None
            will search default location 'DATASETS_DIR' specified in
            definitions.
        splits (list): list of fractional splits 

    Returns:
        dict(Dataframes): Dictionary containing the dataframes corresponding
            to each split inclduing metadata.

    Example:

    Todo:
        - Write Example.
        - More work on user specified splits.
    """

    if path is None:
        path = DATASETS_DIR
    index_file = os.path.join(path, 'cifar10.index.csv')

    indices = None
    if os.path.exists(index_file):
        index_csv = np.loadtxt(index_file)
        indices = torch.tensor(index_csv)
        print('Found predefined indexing file {}'.format(index_file))
    
    trainset = torchvision.datasets.CIFAR10(path, train=True, transform=transform[0], download=False)
    testset = torchvision.datasets.CIFAR10(path, train=False, transform=transform[0], download=False)
    fullset = ConcatDataset([trainset, testset])
    print('Initializing CIFAR10Dataset splits')
    
    # Currently five equal splits
    dset_size = fullset.cumulative_sizes[-1]
    int_splits = []
    for i in range(len(splits)):
        int_splits.append(int(dset_size * splits[i]))
    if sum(int_splits) < dset_size:
        rem = dset_size - sum(int_splits)
        int_splits[-1] += rem

    indices, splitsets = dataset_split(fullset, int_splits, indices=indices)

    if not os.path.exists(index_file):
        print('No predefined indexing file found, so index permutations saving to {}'.format(index_file))
        np.savetxt(index_file, indices.numpy(), fmt='%i', delimiter=',')

    print('Finished splitting data.')

    return splitsets

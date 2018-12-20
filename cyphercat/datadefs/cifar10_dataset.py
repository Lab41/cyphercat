import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision
from cyphercat.definitions import DATASETS_DIR, DATASPLITS_DIR
#from ..utils.file_utils import downloader, unpacker
from cyphercat.utils.file_utils import downloader, unpacker
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os
from .splitter import splitter, dataset_split


def Cifar10_preload_and_split(path=None, splits=[0.36, 0.1, 0.25, 0.25, 0.04]):
    """Index and split librispeech dataset.

    Args:
        path (string): Path to location containing dataset. If left as None
            will search default location 'DATASETS_DIR' specified in
            definitions.
        splits (dict): dictionary with {name:[fractions]} for a user specified
            split. The split will be saved to 'DATASPLITS_DIR' under 'name'

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
    
    trainset = torchvision.datasets.CIFAR10(path, train=True, transform=None, download=False)
    testset = torchvision.datasets.CIFAR10(path, train=False, transform=None, download=False)
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

    dfs = {}

    for split_i in range(len(splitsets)):
        dfs[split_i] = splitsets[split_i]

    ### OBSOLETE, but keep just in case right now
    ## Lists of images+labels defining CIFAR10 train/val sets 
    #orig_train_list = torchvision.datasets.CIFAR10.train_list
    #orig_test_list  = torchvision.datasets.CIFAR10.test_list
    ## Full dataset list for building splits 
    #full_batch_list = orig_train_list + orig_test_list
    ## Load all images into single list
    #features = []
    #labels = []
    #for batch in full_batch_list:
    #    bpath = os.path.join(path, batch[0])
    #    print(bpath)
    #    with open(bpath, mode='rb') as file:
    #        # Encoding type is 'latin1'
    #        ibatch = pickle.load(file, encoding='latin1')
    #        # Get features and labels
    #        ifeat = ibatch['data'].reshape((len(ibatch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    #        ilab = ibatch['labels']
    #    features.extend(ifeat)
    #    labels.extend(ilab)

    print('Finished splitting data.')

    return dfs


class CIFAR10Dataset(Dataset):
    """This class subclasses the torch.utils.data.Dataset.  Calling __getitem__
    will return the transformed librispeech audio sample and it's label

    # Args
        df (Dataframe): Dataframe with audiosample path and metadata.
        seconds (int): Minimum length of audio to include in the dataset. Any
            files smaller than this will be ignored or padded to this length.
        downsampling (int): Downsampling factor.
        label (string): One of {speaker, sex}. Whether to use sex or speaker
            ID as a label.
        stochastic (bool): If True then we will take a random fragment from
            each file of sufficient length. If False we will always take a
            fragment starting at the beginning of a file.
        pad (bool): Whether or not to pad samples with 0s to get them to the
            desired length. If `stochastic` is True then a random number of 0s
            will be appended/prepended to each side to pad the sequence to the
            desired length.
        cache: bool. Whether or not to use the cached index file
    """
    #def __init__(self,  df=None, transform=None, cache=True):
    #def __init__(self, data_struct=None, train_set=True, transform=None):
    #    print("TEST")
        
    def __init__(self,  df=None, transform=None, cache=True):

        self.pad = pad
        self.label = 'class_id'
        self.transform = transform
        
        # load df from splitting function
        self.df = df
        self.num_classes = len(self.df['speaker_id'].unique())
        
        # Convert arbitrary integer labels of dataset to ordered
        # 0-(num_classes - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i
                                   for i in range(self.num_classes())}  
        
        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()[self.label]
            
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = io.imread(img_path)
        label = self.people_to_idx[img_path.split('/')[-2]]

        if self.transform is not None:
            image = self.transform(image)

        if self.transform is not None:
            image = self.transform(image)
            
        return image, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['speaker_id'].unique())

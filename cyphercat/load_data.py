import os
import sys
import shutil
import numpy as np

from .utils.config_utils import DataStruct
from .utils.file_utils import downloader, unpacker

from skimage import io
from torch.utils.data.dataset import Dataset


class LFWDataset(Dataset):
    """
    Faces in the Wild specific dataset class.
    Includes indexing functionality.
    """
    def __init__(self, data_dir='', train_set=True, transform=None): 
        self.test_train_split = 0.8
        self.transform = transform
        n_classes, file_list, class_to_label = self.index(data_dir, train_set)
        self.n_classes = n_classes
        self.file_list = file_list
        self.people_to_idx = class_to_label
                
    def __len__(self): 
        return len(self.file_list)
        
    def __getitem__(self, idx): 
        img_path = self.file_list[idx]
        image = io.imread(img_path)
        label = self.people_to_idx[img_path.split('/')[-2]]
        
        if self.transform is not None: 
            image = self.transform(image)
        
        return image, label

    def index(self, data_dir, is_train_set):
        img_paths = []
        for p in os.listdir(data_dir): 
            for i in os.listdir(os.path.join(data_dir, p)): 
                img_paths.append(os.path.join(data_dir, p, i))
                
        class_list = []
        class_to_idx = {}
        k = 0 
        for i in img_paths: 
            name = i.split('/')[-2]
            if name not in class_to_idx: 
                class_list.append(name)
                class_to_idx[name] = k
                k += 1

        n_classes = len(class_list)
        
        img_paths = np.random.permutation(img_paths)
        
        dataset_size = len(img_paths)
        trainset_size = int(self.test_train_split * dataset_size)
        if is_train_set:
            file_list = img_paths[:trainset_size]
        else:
            file_list = img_paths[trainset_size:]

        return n_classes, file_list, class_to_idx 
        

def custom_preprocessor(out_dir=''):
    """
    Custom preprocessing functions for
    specific data sets.

    Parameters
    ----------
    out_dir   : string
                directory of unpacked data set
                    
    """

    # Get name of data set from output directory
    data_name = os.path.split(out_dir)[1]

    # For tiny-imagenet-200
    if 'tiny' in data_name.lower():

        # Structure the training, validation, and test data directories
        train_dir  = os.path.join(out_dir, 'train')
        class_dirs = [os.path.join(train_dir, o) for o in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, o))]

        for c in class_dirs:
            for f in os.listdir(os.path.join(c, 'images')):
                os.rename(os.path.join(c,'images', f), os.path.join(c, f))
            for d in os.listdir(c):
                if d.find("JPEG") == -1:
                    if os.path.isfile(os.path.join(c, d)):
                        os.remove(os.path.join(c, d))
                    elif os.path.isdir(os.path.join(c, d)):
                        os.rmdir(os.path.join(c, d))

        # Get validation annotations
        with open(os.path.join(out_dir, 'val/val_annotations.txt')) as f:
            content = f.readlines()

        for x in content:
            line = x.split()

            if not os.path.exists(os.path.join(out_dir, 'val/',line[1])):
                os.makedirs(os.path.join(out_dir, 'val/',line[1]))

            new_file_name = os.path.join(out_dir, 'val',line[1],line[0])
            old_file_name = os.path.join(out_dir, 'val/images',line[0])
            os.rename(old_file_name, new_file_name)


    # For LFW
    if 'lfw' in data_name.lower():

        lfw_dir    = out_dir + '_original/'
        os.rename(out_dir, lfw_dir)
        
        people_dir = os.listdir(lfw_dir)

        num_per_class = 20

        for p in people_dir:
            imgs = os.listdir(os.path.join(lfw_dir, p))
            if len(imgs) >= num_per_class:
                shutil.copytree(os.path.join(lfw_dir, p), os.path.join(out_dir, p))
    
    print('{} successfully downloaded and preprocessed.'.format(data_name))



def prep_data(dataset_config=None):
    """
    Function to prepare data set
    based on input configuration

    Parameters
    ----------
    dataset_config  : dictionary
                      parameters from 'data' field
                      of global yaml configuration file
    """
    

    data_struct  = DataStruct(dataset_config)

    data_name    = data_struct.name
    datasets_dir = data_struct.data_path
    out_dir      = data_struct.save_path

    # If dataset already downloaded an unpacked, do nothing
    if os.path.isdir(out_dir):
        print('{} already downloaded, unpacked and processed.'.format(data_name))
        return

    # Check if download is required
    data_url = data_struct.url
    compressed_file_name = downloader(datasets_dir, data_url)

    # Unpack compressed dataset file
    unpacker(compressed_file_name, out_dir)

    # Custom preprocessing steps for data sets
    custom_preprocessor(out_dir)


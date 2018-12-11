import os
import shutil
import numpy as np

from skimage import io
from torch.utils.data.dataset import Dataset


class LFWDataset(Dataset):
    """
    Faces in the Wild specific dataset class.
    Includes indexing functionality.
    Inherets from PyTorch Dataset class.
    """
    def __init__(self, data_struct=None, train_set=True, transform=None):

        self.data_struct = data_struct
        self.custom_prep_data()
        self.test_train_split = 0.8
        self.transform = transform
        n_classes, file_list, class_to_label = self.index(train_set)
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

    def index(self, is_train_set):
        data_dir = self.data_struct.save_path
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

    def custom_prep_data(self):

        data_name = self.data_struct.name
        out_dir = self.data_struct.save_path

        # LFW specific prep steps
        lfw_dir = out_dir + '_original/'

        # If dataset already downloaded an unpacked, do nothing
        if os.path.isdir(lfw_dir):
            return

        os.rename(out_dir, lfw_dir)
        people_dir = os.listdir(lfw_dir)

        num_per_class = 20

        for p in people_dir:
            imgs = os.listdir(os.path.join(lfw_dir, p))
            if len(imgs) >= num_per_class:
                shutil.copytree(os.path.join(lfw_dir, p),
                                os.path.join(out_dir, p))

        print('{} successfully downloaded and preprocessed.'.format(data_name))


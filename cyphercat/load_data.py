import os
import io
import sys
import shutil
import requests
import zipfile
import tarfile

from .config_utils import DataStruct


def downloader(data_struct=None):
    """
    Function to download file from 
    url to specified destination file.
    If file already exists, or the url
    is a path to a valid local file,
    then fn simply returns path to 
    compresseed dataset file.
    
    Parameters
    ----------
    data_struct  : DataStruct
                   data configuration structure

    Returns
    -------
    dest_file    : string
                   path to compressed file
    """
    
    # Need defined url for dataset
    if data_struct.url == '':
        print('The url to download the dataset or path to the compressed data file was not provided.')
        print('Please provide a url, or download and unpack the dataset.\n Exiting...')
        sys.exit()

    data_name    = data_struct.name
    datasets_dir = data_struct.data_path

    url        = data_struct.url
    file_bname = os.path.basename(url)
    dest_file  = os.path.join(datasets_dir, file_bname)
    
    # Check if url is really path to local file
    if os.path.isfile(url):
        dest_file = url

    # Else if dataset zipfile doesn't exist, download it from url
    if not os.path.isfile(dest_file):
        print('Downloading {} file {}...'.format(data_name, file_bname))
        resp = requests.get(url, stream=True)
        with open(dest_file, 'wb') as f:
            shutil.copyfileobj(resp.raw, f)
    else:
        print('Compressed dataset file found, no need to download.')

    return dest_file

def unpacker(compressed_file_name='', out_directory=''):
    """
    Function to extract compressed
    dataset file to specified directory.
    Currently supports extraction of
        - zip
        - gz
    file types.

    Parameters
    ----------
    compressed_file_name  : string
                            dataset file to unpack
    out_directory         : string
                            output directory
    """

    print('Unpacking {} to {}...'.format(compressed_file_name, out_directory))

    file_ext = os.path.splitext(compressed_file_name)[1]

    # Unpack zipfile
    if 'zip' in file_ext:
        with zipfile.ZipFile(compressed_file_name) as zf:
            zf.extractall(os.path.split(out_directory)[0])
    # Unpack gzipfile
    elif 'gz' in file_ext:
        with tarfile.open(compressed_file_name) as tar:
            tar.extractall(path=out_directory)
    else:
        print('File extension {} not recognized for unpacking.\nExiting...')
        sys.exit()


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

        os.rename(os.path.join(out_dir, 'lfw/'), os.path.join(out_dir, 'lfw_original/'))

        lfw_dir    = os.path.join(out_dir, 'lfw_original/')
        people_dir = os.listdir(lfw_dir)

        num_per_class = 20

        new_dir = os.path.join(out_dir, 'lfw_' + str(num_per_class))

        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

        for p in people_dir:
            imgs = os.listdir(os.path.join(lfw_dir, p))
            if len(imgs) >= num_per_class:
                shutil.copytree(os.path.join(lfw_dir, p), os.path.join(new_dir, p))

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

    # Define output directory for data set
    out_dir      = os.path.join(datasets_dir, data_name)

    # If dataset already downloaded an unpacked, do nothing
    if os.path.isdir(out_dir):
        print('{} already downloaded, unpacked and processed.'.format(data_name))
        return

    # Check if download is required
    compressed_file_name = downloader(data_struct)

    # Unpack compressed dataset file
    unpacker(compressed_file_name, out_dir)

    # Custom preprocessing steps for data sets
    custom_preprocessor(out_dir)


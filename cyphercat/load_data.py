import os

from .utils.file_utils import downloader, unpacker


def prep_data(data_struct=None):
    """
    Function to prepare data set
    based on input configuration

    Parameters
    ----------
    data_struct  : config structure
                   parameters from 'data' field
                   of global yaml configuration file
    """
# This comment is too long, it will fail the flake8 check.
    data_name = data_struct.name
    datasets_dir = data_struct.data_path
    out_dir = data_struct.save_path

    # If dataset already downloaded an unpacked, do nothing
    if os.path.isdir(out_dir):
        print('{} already downloaded, unpacked and processed.'
              .format(data_name))
        return

    # Download and unpack any required dataset files
    url_list = data_struct.url
    for data_url in url_list:
        # Check if download is required
        compressed_file_name = downloader(datasets_dir, data_url)

        # Unpack compressed dataset file
        unpacker(compressed_file_name, out_dir)

# OBSOLETE, KEEP FOR TILL TINYIMAGENET DATASET INCLUDED
# def custom_preprocessor(out_dir=''):
#    """
#    Custom preprocessing functions for
#    specific data sets.
#
#    Parameters
#    ----------
#    out_dir   : string
#                directory of unpacked data set
#    """
#
#    # Get name of data set from output directory
#    data_name = os.path.split(out_dir)[1]
#
#    # For tiny-imagenet-200
#    if 'tiny' in data_name.lower():
#
#        # Structure the training, validation, and test data directories
#        train_dir = os.path.join(out_dir, 'train')
#        class_dirs = [os.path.join(train_dir, o) for
#              o in os.listdir(train_dir)
#              if os.path.isdir(os.path.join(train_dir, o))]
#              NOTE: ABOVE NEEDS TO BE FIXED IF THIS CODE IS USED (TOO LONG)
#
#        for c in class_dirs:
#            for f in os.listdir(os.path.join(c, 'images')):
#                os.rename(os.path.join(c, 'images', f), os.path.join(c, f))
#            for d in os.listdir(c):
#                if d.find("JPEG") == -1:
#                    if os.path.isfile(os.path.join(c, d)):
#                        os.remove(os.path.join(c, d))
#                    elif os.path.isdir(os.path.join(c, d)):
#                        os.rmdir(os.path.join(c, d))
#
#        # Get validation annotations
#        with open(os.path.join(out_dir, 'val/val_annotations.txt')) as f:
#            content = f.readlines()
#
#        for x in content:
#            line = x.split()
#
#            if not os.path.exists(os.path.join(out_dir, 'val/', line[1])):
#                os.makedirs(os.path.join(out_dir, 'val/', line[1]))
#
#            new_file_name = os.path.join(out_dir, 'val', line[1], line[0])
#            old_file_name = os.path.join(out_dir, 'val/images', line[0])
#            os.rename(old_file_name, new_file_name)
#
#    print('{} successfully downloaded and preprocessed.'.format(data_name))

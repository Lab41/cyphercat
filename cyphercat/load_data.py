import os
import io
import sys
import shutil
import requests
import zipfile
import tarfile


def downloader(url='', dest_file=''):
    """
    Function to download file from 
    url to specified destination file.
    """
    resp = requests.get(url, stream=True)
    with open(dest_file, 'wb') as f:
        shutil.copyfileobj(resp.raw, f)


def prep_data(data_struct=None):

    data_name = data_struct.name
    datasets_dir = data_struct.data_path
    
    out_dir = os.path.join(datasets_dir, data_name)

    # If dataset already downloaded an unpacked, do nothing
    if os.path.isdir(out_dir):
        print('{} already downloaded, unpacked and processed.'.format(data_name))
        return

    # Need defined url for dataset
    if data_struct.url == '':
        print('The url to download the dataset or path to the compressed data file was not provided.')
        print('Please provide a url, or download and unpack the dataset.\n Exiting...')
        sys.exit()

    url = data_struct.url
    file_bname = os.path.basename(url)
    file_ext = os.path.splitext(file_bname)[1]
    compressed_file_name = os.path.join(datasets_dir, file_bname)

    # Check if url is really path to local file
    if os.path.isfile(url):
        compressed_file_name = url
    # Else if dataset zipfile doesn't exist, download it from url
    elif not os.path.isfile(compressed_file_name):
        print('Downloading {} file {}...'.format(data_name, file_bname))
        downloader(url, compressed_file_name)

    print('{} downloaded. Unpacking to {}...'.format(data_name, datasets_dir))
    if 'zip' in file_ext:
        # Unpack zipfile
        with zipfile.ZipFile(compressed_file_name) as zf:
            zf.extractall(datasets_dir)
    elif 'gz' in file_ext:
        # Unpack gzipfile
        with tarfile.open(compressed_file_name) as tar:
            tar.extractall(path=out_dir)
    else:
        print('File extension {} not recognized for unpacking.\nExiting...')
        sys.exit()

    # For tiny-imagenet-200
    if 'tiny' in data_name.lower():

        # Structure the training, validation, and test data directories
        train_dir = os.path.join(out_dir, 'train')
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

        lfw_dir = os.path.join(out_dir, 'lfw_original/')
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


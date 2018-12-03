import io
import os
import sys
import shutil
import requests
import zipfile
import tarfile


def downloader(save_dir='', url=''):
    """
    Function to download file from 
    url to specified destination file.
    If file already exists, or the url
    is a path to a valid local file,
    then simply returns path to local file.
    
    Parameters
    ----------
    save_dir : string
               directory used for saving file 
    url      : string
               url or path to existing compressed
               dataset file

    Returns
    -------
    dest_file    : string
                   path to compressed file
    """
    
    # Need defined url for dataset
    if url == '':
        print('The url to download the dataset or path to the compressed data file was not provided.')
        print('Please provide a url, or download and unpack the dataset.\n Exiting...')
        sys.exit()

    file_bname = os.path.basename(url)
    dest_file  = os.path.join(save_dir, file_bname)
    
    # Check if url is really path to local file
    if os.path.isfile(url):
        dest_file = url

    # Else if dataset zipfile doesn't exist, download it from url
    if not os.path.isfile(dest_file):
        print('Downloading file {}...'.format(file_bname))
        resp = requests.get(url, stream=True)
        with open(dest_file, 'wb') as f:
            shutil.copyfileobj(resp.raw, f)
    else:
        print('File found, no need to download.')

    return dest_file


def unpacker(compressed_file_name='', out_directory=''):
    """
    Function to extract compressed
    file to specified directory.
    Currently supports extraction of
        - zip
        - gz
    file types.

    Parameters
    ----------
    compressed_file_name  : string
                            file to unpack
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
            tar.extractall(os.path.split(out_directory)[0])
            #tar.extractall(path=out_directory)
    else:
        print('File extension {} not recognized for unpacking.\nExiting...'.format(file_ext))
        sys.exit()



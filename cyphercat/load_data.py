import os
import io
import shutil
import requests
import zipfile
import tarfile


# Color mode dictionary for specifying
# color_mode in data generators
color_mode_dict = {1 : 'grayscale',
                   3 : 'rgb'}


def prep_data(data_struct=None):

    datasets_dir = data_struct.data_path
    data_name = data_struct.name
    url = data_struct.url

    file_bname = os.path.basename(url)
    file_ext = os.path.splitext(file_bname)[1]
    file_name = os.path.join(datasets_dir, file_bname) #os.path.join(datasets_dir, 'tiny-imagenet-200.zip')

    #if os.path.isdir(os.path.join(datasets_dir, data_name, 'val/images/')):
    #    os.rmdir(os.path.join(datasets_dir, data_name, 'val/images/'))

    out_dir = os.path.join(datasets_dir, data_name)

    # If dataset already downloaded an unpacked, do nothing
    if os.path.isdir(out_dir):
        print('{} already downloaded and unpacked.'.format(data_name))
        return

    # If dataset directory doesn't exist continue
    # If dataset zipfile doesn't exist, download it
    if not os.path.isfile(file_name):
        print('Downloading {} to {}'.format(data_name, datasets_dir))
        resp = requests.get(url, stream=True)
        with open(file_name, 'wb') as f:
            shutil.copyfileobj(resp.raw, f)

    print('{} downloaded. Unpacking...'.format(data_name))
    if 'zip' in file_ext:
        # Unpack zipfile
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall(datasets_dir)
    elif 'gz' in file_ext:
        # Unpack gzipfile
        with tarfile.open(file_name) as tar:
            tar.extractall(path=out_dir)

    # For tiny-imagenet-200
    if 'tiny' in name.lower():

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

        print('Tiny ImageNet successfully downloaded and preprocessed.')

    # For LFW
    if 'lfw' in name.lower():
        tar = tarfile.open(os.path.join(datasets_dir,'lfw.tgz'))
        tar.extractall(path=os.path.join(datasets_dir,'lfw/'))

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

        print('LFW successfully downloaded and preprocessed.')

# OBSOLETE - WILL DELETE UPON FULL TESTING

#def get_tiny_imagenet(datasets_dir):
#
#    data_name = 'tiny-imagenet-200'
#    file_name = os.path.join(datasets_dir, 'tiny-imagenet-200.zip')
#
#    #if os.path.isdir(os.path.join(datasets_dir, data_name, 'val/images/')):
#    #    os.rmdir(os.path.join(datasets_dir, data_name, 'val/images/'))
#
#    out_dir = os.path.join(datasets_dir, data_name)
#
#    # If dataset already downloaded an unpacked, do nothing
#    if os.path.isdir(out_dir):
#        print('Tiny ImageNet already downloaded and unpacked.')
#        return
#
#    # If dataset directory doesn't exist continue
#    # If dataset zipfile doesn't exist, download it
#    if not os.path.isfile(file_name):
#        print('Downloading Tiny ImageNet to %s'%datasets_dir)
#        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
#        resp = requests.get(url, stream=True)
#        with open(file_name, 'wb') as f:
#            shutil.copyfileobj(resp.raw, f)
#    # Unpack zipfile
#    print('Tiny ImageNet downloaded. Unpacking...')
#    with zipfile.ZipFile(file_name) as zf:
#        zf.extractall(datasets_dir)
#
#    # Structure the training, validation, and test data directories
#    train_dir = os.path.join(out_dir, 'train')
#    class_dirs = [os.path.join(train_dir, o) for o in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, o))]
#
#    for c in class_dirs:
#        for f in os.listdir(os.path.join(c, 'images')):
#            os.rename(os.path.join(c,'images', f), os.path.join(c, f))
#        for d in os.listdir(c):
#            if d.find("JPEG") == -1:
#                if os.path.isfile(os.path.join(c, d)):
#                    os.remove(os.path.join(c, d))
#                elif os.path.isdir(os.path.join(c, d)):
#                    os.rmdir(os.path.join(c, d))
#
#    # Get validation annotations
#    with open(os.path.join(out_dir, 'val/val_annotations.txt')) as f:
#        content = f.readlines()
#
#    for x in content:
#        line = x.split()
#
#        if not os.path.exists(os.path.join(out_dir, 'val/',line[1])):
#            os.makedirs(os.path.join(out_dir, 'val/',line[1]))
#
#        new_file_name = os.path.join(out_dir, 'val',line[1],line[0])
#        old_file_name = os.path.join(out_dir, 'val/images',line[0])
#        os.rename(old_file_name, new_file_name)
#
#    print('Tiny ImageNet successfully downloaded and preprocessed.')
#
#
#
#def get_lfw(datasets_dir):
#
#    data_name = 'lfw'
#    file_name = os.path.join(datasets_dir, 'lfw.tgz')
#
#    out_dir = os.path.join(datasets_dir, data_name)
#
#    # If dataset already downloaded an unpacked, do nothing
#    if os.path.isdir(out_dir):
#        print('LFW already downloaded and unpacked.')
#        return
#
#    # If dataset directory doesn't exist continue
#    # If dataset zipfile doesn't exist, download it
#    if not os.path.isfile(file_name):
#        print('Downloading LFW to %s'%datasets_dir)
#        url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
#        resp = requests.get(url, stream=True)
#        with open(file_name, 'wb') as f:
#            shutil.copyfileobj(resp.raw, f)
#    # Unpack gzipfile
#    print('LFW downloaded. Unpacking...')
#    with tarfile.open(file_name) as tar:
#        tar.extractall(path=out_dir)
#
#    tar = tarfile.open(os.path.join(datasets_dir,'lfw.tgz'))
#    tar.extractall(path=os.path.join(datasets_dir,'lfw/'))
#
#    os.rename(os.path.join(out_dir, 'lfw/'), os.path.join(out_dir, 'lfw_original/'))
#
#    lfw_dir = os.path.join(out_dir, 'lfw_original/')
#    people_dir = os.listdir(lfw_dir)
#
#    num_per_class = 20
#
#    new_dir = os.path.join(out_dir, 'lfw_' + str(num_per_class))
#
#    if not os.path.isdir(new_dir):
#        os.makedirs(new_dir)
#
#    for p in people_dir:
#        imgs = os.listdir(os.path.join(lfw_dir, p))
#        if len(imgs) >= num_per_class:
#            shutil.copytree(os.path.join(lfw_dir, p), os.path.join(new_dir, p))
#
#    print('LFW successfully downloaded and preprocessed.')

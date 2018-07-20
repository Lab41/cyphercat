import os, shutil
import urllib.request
import zipfile
import tarfile


def get_tiny_imagenet(datasets_dir):

    if os.path.isdir(os.path.join(datasets_dir,'tiny-imagenet-200')):
        print('Tiny ImageNet already downloaded.')
        return

    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)

    print('Downloading Tiny ImageNet')

    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    urllib.request.urlretrieve(url, os.path.join(datasets_dir,'tiny-imagenet-200.zip'))

    z = zipfile.ZipFile(os.path.join(datasets_dir,'tiny-imagenet-200.zip'), 'r')
    z.extractall(datasets_dir)
    z.close()


    train_dir = os.path.join(datasets_dir,'tiny-imagenet-200/train')
    class_dirs = [os.path.join(train_dir, o) for o in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,o))]

    for c in class_dirs:
        for f in os.listdir(os.path.join(c,'images')):
            os.rename(os.path.join(c,'images',f), os.path.join(c,f))
        for d in os.listdir(c):
            if d.find("JPEG") == -1:
                if os.path.isfile(os.path.join(c,d)):
                    os.remove(os.path.join(c,d))
                elif os.path.isdir(os.path.join(c,d)):
                    os.rmdir(os.path.join(c,d))

    with open(os.path.join(datasets_dir,'tiny-imagenet-200/val/val_annotations.txt')) as f:
        content = f.readlines()

    for x in content:
        line = x.split()

        if not os.path.exists(os.path.join(datasets_dir,'tiny-imagenet-200/val/',line[1])):
            os.makedirs(os.path.join(datasets_dir,'tiny-imagenet-200/val/',line[1]))

        new_file_name = os.path.join(datasets_dir,'tiny-imagenet-200/val',line[1],line[0])
        old_file_name = os.path.join(datasets_dir,'tiny-imagenet-200/val/images',line[0])
        os.rename(old_file_name, new_file_name)

    print('Tiny ImageNet successfully downloaded and preprocessed.')



def get_lfw(datasets_dir):

    if os.path.isdir(os.path.join(datasets_dir,'lfw')):
        print('LFW already downloaded.')
        return

    if not os.path.isdir(datasets_dir):
        os.makedirs(datasets_dir)

    print('Downloading LFW.')

    url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    urllib.request.urlretrieve(url, os.path.join(datasets_dir,'lfw.tgz'))

    tar = tarfile.open(os.path.join(datasets_dir,'lfw.tgz'))
    tar.extractall(path=os.path.join(datasets_dir,'lfw/'))

    os.rename(os.path.join(datasets_dir,'lfw/lfw/'), os.path.join(datasets_dir,'lfw/lfw_original/'))


    lfw_dir = os.path.join(datasets_dir,'lfw/lfw_original/')
    people_dir = os.listdir(lfw_dir)


    num_per_class = 20

    new_dir = os.path.join(datasets_dir,'lfw/lfw_'+str(num_per_class))

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)



    for p in people_dir:
        imgs = os.listdir(os.path.join(lfw_dir,p))
        if len(imgs) >= num_per_class:
            shutil.copytree(os.path.join(lfw_dir,p),os.path.join(new_dir,p))

    print('LFW successfully downloaded and preprocessed.')

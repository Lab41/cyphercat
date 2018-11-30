import os, shutil
import urllib.request
import zipfile
import tarfile

# For LibriSpeech download
import codecs
import fnmatch
import requests
import subprocess
import unicodedata
from tensorflow.python.platform import gfile


def get_tiny_imagenet(datasets_dir):

    
    if os.path.isdir(os.path.join(datasets_dir,'tiny-imagenet-200/val/images/')): 
        os.rmdir(os.path.join(datasets_dir,'tiny-imagenet-200/val/images/'))
        
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
    
def maybe_download(archive_name, target_dir, archive_url):
    # this and below audio downloaders adapted from https://github.com/mozilla/DeepSpeech/blob/master/bin/import_librivox.py
    #to run this: data_downloaders._download_and_preprocess_data('data/')
    
    
    # If archive file does not exist, download it...
    archive_path = os.path.join(target_dir, archive_name)

    if not os.path.exists(target_dir):
        print('No path "%s" - creating ...' % target_dir)
        os.makedirs(target_dir)

    if not os.path.exists(archive_path):
        print('No archive "%s" - downloading...' % archive_path)
        req = requests.get(archive_url, stream=True)
        total_size = int(req.headers.get('content-length', 0))
        done = 0
        with open(archive_path, 'wb') as f:
            for data in req.iter_content(1024*1024):
                done += len(data)
                f.write(data)
    else:
        print('Found archive "%s" - not downloading.' % archive_path)
    return archive_path

def _download_and_preprocess_data(data_dir):
    # Conditionally download data to data_dir
    print("Downloading Librivox data set (55GB) into {} if not already present...".format(data_dir))
   
    TRAIN_CLEAN_100_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    TRAIN_CLEAN_360_URL = "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
    TRAIN_OTHER_500_URL = "http://www.openslr.org/resources/12/train-other-500.tar.gz"

    DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"

    TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
    TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

    def filename_of(x): return os.path.split(x)[1]
    train_clean_100 = maybe_download(filename_of(TRAIN_CLEAN_100_URL), data_dir, TRAIN_CLEAN_100_URL)
    train_clean_360 = maybe_download(filename_of(TRAIN_CLEAN_360_URL), data_dir, TRAIN_CLEAN_360_URL)
    train_other_500 = maybe_download(filename_of(TRAIN_OTHER_500_URL), data_dir, TRAIN_OTHER_500_URL)

    dev_clean = maybe_download(filename_of(DEV_CLEAN_URL), data_dir, DEV_CLEAN_URL)
    dev_other = maybe_download(filename_of(DEV_OTHER_URL), data_dir, DEV_OTHER_URL)


    test_clean = maybe_download(filename_of(TEST_CLEAN_URL), data_dir, TEST_CLEAN_URL)
    test_other = maybe_download(filename_of(TEST_OTHER_URL), data_dir, TEST_OTHER_URL)

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting librivox data if not already extracted...")

    LIBRIVOX_DIR = "LibriSpeech"
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-360"), train_clean_360)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()

def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop over transcription files and split each one
    #
    # The format for each file 1-2.trans.txt is:
    #  1-2-0 transcription of 1-2-0.flac
    #  1-2-1 transcription of 1-2-1.flac
    #  ...
    #
    # Each file is then split into several files:
    #  1-2-0.txt (contains transcription of 1-2-0.flac)
    #  1-2-1.txt (contains transcription of 1-2-1.flac)
    #  ...
    #
    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", "utf-8") as fin:
                for line in fin:
                    # Parse each segment line
                    first_space = line.find(" ")
                    seqid, transcript = line[:first_space], line[first_space+1:]

                    # We need to do the encode-decode dance here because encode
                    # returns a bytes() object on Python 3, and text_to_char_array
                    # expects a string.
                    transcript = unicodedata.normalize("NFKD", transcript)  \
                                            .encode("ascii", "ignore")      \
                                            .decode("ascii", "ignore")

                    transcript = transcript.lower().strip()

                    flac_file = os.path.join(root, seqid + ".flac")
                    flac_filesize = os.path.getsize(flac_file)

                    files.append((os.path.abspath(flac_file), flac_filesize, transcript))

    return pd.DataFrame(data=files, columns=["flac_filename", "flac_filesize", "transcript"])

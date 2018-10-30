import os 
from os import path, makedirs
from tensorflow.python.platform import gfile

# Processing
split_and_df = False #True if you haven't already done this step


def maybe_download(archive_name, target_dir, archive_url):
    # If archive file does not exist, download it...
    archive_path = path.join(target_dir, archive_name)

    if not path.exists(target_dir):
        print('No path "%s" - creating ...' % target_dir)
        makedirs(target_dir)

    if not path.exists(archive_path):
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
#     TRAIN_OTHER_500_URL = "http://www.openslr.org/resources/12/train-other-500.tar.gz"

    DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"

    TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
    TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

    def filename_of(x): return os.path.split(x)[1]
    train_clean_100 = maybe_download(filename_of(TRAIN_CLEAN_100_URL), data_dir, TRAIN_CLEAN_100_URL)
    train_clean_360 = maybe_download(filename_of(TRAIN_CLEAN_360_URL), data_dir, TRAIN_CLEAN_360_URL)
#     train_other_500 = maybe_download(filename_of(TRAIN_OTHER_500_URL), data_dir, TRAIN_OTHER_500_URL)

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
#     _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-other-500"), train_other_500)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)

    if split_and_df:
    
        print("Splitting transcriptions and creating dataframes...")

        train_100 = _convert_audio_and_split_sentences(work_dir, "train-clean-100", "train-clean-100-wav")
        train_360 = _convert_audio_and_split_sentences(work_dir, "train-clean-360", "train-clean-360-wav")
    #     train_500 = _convert_audio_and_split_sentences(work_dir, "train-other-500", "train-other-500-wav")

        dev_clean = _convert_audio_and_split_sentences(work_dir, "dev-clean", "dev-clean-wav")
        dev_other = _convert_audio_and_split_sentences(work_dir, "dev-other", "dev-other-wav")

        test_clean = _convert_audio_and_split_sentences(work_dir, "test-clean", "test-clean-wav")
        test_other = _convert_audio_and_split_sentences(work_dir, "test-other", "test-other-wav")

        # Write sets to disk as CSV files
        train_100.to_csv(os.path.join(data_dir, "librivox-train-clean-100.csv"), index=False)
        train_360.to_csv(os.path.join(data_dir, "librivox-train-clean-360.csv"), index=False)
    #     train_500.to_csv(os.path.join(data_dir, "librivox-train-other-500.csv"), index=False)

        dev_clean.to_csv(os.path.join(data_dir, "librivox-dev-clean.csv"), index=False)
        dev_other.to_csv(os.path.join(data_dir, "librivox-dev-other.csv"), index=False)

        test_clean.to_csv(os.path.join(data_dir, "librivox-test-clean.csv"), index=False)
        test_other.to_csv(os.path.join(data_dir, "librivox-test-other.csv"), index=False)

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

    
#code adapted from https://github.com/mozilla/DeepSpeech/blob/master/bin/import_librivox.py
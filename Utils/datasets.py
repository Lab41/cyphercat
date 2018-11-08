from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
from skimage import io
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import os

#from config import PATH, LIBRISPEECH_SAMPLING_RATE
LIBRISPEECH_SAMPLING_RATE = 16000
PATH = '/home/mlomnitz/mlomnitz/Datasets'

sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


def to_categorical(y, num_classes):
    """Transforms an integer class label into a one-hot label (single integer to 1D vector)."""
    if y >= num_classes:
        raise(ValueError, 'Integer label is greater than the number of classes.')
    one_hot = np.zeros(num_classes)
    one_hot[y] = 1
    return one_hot


class LibriSpeechDataset(Dataset):
    """This class subclasses the torch.utils.data.Dataset object. The __getitem__ function will return a raw audio
    sample and it's label.

    This class also contains functionality to build verification tasks and n-shot, k-way classification tasks.

    # Arguments
        subsets: What LibriSpeech datasets to include.
        seconds: Minimum length of audio to include in the dataset. Any files smaller than this will be ignored.
        downsampling:
        label: One of {speaker, sex}. Whether to use sex or speaker ID as a label.
        stochastic: bool. If True then we will take a random fragment from each file of sufficient length. If False we
        will always take a fragment starting at the beginning of a file.
        pad: bool. Whether or not to pad samples with 0s to get them to the desired length. If `stochastic` is True
        then a random number of 0s will be appended/prepended to each side to pad the sequence to the desired length.
        cache: bool. Whether or not to use the cached index file
    """
    def __init__(self, path, subsets, seconds, downsampling, label='speaker', stochastic=True, pad=False,
                 transform = None, cache=True):
        if label not in ('sex', 'speaker'):
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')')

        if int(seconds * LIBRISPEECH_SAMPLING_RATE) % downsampling != 0:
            raise(ValueError, 'Down sampling must be an integer divisor of the fragment length.')

        self.subset = subsets
        self.fragment_seconds = seconds
        self.downsampling = downsampling
        self.fragment_length = int(seconds * LIBRISPEECH_SAMPLING_RATE)
        self.stochastic = stochastic
        self.pad = pad
        self.label = label
        self.transform = transform
        
        print('Initialising LibriSpeechDataset with minimum length = {}s and subsets = {}'.format(seconds, subsets))

        # Convert subset to list if it is a string
        # This allows to handle list of multiple subsets the same a single subset
        if isinstance(subsets, str):
            subsets = [subsets]

        cached_df = []
        found_cache = {s: False for s in subsets}
        if cache:
            # Check for cached files
            for s in subsets:
                subset_index_path = path + '/{}.index.csv'.format(s)
                if os.path.exists(subset_index_path):
                    cached_df.append(pd.read_csv(subset_index_path))
                    found_cache[s] = True

        # Index the remaining subsets if any
        if all(found_cache.values()) and cache:
            self.df = pd.concat(cached_df)
        else:
            df = pd.read_csv(path+'/LibriSpeech/SPEAKERS.TXT', skiprows=11, delimiter='|', error_bad_lines=False)
            df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
            df = df.assign(
                sex=df['sex'].apply(lambda x: x.strip()),
                subset=df['subset'].apply(lambda x: x.strip()),
                name=df['name'].apply(lambda x: x.strip()),
            )

            audio_files = []
            for subset, found in found_cache.items():
                if not found:
                    audio_files += self.index_subset(subset)

            # Merge individual audio files with indexing dataframe
            df = pd.merge(df, pd.DataFrame(audio_files))

            # # Concatenate with already existing dataframe if any exist
            self.df = pd.concat(cached_df+[df])

        # Save index files to data folder
        for s in subsets:
            self.df[self.df['subset'] == s].to_csv(path + '/{}.index.csv'.format(s), index=False)

        # Trim too-small files
        if not self.pad:
            self.df = self.df[self.df['seconds'] > self.fragment_seconds]
        self.num_speakers = len(self.df['id'].unique())

        # Renaming for clarity
        self.df = self.df.rename(columns={'id': 'speaker_id', 'minutes': 'speaker_minutes'})

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.assign(id=self.df.index.values)

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['speaker_id']
        self.datasetid_to_sex = self.df.to_dict()['sex']

        # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes())}

        print('Finished indexing data. {} usable files found.'.format(len(self)))

    def __getitem__(self, index):
        instance, samplerate = sf.read(self.datasetid_to_filepath[index])
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, max(len(instance)-self.fragment_length, 1))
        else:
            fragment_start_index = 0

        instance = instance[fragment_start_index:fragment_start_index+self.fragment_length]

        # Check for required length and pad if necessary
        if self.pad and len(instance) < self.fragment_length:
            less_timesteps = self.fragment_length - len(instance)
            if self.stochastic:
                # Stochastic padding, ensure instance length == self.fragment_length by appending a random number of 0s
                # before and the appropriate number of 0s after the instance
                less_timesteps = self.fragment_length - len(instance)

                before_len = np.random.randint(0, less_timesteps)
                after_len = less_timesteps - before_len

                instance = np.pad(instance, (before_len, after_len), 'constant')
            else:
                # Deterministic padding. Append 0s to reach self.fragment_length
                instance = np.pad(instance, (0, less_timesteps), 'constant')

        if self.label == 'sex':
            sex = self.datasetid_to_sex[index]
            label = sex_to_label[sex]
        elif self.label == 'speaker':
            label = self.datasetid_to_speaker_id[index]
            label = self.speaker_id_mapping[label]
        else:
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')'.format(self.label))

        # Reindex to channels first format as supported by pytorch and downsample by desired amount
        instance = instance[np.newaxis, ::self.downsampling]

        # Add transforms

        if self.transform is not None:
            instance = self.transform(instance)
            
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['speaker_id'].unique())

    @staticmethod
    def index_subset(subset):
        """
        Index a subset by looping through all of it's files and recording their speaker ID, filepath and length.
        :param subset: Name of the subset
        :return: A list of dicts containing information about all the audio files in a particular subset of the
        LibriSpeech dataset
        """
        audio_files = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar
        subset_len = 0
        for root, folders, files in os.walk(path + '/LibriSpeech/{}/'.format(subset)):
            subset_len += len([f for f in files if f.endswith('.flac')])

        progress_bar = tqdm(total=subset_len)
        for root, folders, files in os.walk(path + '/LibriSpeech/{}/'.format(subset)):
            if len(files) == 0:
                continue

            librispeech_id = int(root.split('/')[-2])

            for f in files:
                # Skip non-sound files
                if not f.endswith('.flac'):
                    continue

                progress_bar.update(1)

                instance, samplerate = sf.read(os.path.join(root, f))

                audio_files.append({
                    'id': librispeech_id,
                    'filepath': os.path.join(root, f),
                    'length': len(instance),
                    'seconds': len(instance) * 1. / LIBRISPEECH_SAMPLING_RATE
                })

        progress_bar.close()
        return audio_files

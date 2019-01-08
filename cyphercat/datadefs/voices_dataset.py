from torch.utils.data import Dataset
from cyphercat.definitions import DATASETS_DIR, DATASPLITS_DIR
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import os

LIBRISPEECH_SAMPLING_RATE = 16000

sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


def Voices_preload_and_split(subset='room-1', seconds=3, path=None,
                             pad=False, splits=None, split_type='segment'):
    """Index and split VOiCES dataset.

    Args:
        subset (string): VOiCES subset to parse, load and split.
            Currently can only handle one at a time
        seconds (int): Minimum length of audio samples to include.
        path (string): Path to location containing dataset. If left as None
            will search default location 'DATASETS_DIR' specified in
            definitions.
        pad (bool): Flag to specify whether to pad (with 0's) and keep the
            samples with lenght below the minimum.
        splits (dict): dictionary with {name:[fractions]} for a user specified
            split. The split will be saved to 'DATASPLITS_DIR' under 'name'
        split_type (string): How to do the split by data: 'segment' takes 20%
            of each audio file, and 'file' takes 20% of files for test set

    Returns:
        dict(Dataframes): Dictionary containing the dataframes corresponding
            to each split inclduing metadata.

    Example:

    Todo:
        - Write Example.
        - More work on user specified splits.
        - Add option and functionality to split longer recording into samples
        of length 'seconds' to augment data.
    """

    fragment_seconds = seconds
    if path is None:
        path = DATASETS_DIR
    index_file = path + '/VOiCES-{}.index.csv'.format(subset)

    speaker_file = '/VOiCES/Lab41-SRI-VOiCES-speaker-gender-dataset.tbl'

    print('Initialising VOiCESDataset with minimum length = {}s'
          ' and subset = {}'.format(seconds, subset))

    # Check for cached files
    subset_index_path = index_file
    if os.path.exists(subset_index_path):
        df = pd.read_csv(subset_index_path)
    # otherwise cache them
    else:
        df = pd.read_csv(path+speaker_file, skiprows=0,
                         delimiter=' ', error_bad_lines=False)
        df.columns = [col.strip().replace(';', '').lower()
                      for col in df.columns]
        df = df.assign(
            sex=df['gender'].apply(lambda x: x.strip()),
            subset=df['dataset'].apply(lambda x: x.strip()),
        )
        df = df.rename(columns={'speaker': 'id', 'gender': 'sex',
                                'dataset': 'subset'})

        audio_files = index_subset(path, subset)
        # Merge individual audio files with indexing dataframe
        df = pd.merge(df, pd.DataFrame(audio_files))

        # Remove duplicate column names
        df = df[['id', 'sex', 'subset', 'filepath', 'length', 'seconds']]

        # Add additional useful columns to dataframe:
        snippets = []
        mikes = []
        degrees = []
        noises = []

        for i in df.index:
            snip = df.filepath[i]

            sg = snip.index('sg')
            snippets.append(snip[sg+2:sg+6])

            mc = snip.index('mc')
            mikes.append(snip[mc+2:mc+4])

            dg = snip.index('dg')
            degrees.append(snip[dg+2:dg+5])

            rm = snip.index('rm')
            dash = snip[rm:].index('/')  # Find first / after rm
            noises.append(snip[dash+1:dash+5])

        df = df.assign(Section=snippets, Mic=mikes,
                       Degree=degrees, Noise=noises)

        mins = (df.groupby('id').sum()['seconds']/60)
        min_dict = mins.to_dict()
        df = df.assign(speaker_minutes=df['id'])
        df['speaker_minutes'] = df['speaker_minutes'].map(min_dict)

        # Save index files to data folder

        df.to_csv(index_file, index=False)

    # Add another column! Testing out here, before putting in above df
    index_file2 = path + '/VOiCES-{}.index2.csv'.format(subset)
    # Check for cached files
    subset_index_path2 = index_file2
    if os.path.exists(subset_index_path):
        df = pd.read_csv(subset_index_path2)
    else:
        df = pd.read_csv(subset_index_path)  # take 1st df
        noises = []

        for i in df.index:
            snip = df.filepath[i]
            # Find where room is:
            rm = snip.index('rm')
            dash = snip[rm:].index('/')  # Find first / after rm
            noises.append(snip[dash+1:dash+5])
        df = df.assign(Noise=noises)

        df.to_csv(index_file2, index=False)

    # Trim too-small files
    if not pad:
        df = df[df['seconds'] > fragment_seconds]
    num_speakers = len(df['id'].unique())

    # Renaming for clarity
    df = df.rename(columns={'id': 'speaker_id'})

    # Index of dataframe has direct correspondence to item in dataset
    df = df.reset_index(drop=True)
    df = df.assign(id=df.index.values)

    # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers
    # - 1) labels
    unique_speakers = sorted(df['speaker_id'].unique())

    print('Finished indexing data. {} usable files found.'.format(len(df)))

    dfs = {}  # Dictionary of dataframes

    # split df into data-subsets
    if splits is None:
        # Default behaviour will be to load cyphercat default splits
        # Check if splits exists.
        if split_type == 'file':
            ndfs = 5
        elif split_type == 'segment':
            ndfs = 2
        splits_ready = [False]*ndfs
        for i_split in range(ndfs):
            if os.path.exists(DATASPLITS_DIR+'/VOiCES-%s/VOiCES_%i.csv' %
                              (subset, i_split)):
                splits_ready[i_split] = True

        if all(splits_ready):  # Found all of the relelvant splits
            print('Found default splits, loading dataframe')
            dfs = {}
            for i_split in range(ndfs):
                dfs[i_split] = pd.read_csv(DATASPLITS_DIR +
                                           '/VOiCES-%s/VOiCES_%i.csv' %
                                           (subset, i_split))

        else:
            # Case when splits not found. This should only occur first time
            # LibriSpeech is parsed by developers (not users), so will include
            # a warning
            print('WARNING: Creating default splits for VOiCES!')
            if split_type == 'file':
                dfs = default_splitter(dfs, df, unique_speakers)
            elif split_type == 'segment':
                dfs = default_splitter2(dfs, df, unique_speakers)
            # write the default dataframes
            for i_df, this_df in enumerate(dfs):
                dfs[this_df] = dfs[this_df].drop(columns=['id'])
                dfs[this_df].rename(columns={'level_0': 'idx_in_original_df'},
                                    inplace=True)
                dfs[this_df].to_csv(DATASPLITS_DIR+'/VOiCES-%s/VOiCES_%i.csv' %
                                    (subset, i_df), index=False)
    else:
        name = list(splits.keys())[0]
        print('Creating user defined splits under name %s' %
              (list(splits.keys())[0]))
        total = 0
        for fraction in splits[name]:
            total += fraction
        if total != 1.:
            raise('Data split doesn\'t not add up to 1.')
        # this creates user selescted splits according to the list provided
        # num speakers for train & test is the same.
        # the below was solved with a system of equations
        # Amount data below depends on train data
        n = int(num_speakers//(2+2*splits[0]))
        # n is train data for shadow & target networks

        unique_speakers1 = unique_speakers[:n]  # Target
        unique_speakers2 = unique_speakers[n:2*n]  # Shadow
        unique_speakers3 = unique_speakers[2*n:]  # Out (target + shadow)

        dfs = splitter(dfs, df, unique_speakers1, splits, 0)
        dfs = splitter(dfs, df, unique_speakers2, splits, 2)
        # split out data for attack train  + test evenly
        dfs = splitter(dfs, df, unique_speakers3, splits=[0.5, 0.5], N=4)

    for d in dfs:
        this_df = dfs[d]
        male_df = this_df[this_df['sex'] == 'M']
        female_df = this_df[this_df['sex'] == 'F']
        print('\t\t ---- Split %i ---- \n\tUnique speakers \t Samples' % d)
        print('Male:\t\t %i\t\t %i' %
              (len(male_df['speaker_id'].unique()), len(male_df)))
        print('Female:\t\t %i\t\t %i' %
              (len(female_df['speaker_id'].unique()), len(female_df)))
        print('Total:\t\t %i\t\t %i' %
              (len(this_df['speaker_id'].unique()), len(this_df)))

    print('Finished splitting data.')

    return dfs


def index_subset(path=None, subset=None):
    """Index a subset by looping through all of it's files and recording their
    speaker ID, filepath and length.

    Args:
        subset (string): Name of the subset.
        path (string): Path to search for files to parse.

    Returns:
        (list(dicts)): A list of dicts containing information about all the
            audio files in a particular subset of the LibriSpeech dataset.

    Example:

    Todo:
        - Write example.
    """
    audio_files = []
    print('Indexing {}...'.format(subset))
    # Quick first pass to find total for tqdm bar
    subset_len = 0
    for root, folders, files in os.walk(path +
                                        '/VOiCES/{}/'.format(subset)):
        subset_len += len([f for f in files if f.endswith('.wav')])

    progress_bar = tqdm(total=subset_len)
    for root, folders, files in os.walk(path +
                                        '/VOiCES/{}/'.format(subset)):

        if len(files) == 0:
            continue

        for f in files:
            # Skip non-sound files
            if not f.endswith('.wav'):
                continue

            progress_bar.update(1)

            librispeech_id = int(root[-4:])

            instance, samplerate = sf.read(os.path.join(root, f))

            audio_files.append({
                'id': librispeech_id,
                'filepath': os.path.relpath(os.path.join(root, f), path),
                'length': len(instance),
                'seconds': len(instance) * 1. / LIBRISPEECH_SAMPLING_RATE
            })

    progress_bar.close()
    return audio_files


def default_splitter(dfs=None, df=None, unique_speakers=0):
    """ Performs cycpercat default split for librspeech dataset.

    Args:
        df (Dataframe): Dataframe to split.
        unique_speakers (int): Number of unique speakers in the dataframe

    Returns:
        dict(Dataframes): Returns a dictionary containing the dataframes for
            each of the splits.

    Example:

    Todo:
        -Write example.
    """
    # split the df by sex
    male_df = df[df['sex'] == 'M']
    female_df = df[df['sex'] == 'F']
    #
    unique_male = sorted(male_df['speaker_id'].unique())
    unique_female = sorted(female_df['speaker_id'].unique())
    n_male = len(unique_male)//2
    n_female = len(unique_female)//2
    # male splits
    m_dfs = {}
    m_dfs = splitter(m_dfs, male_df, unique_male[:n_male], [0.8, 0.2], 0)
    m_dfs = splitter(m_dfs, male_df, unique_male[n_male:], [0.5, 0.5], 2)
    m_dfs[4] = m_dfs[0][:len(m_dfs[1])]
    # female splits
    f_dfs = {}
    f_dfs = splitter(f_dfs, female_df, unique_female[:n_female], [0.8, 0.2], 0)
    f_dfs = splitter(f_dfs, female_df, unique_female[n_female:], [0.5, 0.5], 2)
    f_dfs[4] = f_dfs[0][:len(f_dfs[1])]
    # merge male and female into final splits
    for i_split in range(5):
        print('Merging split %i\n Male: %i and Female: %i' %
              (i_split, len(m_dfs[i_split]), len(f_dfs[i_split])))
        dfs[i_split] = m_dfs[i_split].append(f_dfs[i_split])

    return dfs


def default_splitter2(dfs=None, df=None, unique_speakers=0):
    """ Performs cycpercat default split for librspeech dataset.
    Args:
        df (Dataframe): Dataframe to split.
        unique_speakers (int): Number of unique speakers in the dataframe
    Returns:
        dict(Dataframes): Returns a dictionary containing the dataframes for
            each of the splits.
    Example:
    Todo:
        -Write example.
    """
    # split the df by sex
    male_df = df[df['sex'] == 'M']
    female_df = df[df['sex'] == 'F']
    #
    unique_male = sorted(male_df['speaker_id'].unique())
    unique_female = sorted(female_df['speaker_id'].unique())
    n_male = len(unique_male)//2
    n_female = len(unique_female)//2
    # male splits
    m_dfs = {}
    # Split into train & shadow
    m_dfs = splitter(m_dfs, male_df, unique_male[n_male:], [0.5, 0.5], 0)
    # female splits
    f_dfs = {}
    f_dfs = splitter(f_dfs, female_df, unique_female[n_female:], [0.5, 0.5], 0)
    # merge male and female into final splits
    for i_split in range(2):
        print('Merging split %i\n Male: %i and Female: %i' %
              (i_split, len(m_dfs[i_split]), len(f_dfs[i_split])))
        dfs[i_split] = m_dfs[i_split].append(f_dfs[i_split])

    return dfs


def splitter(dfs, df, unique_speakers, splits, N):
    """ Splits the data for given unqie speakers according to specified fractions.

    Args:
        dfs (dict(Dataframe): Current dictionary of dataframes. New splits
            will be concatenated to this dict.
        df (Dataframe): Dataframe containg all of the data and metadata.
        unique_speakers (list(int)): List containing the indices of speakers
            to include in these splits.
        splits (list(float)): List containing the fraction of the data to be
            included in each split.
        N (int): index to assign new splits when appending to dfs.

    Returns:
        (dict(Dataframe)): Updated dictionary of data splits.

    Example:

    Todo:
        - Add example.
    """
    # N is to keep track of the dataframe dict keys
    n_splits = len(splits)
    for speaker in unique_speakers:  # For each speaker

        # Speaker = valid_sequence.unique_speakers[0]
        tot_files = sum(df['speaker_id'] == speaker)

        mini_df = df[df['speaker_id'] == speaker]
        mini_df = mini_df.reset_index()

        used_files = 0
        start_file = 0
        for idx, s in enumerate(splits):  # For each split
            if idx != n_splits-1:
                n_files = int(s*tot_files)
                used_files += n_files
            else:
                n_files = tot_files - used_files

            # get stop index for the desired # of files:
            stop_file = start_file + n_files

            # initialize if first speaker, or append if later speaker
            if speaker == unique_speakers[0]:
                dfs[idx + N] = (mini_df.iloc[start_file:stop_file])
            else:
                dfs[idx + N] = dfs[idx + N].append(mini_df.iloc[start_file:
                                                                stop_file])

            # update start_file
            start_file += n_files

    for idx in range(n_splits):  # For each dataframe
        dfs[idx + N] = dfs[idx + N].reset_index()

    return dfs


class Voices_dataset(Dataset):
    """This class subclasses the torch.utils.data.Dataset.  Calling __getitem__
    will return the transformed VOiCES audio sample and it's label

    # Args
        df (Dataframe): Dataframe with audiosample path and metadata.
        seconds (int): Minimum length of audio to include in the dataset. Any
            files smaller than this will be ignored or padded to this length.
        downsampling (int): Downsampling factor.
        label (string): One of {speaker, sex}. Whether to use sex or speaker
            ID as a label.
        stochastic (bool): If True then we will take a random fragment from
            each file of sufficient length. If False we will always take a
            fragment starting at the beginning of a file.
        pad (bool): Whether or not to pad samples with 0s to get them to the
            desired length. If `stochastic` is True then a random number of 0s
            will be appended/prepended to each side to pad the sequence to the
            desired length.
        cache: bool. Whether or not to use the cached index file
    """

    def __init__(self,  df=None, seconds=3, downsampling=1, label='speaker',
                 stochastic=True, pad=False, transform=None, cache=True):
        if label not in ('sex', 'speaker'):
            raise(ValueError,
                  'Label type must be one of (\'sex\',\'speaker\')')

        if int(seconds * LIBRISPEECH_SAMPLING_RATE) % downsampling != 0:
            raise(ValueError, 'Down sampling must be an integer divisor of the'
                  ' fragment length.')

        self.fragment_seconds = seconds
        self.downsampling = downsampling
        self.fragment_length = int(seconds * LIBRISPEECH_SAMPLING_RATE)
        self.stochastic = stochastic
        self.pad = pad
        self.label = label
        self.transform = transform

        # load df from splitting function
        self.df = df
        self.num_speakers = len(self.df['speaker_id'].unique())

        # Convert arbitrary integer labels of dataset to ordered
        # 0-(num_speakers - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i
                                   for i in range(self.num_classes())}

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['speaker_id']
        self.datasetid_to_sex = self.df.to_dict()['sex']

    def __getitem__(self, index):
        instance, samplerate = sf.read(
            os.path.join(DATASETS_DIR, self.datasetid_to_filepath[index]))
        # Choose a random sample of the file
        if self.stochastic:
            upper_bound = max(len(instance) - self.fragment_length, 1)
            fragment_start_index = np.random.randint(0, upper_bound)
        else:
            fragment_start_index = 0

        instance = instance[fragment_start_index:
                            fragment_start_index+self.fragment_length]

        # Check for required length and pad if necessary
        if self.pad and len(instance) < self.fragment_length:
            less_timesteps = self.fragment_length - len(instance)
            if self.stochastic:
                # Stochastic padding, ensure instance length
                # by appending a random number of 0s before and the
                # appropriate number of 0s after the instance
                less_timesteps = self.fragment_length - len(instance)

                before_len = np.random.randint(0, less_timesteps)
                after_len = less_timesteps - before_len

                instance = np.pad(instance, (before_len, after_len),
                                  'constant')
            else:
                # Deterministic padding. Append 0s to reach desired length
                instance = np.pad(instance, (0, less_timesteps), 'constant')

        if self.label == 'sex':
            sex = self.datasetid_to_sex[index]
            label = sex_to_label[sex]
        elif self.label == 'speaker':
            label = self.datasetid_to_speaker_id[index]
            label = self.speaker_id_mapping[label]
        else:
            raise(ValueError, 'Label type must be one of (\'sex\','
                  '\'speaker\')'.format(self.label))

        # Reindex to channels first format as supported by pytorch and
        # downsample by desired amount
        instance = instance[np.newaxis, ::self.downsampling]

        # Add transforms

        if self.transform is not None:
            instance = self.transform(instance)

        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['speaker_id'].unique())

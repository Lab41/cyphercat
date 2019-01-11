from torch.utils.data import Dataset
from cyphercat.definitions import DATASETS_DIR, DATASPLITS_DIR
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import os
from .splitter import splitter, splitter2

LIBRISPEECH_SAMPLING_RATE = 16000

sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}


def load_or_index_subset(subset=None, path=None, fragment_seconds=3,
                         pad=False):
    """ Subroutine to either load existing subset dataframe or index and save it

    Args:
        subset (string): Librispeech subset to either load or index.
        path (string): Path to search for or save indexed subset.
        fragment_seconds (float): Number of seconds for audio samples.
        pad (bool): If true will accept short framgents and pad with silence.

    Returns:
        (pandas.Dataframe): Returns indexed subset in dataframe.
    """
    index_file = path + '/VOiCES-{}.index.csv'.format(subset)

    subset_index_path = index_file
    if os.path.exists(subset_index_path):
        df = pd.read_csv(subset_index_path)
    # otherwise cache them
    else:
        print('Files not found, indexing {}'.format(subset))
        speaker_file = '/VOiCES/Lab41-SRI-VOiCES-speaker-gender-dataset.tbl'
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
            noises.append(snip[rm:][dash+1:dash+5])

        df = df.assign(Section=snippets, Mic=mikes,
                       Degree=degrees, Noise=noises)

        mins = (df.groupby('id').sum()['seconds']/60)
        min_dict = mins.to_dict()
        df = df.assign(speaker_minutes=df['id'])
        df['speaker_minutes'] = df['speaker_minutes'].map(min_dict)

        # Save index files to data folder
        df.to_csv(index_file, index=False)

    # Trim too-small files
    if not pad:
        df = df[df['seconds'] > fragment_seconds]
    # Renaming for clarity
    df = df.rename(columns={'id': 'speaker_id'})

    # Index of dataframe has direct correspondence to item in dataset
    df = df.reset_index(drop=True)
    df = df.assign(id=df.index.values)

    print('\t Finished indexing {}. {} usable files found.'.format(subset,
                                                                   len(df)))

    return df


def Voices_preload_and_split(subset='room-1', seconds=3,
                             path=None, pad=False, splits=None):
    """Index and split librispeech dataset.

    Args:
        subset (string): LibriSpeech subset to parse, load and split.
            Currently can only handle one at a time
        seconds (int): Minimum length of audio samples to include.
        path (string): Path to location containing dataset. If left as None
            will search default location 'DATASETS_DIR' specified in
            definitions.
        pad (bool): Flag to specify whether to pad (with 0's) and keep the
            samples with lenght below the minimum.
        splits (dict): dictionary with {name:[fractions]} for a user specified
            split. The split will be saved to 'DATASPLITS_DIR' under 'name'

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
    num_splits = 6
    fragment_seconds = seconds
    if path is None:
        path = DATASETS_DIR

    print('Initialising VOiCESDataset with minimum length = {}s'
          ' and subset = {}'.format(seconds, subset))
    df = load_or_index_subset(subset=subset, path=path,
                              fragment_seconds=fragment_seconds, pad=pad)
    # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers
    # - 1) labels
    unique_speakers = sorted(df['speaker_id'].unique())

    # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers
    # - 1) labels

    dfs = {}  # dictionary of dataframes
    sample_dfs = {}
    # split df into data-subsets
    if splits is None:
        # Default behaviour will be to load cyphercat default splits
        # check if splits exists.
        print('Build/load speaker membership inference splits')
        splits_ready = [False]*num_splits
        for i_split in range(num_splits):
            if os.path.exists(DATASPLITS_DIR+'/VOiCES-%s/speaker_splits/'
                              'VOiCES_%i.csv' % (subset, i_split)):
                splits_ready[i_split] = True

        if all(splits_ready):  # Found all of the relelvant splits
            print('Found default speaker splits, loading dataframe')
            dfs = {}
            for i_split in range(num_splits):
                dfs[i_split] = pd.read_csv(DATASPLITS_DIR +
                                           '/VOiCES-%s/speaker_splits/'
                                           'VOiCES_%i.csv' % (subset, i_split))

        else:
            # Case when splits not found. This should only occur first time
            # VOiCES is parsed by developers (not users), so will include
            # a warning
            print('WARNING: Creating default speaker splits for VOiCES!')
            dfs = default_speaker_splitter2(dfs, df)
            # write the default dataframes
            for i_df, this_df in enumerate(dfs):
                dfs[this_df] = dfs[this_df].drop(columns=['id'])
                dfs[this_df].rename(columns={'level_0': 'idx_in_original_df'},
                                    inplace=True)
                dfs[this_df].to_csv(DATASPLITS_DIR+'/VOiCES-%s/speaker_splits/'
                                    'VOiCES_%i.csv' % (subset, i_df),
                                    index=False)

        print('Build/load sample membership inference splits')
        splits_ready = [False]*(num_splits-1)
        for i_split in range(num_splits-1):
            if os.path.exists(DATASPLITS_DIR+'/VOiCES-%s/sample_splits/'
                              'VOiCES_%i.csv' % (subset, i_split)):
                splits_ready[i_split] = True

        if all(splits_ready):  # Found all of the relelvant splits
            print('Found default sample splits, loading dataframe')
            sample_dfs = {}
            for i_split in range(num_splits-1):
                sample_dfs[i_split] = pd.read_csv(DATASPLITS_DIR +
                                                  '/VOiCES-%s/sample_splits/'
                                                  'VOiCES_%i.csv' % (subset,
                                                                     i_split))

        else:
            # Case when splits not found. This should only occur first time
            # LibriSpeech is parsed by developers (not users), so will include
            # a warning
            print('WARNING: Creating default sample splits for VOiCES!')
            sample_dfs = default_sample_splitter(sample_dfs, df)
            # write the default dataframes
            for i_df, this_df in enumerate(sample_dfs):
                sample_dfs[this_df] = sample_dfs[this_df].drop(columns=['id'])
                sample_dfs[this_df].rename(columns={'level_0':
                                                    'idx_in_original_df'},
                                           inplace=True)
                sample_dfs[this_df].to_csv(DATASPLITS_DIR+'/VOiCES-%s/'
                                           'sample_splits/VOiCES_%i.csv' %
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
        # amt data depends on train data
        n = int(len(unique_speakers)//(2+2*splits[0]))
        # n is train data for shadow & target networks

        unique_speakers1 = unique_speakers[:n]  # target
        unique_speakers2 = unique_speakers[n:2*n]  # shadow
        unique_speakers3 = unique_speakers[2*n:]  # out (target + shadow)

        dfs = splitter(dfs=dfs, df=df, unique_categories=unique_speakers1,
                       category_id='speaker_id', splits=splits, N=0)
        dfs = splitter(dfs=dfs, df=df, unique_categories=unique_speakers2,
                       category_id='speaker_id', splits=splits, N=2)

        # split out data for attack train  + test evenly
        dfs = splitter(dfs=dfs, df=df, unique_categories=unique_speakers3,
                       category_id='speaker_id', splits=[0.5, 0.5], N=4)

    print('\n ------- Speaker split statistics ------- ')
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
    print(' ---------------------------------------- \n')
    print(' ------- Sample split statistics -------- ')
    for d in sample_dfs:
        this_df = sample_dfs[d]
        male_df = this_df[this_df['sex'] == 'M']
        female_df = this_df[this_df['sex'] == 'F']
        print('\t\t ---- Split %i ---- \n\tUnique speakers \t Samples' % d)
        print('Male:\t\t %i\t\t %i' %
              (len(male_df['speaker_id'].unique()), len(male_df)))
        print('Female:\t\t %i\t\t %i' %
              (len(female_df['speaker_id'].unique()), len(female_df)))
        print('Total:\t\t %i\t\t %i' %
              (len(this_df['speaker_id'].unique()), len(this_df)))
    print(' ---------------------------------------- \n')
    print('Finished splitting data.')

    return dfs, sample_dfs


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
                                        '/LibriSpeech/{}/'.format(subset)):
        subset_len += len([f for f in files if f.endswith('.flac')])

    progress_bar = tqdm(total=subset_len)
    for root, folders, files in os.walk(path +
                                        '/LibriSpeech/{}/'.format(subset)):
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
                'filepath': os.path.relpath(os.path.join(root, f), path),
                'length': len(instance),
                'seconds': len(instance) * 1. / LIBRISPEECH_SAMPLING_RATE
            })

    progress_bar.close()
    return audio_files


def default_speaker_splitter(dfs=None, df=None):
    """ Performs cycpercat default split for librspeech dataset.

    Args:
        dfs (dict(Dataframe)): Current dictionary of dataframes.
                               Splits concatenated to this dict.
        df (Dataframe): Dataframe to split.

    Returns:
        dict(Dataframes): Returns a dictionary containing the dataframes for
            each of the splits.

    Example:

    Todo:
        -Write example.
    """
    # defining dataset category
    cat_id = 'speaker_id'
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
    # splits speakers in 0.8/0.2 split for target
    m_dfs = splitter2(dfs=m_dfs, df=male_df,
                      unique_categories=unique_male[:n_male],
                      category_id=cat_id, splits=[0.8, 0.2], N=0)
    # splits by speaker for attack
    m_dfs = splitter2(dfs=m_dfs, df=male_df,
                      unique_categories=unique_male[n_male:],
                      category_id=cat_id, splits=[0.5, 0.5],
                      N=2, split_by_class=True)
    m_dfs[4] = m_dfs[0][:len(m_dfs[1])]
    # female splits
    f_dfs = {}
    f_dfs = splitter2(dfs=f_dfs, df=female_df,
                      unique_categories=unique_female[:n_female],
                      category_id=cat_id, splits=[0.8, 0.2], N=0)
    f_dfs = splitter2(dfs=f_dfs, df=female_df,
                      unique_categories=unique_female[n_female:],
                      category_id=cat_id, splits=[0.5, 0.5], N=2,
                      split_by_class=True)
    f_dfs[4] = f_dfs[0][:len(f_dfs[1])]
    # merge male and female into final splits
    for i_split in range(5):
        print('Merging split %i\n Male: %i and Female: %i' %
              (i_split, len(m_dfs[i_split]), len(f_dfs[i_split])))
        dfs[i_split] = m_dfs[i_split].append(f_dfs[i_split])

    return dfs


def default_speaker_splitter2(dfs=None, df=None):
    """ Performs cycpercat default split for librspeech dataset.

    Args:
        dfs (dict(Dataframe)): Current dictionary of dataframes.
                               Splits concatenated to this dict.
        df (Dataframe): Dataframe to split.

    Returns:
        dict(Dataframes): Returns a dictionary containing the dataframes for
            each of the splits.

    Example:

    Todo:
        -Write example.
    """
    # defining dataset category
    cat_id = 'speaker_id'
    # split the df by sex
    male_df = df[df['sex'] == 'M']
    female_df = df[df['sex'] == 'F']
    #
    unique_male = sorted(male_df['speaker_id'].unique())
    unique_female = sorted(female_df['speaker_id'].unique())
    # Below math to get the data volume for splits 4 & 5 similar
    n_male = len(unique_male)//50
    n_female = len(unique_female)//50
    n1 = 23
    n2 = 46
    # male splits
    m_dfs = {}
    # splits speakers in 0.8/0.2 split for target
    m_dfs = splitter2(dfs=m_dfs, df=male_df,
                      unique_categories=unique_male[:n_male*n1],
                      category_id=cat_id, splits=[0.8, 0.2], N=0)
    # splits by speaker for attack
    m_dfs = splitter2(dfs=m_dfs, df=male_df,
                      unique_categories=unique_male[n_male*n1:n_male*n2],
                      category_id=cat_id, splits=[0.5, 0.5],
                      N=2, split_by_class=True)
    # split off unheard speakers for outset
    m_dfs = splitter2(dfs=m_dfs, df=male_df,
                      unique_categories=unique_male[n_male*n2:],
                      category_id=cat_id, splits=[0, 1],
                      N=4, split_by_class=True)
    # Replace in set with subset of df0
    m_dfs[4] = m_dfs[0][:len(m_dfs[1])]
    # female splits
    f_dfs = {}
    f_dfs = splitter2(dfs=f_dfs, df=female_df,
                      unique_categories=unique_female[:n_female*n1],
                      category_id=cat_id, splits=[0.8, 0.2], N=0)
    f_dfs = splitter2(dfs=f_dfs, df=female_df,
                      unique_categories=unique_female[n_female*n1:n_female*n2],
                      category_id=cat_id, splits=[0.5, 0.5], N=2,
                      split_by_class=True)
    f_dfs = splitter2(dfs=f_dfs, df=female_df,
                      unique_categories=unique_female[n_female*n2:],
                      category_id=cat_id, splits=[0, 1], N=4,
                      split_by_class=True)
    f_dfs[4] = f_dfs[0][:len(f_dfs[1])]
    # merge male and female into final splits
    for i_split in range(6):
        print('Merging split %i\n Male: %i and Female: %i' %
              (i_split, len(m_dfs[i_split]), len(f_dfs[i_split])))
        dfs[i_split] = m_dfs[i_split].append(f_dfs[i_split])

    return dfs


def default_sample_splitter(dfs=None, df=None):
    """ Performs cycpercat default split for librspeech dataset.

    Args:
        dfs (dict(Dataframe)): Current dictionary of dataframes.
                               Splits concatenated to this dict.
        df (Dataframe): Dataframe to split.

    Returns:
        dict(Dataframes): Returns a dictionary containing the dataframes for
            each of the splits.

    Example:

    Todo:
        -Write example.
    """
    # defining dataset category
    cat_id = 'speaker_id'
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
    m_dfs = splitter2(dfs=m_dfs, df=male_df,
                      unique_categories=unique_male[:n_male],
                      category_id=cat_id, splits=[0.8, 0.2], N=0)
    m_dfs = splitter2(dfs=m_dfs, df=male_df,
                      unique_categories=unique_male[n_male:],
                      category_id=cat_id, splits=[0.5, 0.5], N=2)
    m_dfs[4] = m_dfs[0][:len(m_dfs[1])]
    # female splits
    f_dfs = {}
    f_dfs = splitter2(dfs=f_dfs, df=female_df,
                      unique_categories=unique_female[:n_female],
                      category_id=cat_id, splits=[0.8, 0.2], N=0)
    f_dfs = splitter2(dfs=f_dfs, df=female_df,
                      unique_categories=unique_female[n_female:],
                      category_id=cat_id, splits=[0.5, 0.5], N=2)
    f_dfs[4] = f_dfs[0][:len(f_dfs[1])]
    # merge male and female into final splits
    for i_split in range(5):
        print('Merging split %i\n Male: %i and Female: %i' %
              (i_split, len(m_dfs[i_split]), len(f_dfs[i_split])))
        dfs[i_split] = m_dfs[i_split].append(f_dfs[i_split])

    return dfs


class Voices_dataset(Dataset):
    """This class subclasses the torch.utils.data.Dataset.  Calling __getitem__
    will return the transformed librispeech audio sample and it's label

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
            raise(ValueError, 'Label type must be one of (\'sex\','
                  '\'speaker\')')

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

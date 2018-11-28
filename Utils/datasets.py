from torch.utils.data import Dataset
from tqdm import tqdm
import soundfile as sf
import pandas as pd
import numpy as np
import os

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

def Libri_preload_and_split(path,subsets,seconds,pad=False,cache=True,splits = [.8,.2], attacking = False):
    fragment_seconds = seconds
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
        df = pd.concat(cached_df)
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
                audio_files += index_subset(path, subset)

        # Merge individual audio files with indexing dataframe
        df = pd.merge(df, pd.DataFrame(audio_files))

        # # Concatenate with already existing dataframe if any exist
        df = pd.concat(cached_df+[df])

    # Save index files to data folder
    for s in subsets:
        df[df['subset'] == s].to_csv(path + '/{}.index.csv'.format(s), index=False)

    # Trim too-small files
    if not pad:
        df = df[df['seconds'] > fragment_seconds]
    num_speakers = len(df['id'].unique())

    # Renaming for clarity
    df = df.rename(columns={'id': 'speaker_id', 'minutes': 'speaker_minutes'})

    # Index of dataframe has direct correspondence to item in dataset
    df = df.reset_index(drop=True)
    df = df.assign(id=df.index.values)

    # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
    unique_speakers = sorted(df['speaker_id'].unique())

    print('Finished indexing data. {} usable files found.'.format(len(df)))

    dfs = {} #dictionary of dataframes
        
        
    #split df into data-subsets
    if attacking == 1: #adversary 1, which requires an additional split for a shadow network
        #splits unique speakers into three unequal parts. 
        # num speakers for train & test is the same.
        # the below was solved with a system of equations
        n = int(num_speakers//(2+2*splits[0]))#amt data depends on amt train dat
        #n is train data for shadow & target networks
        
        unique_speakers1 = unique_speakers[:n] #target 
        unique_speakers2 = unique_speakers[n:2*n] # shadow
        unique_speakers3 = unique_speakers[2*n:] # out (target + shadow)

        dfs = splitter(dfs,df,unique_speakers1, splits,0)
        dfs = splitter(dfs,df,unique_speakers2, splits,2)
        dfs = splitter(dfs,df,unique_speakers3, splits=[0.5,0.5],N = 4) #split out data for attack train  + test evenly
        
    elif attacking == 3: #adversary 3, which just requires in & out data
        #splits unique speakers into two unequal parts. 
        # the below was solved with a system of equations
        n = int(num_speakers//(1+splits[0]))#amt data depends on amt train dat
        #n is train data for target networks
        
        unique_speakers1 = unique_speakers[:n] #target 
        unique_speakers2 = unique_speakers[n:] # out (target + shadow)

        dfs = splitter(dfs,df,unique_speakers1, splits,0)
        dfs = splitter(dfs,df,unique_speakers2, splits=[1,0],N=2) #split out data for just attack eval
    else: # just split into train & test
        dfs = splitter(dfs, df,unique_speakers, splits, 0)
 
    #check that the splits were as desired:
        
    for d in dfs:
        print(len(dfs[d]))

    print('Finished splitting data.')

    return dfs

def index_subset(path , subset):
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

def VOiCES_preload_and_split(path,subsets,seconds,pad=False,cache=True,splits = [.8,.2], attacking = False):
    fragment_seconds = seconds
    
    speakerF = 'Lab41-SRI-VOiCES-speaker-gender-dataset_SUBSET.csv'
    print('Initialising VOiCES Dataset with minimum length = {}s and subsets = {}'.format(seconds, subsets))

    for file in os.listdir(path):
        if file.startswith("part"):
            subsets.append(file)

    # Convert subset to list if it is a string
    # This allows to handle list of multiple subsets the same a single subset
    if isinstance(subsets, str):
        subsets = [subsets]

    cached_df = []
    found_cache = {s: False for s in subsets}
    if cache:
        # Check for cached files
#         if data == 'Libri':
#             for s in subsets:
#                 subset_index_path = path + '/{}.index.csv'.format(s)
#                 if os.path.exists(subset_index_path):
#                     print(subset_index_path)
#                     cached_df.append(pd.read_csv(subset_index_path))
#                     found_cache[s] = True
       
        subset_index_path = path + '/index.csv'
        if os.path.exists(subset_index_path):
            cached_df.append(pd.read_csv(subset_index_path))
            for s in subsets:
                found_cache[s] = True


    # Index the remaining subsets if any
    if all(found_cache.values()) and cache:
        df = pd.concat(cached_df)
    else:
        df = pd.read_csv(path+speakerF, skiprows=0, delimiter=',', error_bad_lines=False)
        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.rename(columns={'speaker': 'id', 'gender': 'sex','dataset':'subset'})
        df = df.assign(
            sex=df['sex'].apply(lambda x: x.strip()),
            subset=df['subset'].apply(lambda x: x.strip()),
        )
        
        audio_files = []
        for subset, found in found_cache.items():
            if not found:
                audio_files += index_subset_VOiCES(path, subset)
                
       
        # Merge individual audio files with indexing dataframe
        df = pd.merge(df, pd.DataFrame(audio_files))
       
        # # Concatenate with already existing dataframe if any exist
        
        df = pd.concat(cached_df+[df])
    

    # Save index files to data folder
#     if data == 'Libri':
#         for s in subsets:
#             df[df['subset'] == s].to_csv(path + '/{}.index.csv'.format(s), index=False)  
    df.to_csv(path + '/index.csv', index=False)

    # Trim too-small files
    if not pad:
        df = df[df['seconds'] > fragment_seconds]
    num_speakers = len(df['id'].unique())

    # Renaming for clarity
    df = df.rename(columns={'id': 'speaker_id'})
    
    # Index of dataframe has direct correspondence to item in dataset
    df = df.reset_index(drop=True)
    df = df.assign(id=df.index.values)

    # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
    unique_speakers = sorted(df['speaker_id'].unique())

    print('Finished indexing data. {} usable files found.'.format(len(df)))

    dfs = {} #dictionary of dataframes
        
        
    #split df into data-subsets
    if attacking == 1: #adversary 1, which requires an additional split for a shadow network
        #splits unique speakers into three unequal parts. 
        # num speakers for train & test is the same.
        # the below was solved with a system of equations
        n = int(num_speakers//(2+2*splits[0]))#amt data depends on amt train dat
        #n is train data for shadow & target networks
        
        unique_speakers1 = unique_speakers[:n] #target 
        unique_speakers2 = unique_speakers[n:2*n] # shadow
        unique_speakers3 = unique_speakers[2*n:] # out (target + shadow)

        dfs = splitter(dfs,df,unique_speakers1, splits,0)
        dfs = splitter(dfs,df,unique_speakers2, splits,2)
        dfs = splitter(dfs,df,unique_speakers3, splits=[0.5,0.5],N = 4) #split out data for attack train  + test evenly
        
    elif attacking == 3: #adversary 3, which just requires in & out data
        #splits unique speakers into two unequal parts. 
        # the below was solved with a system of equations
        n = int(num_speakers//(1+splits[0]))#amt data depends on amt train dat
        #n is train data for target networks
        
        unique_speakers1 = unique_speakers[:n] #target 
        unique_speakers2 = unique_speakers[n:] # out (target + shadow)

        dfs = splitter(dfs,df,unique_speakers1, splits,0)
        dfs = splitter(dfs,df,unique_speakers2, splits=[1,0],N=2) #split out data for just attack eval
    else: # just split into train & test
        dfs = splitter(dfs, df,unique_speakers, splits, 0)
 
    #check that the splits were as desired:
        
    for d in dfs:
        print(len(dfs[d]))

    print('Finished splitting data.')

    return dfs

def index_subset_VOiCES(path , subset):
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
    addpath = ''
    ftype = '.wav'

    for root, folders, files in os.walk(path + addpath +'{}/'.format(subset)):
        subset_len += len([f for f in files if f.endswith(ftype)])
       
    progress_bar = tqdm(total=subset_len)
    for root, folders, files in os.walk(path + addpath + '{}/'.format(subset)):
        if len(files) == 0:
            continue
        
        for f in files:
            # Skip non-sound files
            if not f.endswith(ftype):
                continue
                     
            
            librispeech_id = int(f[f.index('sp')+2:f.index('sp')+6])
                
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
      
def splitter(dfs,df,unique_speakers, splits,N):
    #N is to keep track of the dataframe dict keys
    n_splits = len(splits)
    for speaker in unique_speakers: #for each speaker

    # speaker = valid_sequence.unique_speakers[0]
        tot_files = sum(df['speaker_id']==speaker)

        mini_df = df[df['speaker_id']==speaker]    
        mini_df = mini_df.reset_index()

        used_files = 0
        start_file = 0
        for idx, s in enumerate(splits): #for each split
            if idx != n_splits-1:
                n_files = int(s*tot_files)
                used_files += n_files
            else:
                n_files = tot_files - used_files

            #get stop index for the desired # of files:
            stop_file = start_file + n_files

            #initialize if first speaker, or append if later speaker
            if speaker == unique_speakers[0]:
                dfs[idx + N] = (mini_df.iloc[start_file:stop_file])
            else:
                dfs[idx + N] = dfs[idx + N].append(mini_df.iloc[start_file:stop_file])

            #update start_file
            start_file += n_files

    for idx in range(n_splits): #for each dataframe
        dfs[idx + N] = dfs[idx + N].reset_index()

    return dfs

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
    def __init__(self, path, df, seconds, downsampling, label='speaker', stochastic=True, pad=False,
                 transform = None, cache=True):
        if label not in ('sex', 'speaker'):
            raise(ValueError, 'Label type must be one of (\'sex\', \'speaker\')')

        if int(seconds * LIBRISPEECH_SAMPLING_RATE) % downsampling != 0:
            raise(ValueError, 'Down sampling must be an integer divisor of the fragment length.')

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
        
        # Convert arbitrary integer labels of dataset to ordered 0-(num_speakers - 1) labels
        self.unique_speakers = sorted(self.df['speaker_id'].unique())
        self.speaker_id_mapping = {self.unique_speakers[i]: i for i in range(self.num_classes())}  
        
        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_speaker_id = self.df.to_dict()['speaker_id']
        self.datasetid_to_sex = self.df.to_dict()['sex']
        
        

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


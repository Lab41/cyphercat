import sys
sys.path.insert(0, '../../')
import cyphercat as cc

print('Downloading')
cc.download_and_preprocess_data(cc.DATASETS_DIR)
print('Loading splits')
dfs = cc.Libri_preload_and_split()
print('Initializing dataset')
test_set = cc.LibriSpeechDataset(df=dfs[4])
print('Succesfully loaded libri-speech')

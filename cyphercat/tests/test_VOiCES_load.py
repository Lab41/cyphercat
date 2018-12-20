import sys
sys.path.insert(0, '../../')
import cyphercat as cc

print('Loading splits')
dfs = cc.Voices_preload_and_split()
print('Initializing dataset')
test_set = cc.LibriSpeechDataset(df=dfs[4])
print('Succesfully loaded VOiCES')

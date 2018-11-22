import torch
import librosa as libr
import numpy as np


class ToMFCC:
    '''
    Transformation to convert soundfile loaded via LibriSpeechDataset to Mel-
    frequency cepstral coefficients (MFCCs)
    Args: 
    number_of_mels: Number of bins to use for cepstral coefficients
    Returns:
    torch.float tensor
    '''
    def __init__(self, number_of_mels=128):
        self.number_of_mels = number_of_mels
        
    def __call__(self, y):
        dims = y.shape
        y = libr.feature.melspectrogram(np.reshape(y, (dims[1],)), 16000,
                                        n_mels=self.number_of_mels, fmax=8000)
        y = libr.feature.mfcc(S=libr.power_to_db(y))
        y = torch.from_numpy(y)                           
        return y.float()


class STFT:
    '''
    Short-time Fourier transform (STFT) for librosa dataset
    Args: 
    phase: If true, will return the magnitude and phase of the transformation, 
    if false only returns magnitude
    Returns:
    torch.float tensor
    '''
    def __init__(self, phase=False):
        self.phase = phase

    def __call__(self, y):
        dims = y.shape
        y = libr.core.stft(np.reshape(y, (dims[1],)))
        y, phase = np.abs(y), np.angle(y)
        y = torch.from_numpy(y).permute(1, 0)
        phase = torch.from_numpy(phase).permute(1, 0)
        if self.phase:
            return torch.cat( (y, phase), dim=0).float()
        else:
            return y.float()

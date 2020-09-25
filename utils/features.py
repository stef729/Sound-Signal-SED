# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:07:24 2020

@author: CS
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import csv
import random
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import config
import soundfile

class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
    
    def transform_multichannel(self, multichannel_audio):
        '''Extract feature of a multichannel audio file. 
        
        Args:
          multichannel_audio: (samples, channels_num)
          
        Returns:
          feature: (channels_num, frames_num, freq_bins)
        '''
        
        (samples, channels_num) = multichannel_audio.shape
        
        feature = np.array([self.transform_singlechannel(
            multichannel_audio[:, m]) for m in range(channels_num)])
        
        return feature
    
    
    def transform_singlechannel(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram
    
    
    
    
    
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def read_multichannel_audio(audio_path, target_fs=None):

    (multichannel_audio, fs) = soundfile.read(audio_path)
    '''(samples, channels_num)'''
    
    if target_fs is not None and fs != target_fs:
        if multichannel_audio.ndim < 2:
            multichannel_audio = multichannel_audio.reshape(multichannel_audio.shape[0], 1)
        (samples, channels_num) = multichannel_audio.shape
        
        multichannel_audio = np.array(
            [librosa.resample(
                multichannel_audio[:, i], 
                orig_sr=fs, 
                target_sr=target_fs) 
            for i in range(channels_num)]).T
    return multichannel_audio, fs





 
    # Calculate feature for each audio file and write out to hdf5. 
def calculate_feature_for_each_audio_file(dataset_dir, workspace):
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second

    # Paths    
    metas_dir = os.path.join(dataset_dir, 'metadata_dev')
    audios_dir = os.path.join(dataset_dir, 'foa_dev')
    features_dir = os.path.join(workspace, 'features')
        
    create_folder(features_dir)
    
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)
    
    # Extract features and targets
    meta_names = sorted(os.listdir(metas_dir))
    

    
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    
    # 读入所有数据
    for (n, meta_name) in enumerate(meta_names):        
        meta_path = os.path.join(metas_dir, meta_name)
        bare_name = os.path.splitext(meta_name)[0]
        audio_path = os.path.join(audios_dir, '{}.wav'.format(bare_name))
        feature_path = os.path.join(features_dir, '{}.h5'.format(bare_name))
        
        df = pd.read_csv(meta_path, sep=',')
        event_array = df['sound_event_recording'].values
        start_time_array = df['start_time'].values
        end_time_array = df['end_time'].values


        # Read audio
        # audio,sr = librosa.load(audio_path, sample_rate)
        (multichannel_audio, _) = read_multichannel_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
        
        
        
        # Extract feature
        # feature = feature_extractor.transform_singlechannel(audio)
        feature = feature_extractor.transform_multichannel(multichannel_audio)
        
        with h5py.File(feature_path, 'w') as hf:
            hf.create_dataset('feature', data=feature, dtype=np.float32)
            
            hf.create_group('target')
            hf['target'].create_dataset('event', data=[e.encode() for e in event_array], dtype='S20')
            hf['target'].create_dataset('start_time', data=start_time_array, dtype=np.float32)
            hf['target'].create_dataset('end_time', data=end_time_array, dtype=np.float32)
        
        print(n, feature_path, feature.shape)
    
    print('Extract features finished! {:.3f} s'.format(time.time() - extract_time))
        
#  Calculate and write out scalar of development data.     
def calculate_scalar(workspace):
    
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
   
    # Paths  
    features_dir = os.path.join(workspace, 'features')    
    scalar_path = os.path.join(workspace, 'scalars', 'scalar.h5')    
    create_folder(os.path.dirname(scalar_path))

    # Load data
    load_time = time.time()
    feature_names = os.listdir(features_dir)
    all_features = []
    
    for feature_name in feature_names:
        feature_path = os.path.join(features_dir, feature_name)
        
        with h5py.File(feature_path, 'r') as hf:
            feature = hf['feature'][:]
            all_features.append(feature)
            
    print('Load feature time: {:.3f} s'.format(time.time() - load_time))
    
    # Calculate scalar
    all_features = np.concatenate(all_features, axis=1)
    (mean, std) = calculate_scalar_of_tensor(all_features)
    
    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)
    
    print('All features: {}'.format(all_features.shape))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    print('Write out scalar to {}'.format(scalar_path))






if __name__ == '__main__':
    dataset_dir = 'D:/Project/Sound/Target Detection/data/development'
    workspace = 'D:/Project/Sound/Target Detection/1_results'
    if True:
        calculate_feature_for_each_audio_file(dataset_dir, workspace)
    elif False:
        calculate_scalar(workspace)
    else :
        raise Exception('Incorrect arguments!')


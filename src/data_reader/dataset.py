import numpy as np
import torch
from torch.utils import data
import h5py
from scipy.io import wavfile
from collections import defaultdict
from random import randint

class ForwardLibriSpeechRawXXreverseDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        original = self.h5f[utt_id][:]

        return utt_id, self.h5f[utt_id][:], original[::-1].copy()
 
class ForwardLibriSpeechReverseRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        original = self.h5f[utt_id][:]

        return utt_id, original[::-1].copy() # reverse
    
class ForwardLibriSpeechRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 

        return utt_id, self.h5f[utt_id][:]
    

class ReverseRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ RawDataset trained reverse;
            raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)
        """
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = i.split(' ')[1]
            self.spk2idx[spk] = int(idx)
        """
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        original = self.h5f[utt_id][index:index+self.audio_window]
        return original[::-1].copy() # reverse 

class ForwardDatasetSITWSilence(data.Dataset):
    ''' dataset for forward passing sitw without vad '''
    def __init__(self, wav_file):
        """ wav_file: /export/c01/jlai/thesis/data/sitw_dev_enroll/wav.scp
        """
        self.wav_file  = wav_file

        with open(wav_file) as f:
            temp = f.readlines()
        self.utts = [x.strip().split(' ')[0] for x in temp]
        self.wavs = [x.strip().split(' ')[1] for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        wav_path = self.wavs[index] # get the wav file path 
        fs, data = wavfile.read(wav_path)

        return self.utts[index], data

class ForwardDatasetSwbdSreSilence(data.Dataset):
    ''' dataset for forward passing swbd_sre or sre16 without vad '''
    def __init__(self, wav_dir, scp_file):
        """ wav_dir: /export/c01/jlai/thesis/data/swbd_sre_combined/wav/
            list_file: /export/c01/jlai/thesis/data/swbd_sre_combined/list/log/swbd_sre_utt.{1..50}.scp
        """
        self.wav_dir  = wav_dir

        with open(scp_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        path   = self.wav_dir + utt_id
        fs, data = wavfile.read(path)

        return utt_id, data

class RawDatasetSwbdSreOne(data.Dataset):
    ''' dataset for swbd_sre with vad ; for training cpc with ONE voiced segment per recording '''
    def __init__(self, raw_file, list_file):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training3.txt, list/val3.txt
        """
        self.raw_file  = raw_file 

        with open(list_file) as f:
            temp = f.readlines()
        all_utt = [x.strip() for x in temp]
    
        # dictionary mapping unique utt id to its number of voied segments
        self.utts = defaultdict(lambda: 0)
        for i in all_utt: 
            count  = i.split('-')[-1]
            utt_uniq = i[:-(len(count)+1)]
            self.utts[utt_uniq] += 1 # count 

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts.keys()[index] # get the utterance id 
        count  = self.utts[utt_id] # number of voiced segments for the utterance id  
        select = randint(1, count)
        h5f = h5py.File(self.raw_file, 'r')
        
        return h5f[utt_id+'-'+str(select)][:]

class RawDatasetSwbdSreSilence(data.Dataset):
    ''' dataset for swbd_sre without vad; for training cpc with ONE voiced/unvoiced segment per recording '''
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training2.txt, list/val2.txt
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        h5f = h5py.File(self.raw_file, 'r')
        utt_len = h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 

        return h5f[utt_id][index:index+self.audio_window]

class RawDatasetSwbdSre(data.Dataset):
    ''' dataset for swbd_sre with vad ; for training cpc with ONE voiced segment per recording '''
    def __init__(self, raw_file, list_file):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training.txt
        """
        self.raw_file  = raw_file 

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        h5f = h5py.File(self.raw_file, 'r')

        return h5f[utt_id][:]

class RawDatasetSpkClass(data.Dataset):
    def __init__(self, raw_file, list_file, index_file, audio_window, frame_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            index_file: spk2idx
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.frame_window = frame_window

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = int(i.split(' ')[1])
            self.spk2idx[spk] = idx

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        h5f = h5py.File(self.raw_file, 'r')
        utt_len = h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        speaker = utt_id.split('-')[0]
        label   = torch.tensor(self.spk2idx[speaker])

        return h5f[utt_id][index:index+self.audio_window], label.repeat(self.frame_window)

class RawXXreverseDataset(data.Dataset):
    ''' RawDataset but returns sequence twice: x, x_reverse '''
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        original = self.h5f[utt_id][index:index+self.audio_window]
        return original, original[::-1].copy() # reverse

class RawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)
        """
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = i.split(' ')[1]
            self.spk2idx[spk] = int(idx)
        """
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        return self.h5f[utt_id][index:index+self.audio_window]

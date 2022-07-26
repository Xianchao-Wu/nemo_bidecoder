#!/usr/bin/env python3
# encoding: utf-8

import sys
import argparse
import json
import codecs
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader

torchaudio.set_audio_backend("sox_io")


class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''

    def __init__(self, feat_dim, resample_rate, configs):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate
        self.configs = configs
        pass

    def _get_feat_value(self, key, default_value):
        if key in self.configs['model']['preprocessor']:
            return self.configs['model']['preprocessor'][key]
        return default_value

    def _get_mat(self, waveform, temp_resample_rate):
        dither = self._get_feat_value('dither', 0.0) 

        frame_length = self._get_feat_value('window_size', 25.0)
        if frame_length < 1.0:
            frame_length *= 1000.0 # 0.025 in config -> 25.0

        frame_shift = self._get_feat_value('window_stride', 10.0)
        if frame_shift < 1.0:
            frame_shift *= 1000.0 # 0.01 in config -> 10.0

        energy_floor = self._get_feat_value('energy_floor', 0.0)
        
        # (‘hamming’|’hanning’|’povey’|’rectangular’|’blackman’) (Default: 'povey')
        window_type = self._get_feat_value('window', 'hann')
        if window_type == 'hann':
            window_type = 'hanning'

        use_log_fbank = self._get_feat_value('log', True)

        #import ipdb; ipdb.set_trace()
        mat = kaldi.fbank(waveform,
                          num_mel_bins = self.feat_dim,
                          dither = dither,
                          frame_length = frame_length,
                          frame_shift = frame_shift,
                          energy_floor = energy_floor,
                          window_type = window_type,
                          use_log_fbank = use_log_fbank,
                          sample_frequency = temp_resample_rate)
        return mat

    def __call__(self, batch):
        #import ipdb; ipdb.set_trace()
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            value = item.strip().split(",")
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            sample_rate = torchaudio.backend.sox_io_backend.info(wav_path).sample_rate
            temp_resample_rate = sample_rate
            # len(value) == 3 means segmented wav.scp,
            # len(value) == 1 means original wav.scp
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                waveform, sample_rate = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(item)

            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                temp_resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=temp_resample_rate)(waveform)
            
            mat = self._get_mat(waveform, temp_resample_rate) 

            mean_stat += torch.sum(mat, axis=0)
            var_stat += torch.sum(torch.square(mat), axis=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat

class AudioDataset(Dataset):
    def __init__(self, data_file, min_dur=0.0):
        self.items = []
        #with codecs.open(data_file, 'r', encoding='utf-8') as f:
        #    for line in f:
        #        arr = line.strip().split()
        #        # key, full_file_path
        #        self.items.append((arr[0], arr[1]))
        with open(data_file) as br:
            for aline in br.readlines():
                objs = json.loads(aline)
                if objs['duration'] >= min_dur:
                    self.items.append(objs['audio_filepath'])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract CMVN stats from training dataset')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='',
                        help='training yaml conf')
    parser.add_argument('--in_train_json_file', 
                        default=None, 
                        help='input json file for training')
    parser.add_argument('--out_cmvn',
                        default='global_cmvn',
                        help='output global cmvn file')
    parser.add_argument('--min_duration',
                        default=0.1,
                        type=float,
                        help='minimum wav duration for cmvn computing, shorter wavs are not used')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size for data processing')

    doc = "Print log after every log_interval audios are processed."
    parser.add_argument("--log_interval", type=int, default=1000, help=doc)
    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    #import ipdb; ipdb.set_trace()

    feat_dim = configs['model']['preprocessor']['features'] # 80 dim

    resample_rate = 0
    if 'resample_conf' in configs['model']:
        resample_rate = configs['model']['resample_conf']['resample_rate'] # NOTE
        print('using resample and new sample rate is {}'.format(resample_rate))

    collate_func = CollateFunc(feat_dim, resample_rate, configs)
    dataset = AudioDataset(args.in_train_json_file, min_dur=args.min_duration)
    batch_size = args.batch_size #2 #0

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             sampler=None,
                             num_workers=args.num_workers,
                             collate_fn=collate_func)

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_size

            if wav_number % args.log_interval == 0:
                print(f'processed {wav_number} wavs, {all_number} frames',
                      file=sys.stderr,
                      flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }
    #import ipdb; ipdb.set_trace()
    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))

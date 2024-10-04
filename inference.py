import glob
import os
import argparse
import json
import torch
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from datasets.dataset import mag_pha_stft, mag_pha_istft
from cal_metrics.compute_metrics import compute_metrics
from models.generator import MambaSEUNet
from models.pcs400 import cal_pcs
import soundfile as sf

import numpy as np

from utils.util import load_config


def load_checkpoint(filepath, device):
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

# handle audio slicing
def process_audio_segment(noisy_wav, model, device, n_fft, hop_size, win_size, compress_factor, sampling_rate, segment_size):
    segment_size = segment_size
    n_fft = n_fft
    hop_size = hop_size
    win_size = win_size
    compress_factor = compress_factor
    sampling_rate = sampling_rate

    norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
    noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
    orig_size = noisy_wav.size(1)

    # whether zeros need to be padded
    if noisy_wav.size(1) >= segment_size:
        num_segments = noisy_wav.size(1) // segment_size
        last_segment_size = noisy_wav.size(1) % segment_size
        if last_segment_size > 0:
            last_segment = noisy_wav[:, -segment_size:]
            noisy_wav = noisy_wav[:, :-last_segment_size]
            segments = torch.split(noisy_wav, segment_size, dim=1)
            segments = list(segments)
            segments.append(last_segment)
            reshapelast=1
        else:
            segments = torch.split(noisy_wav, segment_size, dim=1)
            reshapelast = 0

    else:
        # padding
        padded_zeros = torch.zeros(1, segment_size - noisy_wav.size(1)).to(device)
        noisy_wav = torch.cat((noisy_wav, padded_zeros), dim=1)
        segments = [noisy_wav]
        reshapelast = 0

    processed_segments = []

    for i, segment in enumerate(segments):

        noisy_amp, noisy_pha, noisy_com = mag_pha_stft(segment, n_fft, hop_size, win_size, compress_factor)
        amp_g, pha_g, com_g = model(noisy_amp.to(device, non_blocking=True), noisy_pha.to(device, non_blocking=True))
        audio_g = mag_pha_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
        audio_g = audio_g / norm_factor
        audio_g = audio_g.squeeze()
        if reshapelast == 1 and i == len(segments) - 2:
            audio_g = audio_g[ :-(segment_size-last_segment_size)]

        processed_segments.append(audio_g)

    processed_audio = torch.cat(processed_segments, dim=-1)
    print(processed_audio.size())

    processed_audio = processed_audio[:orig_size]
    print(processed_audio.size())
    print(orig_size)

    return processed_audio

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']
    segment_size = cfg['training_cfg']['segment_size']

    model = MambaSEUNet(cfg).to(device)
    state_dict = load_checkpoint(args.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    metrics_total = np.zeros(6)
    count = 0

    with torch.no_grad():
        for i, fname in enumerate(os.listdir(args.input_clean_wavs_dir)):
            print(fname, args.input_clean_wavs_dir)
            noisy_wav, _ = librosa.load(os.path.join(args.input_noisy_wavs_dir, fname), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            output_audio = process_audio_segment(noisy_wav, model, device, n_fft, hop_size, win_size, compress_factor, sampling_rate, segment_size)
            if args.post_processing_PCS == True:
                output_audio = cal_pcs(output_audio.squeeze().cpu().numpy())
            output_file = os.path.join(args.output_folder, fname)
            sf.write(output_file, output_audio.cpu().numpy(), sampling_rate, 'PCM_16')

            clean_wav, sr = librosa.load(os.path.join(args.input_clean_wavs_dir, fname), sr=sampling_rate)
            out1 = output_audio.cpu()
            output_audio = out1.numpy()

            metrics = compute_metrics(clean_wav, output_audio, sr, 0)
            metrics = np.array(metrics)
            metrics_total += metrics
            count += 1

        metrics_avg = metrics_total / count
        print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2],
              'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_clean_wavs_dir', default='../Data_16k/clean_test')
    parser.add_argument('--input_noisy_wavs_dir', default='../Data_16k/noisy_test')
    parser.add_argument('--output_folder', default='results/g_best')
    parser.add_argument('--config', default='recipes/Mamba-SEUNet/Mamba-SEUNet.yaml')
    parser.add_argument('--checkpoint_file', default='ckpts/g_best.pth')
    parser.add_argument('--post_processing_PCS', default=False)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        #device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    inference(args, device)

if __name__ == '__main__':
    main()


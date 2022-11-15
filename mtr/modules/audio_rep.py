import os
import torch
import torchaudio
import torch.nn as nn

class TFRep(nn.Module):
    """
    time-frequency represntation
    """
    def __init__(self, 
                sample_rate= 16000,
                f_min=0,
                f_max=8000,
                n_fft=1024,
                win_length=1024,
                hop_length = int(0.01 * 16000),
                n_mels = 128,
                power = None,
                pad= 0,
                normalized= False,
                center= True,
                pad_mode= "reflect"
                ):
        super(TFRep, self).__init__()
        self.window = torch.hann_window(win_length)
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            power = power
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels, 
            sample_rate,
            f_min,
            f_max,
            n_fft // 2 + 1)
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def melspec(self, wav):
        spec = self.spec_fn(wav)
        power_spec = spec.real.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec

    def spec(self, wav):
        spec = self.spec_fn(wav)
        real = spec.real
        imag = spec.imag
        power_spec = real.abs().pow(2)
        eps = 1e-10
        mag = torch.clamp(mag ** 2 + phase ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return power_spec, imag, mag, cos, sin

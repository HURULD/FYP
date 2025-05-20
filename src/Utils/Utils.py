from typing import Literal, Optional
import numpy as np

def GenSignal(sig_type:Literal['noise','sine','cosine','impulse'],len:float,sample_rate:int,frequency:Optional[int]=0,format:Literal['real','complex']='complex'):
    output=np.zeros(round(len*sample_rate))
    if sig_type == 'noise':
        if format == 'real' or format == 'complex':
            output = np.random.normal(0,1,size=round(len*sample_rate))
            return output
    if sig_type == 'sine':
        output = np.sin(np.arange(len*sample_rate)*frequency)
    else:
        raise NotImplementedError()
    
def MovingAverage(x:np.ndarray, window_size=10):
    out = np.array([])
    bins = np.array_split(x, np.floor(len(x)/window_size))
    bins = np.apply_over_axes(np.mean,np.array(bins),1)
    for bin in bins:
        np.append(out, bin)
    return out

def compute_rtf(mics_rir:np.ndarray, reference=0, n_fft=None):
    # Computes the RTF for all microphones from a given reference microphone
    # pick an FFT size (power of two) at least as large as the longer RIR
    if n_fft is None:
        L = max([len(mic) for mic in mics_rir])
        n_fft = 1
        while n_fft < L:
            n_fft *= 2
    
    # Compute the frequency-domain representation of each mic's RIR
    mics_h = [np.fft.rfft(mic, n=n_fft) for mic in mics_rir]
    
    # Avoid numerical issues
    EPS = 1e-12
    mics_h_safe = [np.where(np.abs(mic_h) < EPS, EPS, mic_h) for mic_h in mics_h]
    
    # Ratio
    RTF = [np.divide(mic_h, mics_h_safe[reference]) for mic_h in mics_h_safe]
    return RTF
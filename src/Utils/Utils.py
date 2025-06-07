from typing import Literal, Optional
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def GenSignal(sig_type:Literal['noise','sine','cosine','impulse'],len:float,sample_rate:int,frequency:Optional[int]=0,format:Literal['real','complex']='real'):
    """Generate a signal of single type with given length and sample rate.

    Args:
        sig_type (Literal[&#39;noise&#39;,&#39;sine&#39;,&#39;cosine&#39;,&#39;impulse&#39;]): _type_ of signal to generate.
        len (float): _length_ of the signal in seconds.
        sample_rate (int): _sample rate_ in Hz.
        frequency (Optional[int], optional): _frequency_ of sinusoid signal. Defaults to 0.
        format (Literal[&#39;real&#39;,&#39;complex&#39;], optional): Number format of data. Defaults to 'real'.

    Raises:
        NotImplementedError: _description_

    Returns:
        NDArray: Array of generated signal data.
    """
    
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
    return sliding_window_view(x, window_shape=window_size).mean(axis=-1)

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
    RTF = np.array([np.divide(mic_h, mics_h_safe[reference]) for mic_h in mics_h_safe])
    return RTF

def copy_room_spec_to_output(room_spec: dict, output_dir: str):
    """
    Copy the room specification to the output directory.
    
    Args:
        room_spec (dict): The room specification dictionary.
        output_dir (str): The output directory path.
    """
    import os
    import yaml
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, 'room_spec.yaml'), 'w') as f:
        yaml.dump(room_spec, f, default_flow_style=False)
        
def compute_multichannel_stft(signals, frame_len=512, hop_size=256, fft_len=512):
    """
    STFT of multichannel signal.
    Returns: STFTs with shape (M, F, T)
    """
    M, L = signals.shape
    stfts = []
    for m in range(M):
        _, _, Zxx = stft(signals[m], nperseg=frame_len, noverlap=frame_len-hop_size, nfft=fft_len)
        stfts.append(Zxx)
    return np.stack(stfts, axis=0)  # shape (M, F, T)

def compute_cpsd_matrices(stfts):
    """
    Computes CPSD matrices from multichannel STFT data.
    stfts: shape (M, F, T)
    Returns: CPSD matrices with shape (M, M, F, T)
    """
    M, F, T = stfts.shape
    cpsd = np.zeros((M, M, F, T), dtype=np.complex128)

    for t in range(T):
        for f in range(F):
            X = stfts[:, f, t].reshape(M, 1)  # (M, 1)
            cpsd[:, :, f, t] = X @ X.conj().T  # Outer product
    return cpsd
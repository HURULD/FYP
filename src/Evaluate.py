import numpy as np
import Estimators.AdaptiveFilters as AdaptiveFilters
import scipy.signal
import config_handler

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

def meansquared_error(x, y):
    # Handle case where x and y are not the same length
    if len(x) != len(y):
        #Compute MSE for the overlapping region
        overlap = min(len(x), len(y))
        x = x[:overlap]
        y = y[:overlap]
    return np.mean((x - y) ** 2)

def meansquared_error_delay_corrected(x, y):
    correlation = scipy.signal.correlate(x, y, mode='full')
    delay = np.argmax(correlation) - (len(y) - 1)
    x = np.roll(x, -delay)
    print(f"Estimated delay: {delay} samples, {delay / config_handler.get_config().audio.sample_rate} seconds")
    return meansquared_error(x, y)

def filter_total_mse(x, y, filter:AdaptiveFilters.AdaptiveFilter):
    y_hat = filter.apply_filter(x)
    return meansquared_error_delay_corrected(y, y_hat)

def filter_step_error(x, y, filter:AdaptiveFilters.AdaptiveFilter):
    y_hat = np.zeros(len(y))
    error = np.zeros(len(y))
    for i in range(len(x)):
        y_hat[i], error[i] = filter.step_update(x[i], y[i])
    return error
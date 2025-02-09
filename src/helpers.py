import numpy as np

def compute_rtf(h_mic1, h_mic2, n_fft=None):
    import numpy as np
    
    # pick an FFT size (power of two) at least as large as the longer RIR
    if n_fft is None:
        L = max(len(h_mic1), len(h_mic2))
        n_fft = 1
        while n_fft < L:
            n_fft *= 2
    
    # Compute the frequency-domain representation
    H1 = np.fft.rfft(h_mic1, n=n_fft)
    H2 = np.fft.rfft(h_mic2, n=n_fft)

    # Avoid numerical issues
    EPS = 1e-12
    H1_safe = np.where(np.abs(H1) < EPS, EPS, H1)
    
    # Ratio
    RTF = H2 / H1_safe
    
    return RTF

def meansquared_error(x, y):
    # Handle case where x and y are not the same length
    if len(x) != len(y):
        #Compute MSE for the overlapping region
        overlap = min(len(x), len(y))
        x = x[:overlap]
        y = y[:overlap]
    return np.mean((x - y) ** 2)
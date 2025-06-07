from . import AdaptiveFilters
import numpy as np
import scipy.signal

class RTFEstimator():
    
    def __init__(self, filter: AdaptiveFilters.AdaptiveFilter):
        self.filter = filter
        
    def estimate_rtf(self, mic_signal, reference_idx = 0):
        """
        Estimate the RTF from the microphone signal.
        :param mic_signal: The microphone signal of shape (n_mics, n_samples).
        :return: The estimated RTF. 
        """
        n_mics, n_samples = mic_signal.shape
        rtf_estimate = np.zeros((n_mics, self.filter.tap_count))
        for i in range(n_mics):
            if i == reference_idx:
                continue
            # Apply the adaptive filter to estimate the RRIR
            self.filter.w = np.zeros(self.filter.tap_count)
            _, _ = self.filter.full_simulate(mic_signal[i], mic_signal[reference_idx])
            rtf_estimate[i] = self.filter.w
        # Normalize the RRIR estimate
        #rtf_estimate /= np.linalg.norm(rtf_estimate, axis=1, keepdims=True)
        # FFT to get the RTF
        rtf_estimate = np.fft.rfft(rtf_estimate, axis=1)
        return rtf_estimate
    
    # TODO: Add a step update func for learning curve
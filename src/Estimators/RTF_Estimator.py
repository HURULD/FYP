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
            # Apply the adaptive filter to estimate the RRIR
            self.filter.w = np.zeros(self.filter.tap_count)
            _, _ = self.filter.full_simulate(mic_signal[i], mic_signal[reference_idx])
            rtf_estimate[i] = self.filter.w
        # Normalize the RRIR estimate
        #rtf_estimate /= np.linalg.norm(rtf_estimate, axis=1, keepdims=True)
        # FFT to get the RTF
        rtf_estimate = np.fft.rfft(rtf_estimate, axis=1)
        return rtf_estimate
    
    def estimate_rtf_step_update(self, mics, reference_idx=0):
        """
        Generator to estimate the RTF from the microphone signal using step updates.
        :param mics: The microphone signal of shape (n_mics, n_samples).
        :return: The estimated RTF.
        """
        n_mics, n_samples = mics.shape
        rtf_estimate = np.zeros((n_mics, self.filter.tap_count))
        for j in range(n_samples):
            for i in range(n_mics):
                # Apply the adaptive filter to estimate the RRIR
                self.filter.w = rtf_estimate[i]
                _, _ = self.filter.step_update(mics[i, j], mics[reference_idx, j])
                rtf_estimate[i] = self.filter.w
            yield rtf_estimate
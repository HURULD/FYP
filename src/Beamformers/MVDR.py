import numpy as np
from Utils import Utils
import pyroomacoustics as pra
import scipy.signal as sig
import scipy.fft as fft

class MVDR():
    """ 
    [WARNING] NOT IMPLEMENTED YET
    """
    def __init__(self, mic_pos, fs: int, taps:int, rtf: np.ndarray, noise_covariance: np.ndarray):
        self.mic_pos = mic_pos
        self.fs = fs
        self.tap_length = taps
        self.rtf = rtf
        self.noise_covariance = noise_covariance

    def apply_beamformer(self, signal: np.ndarray):
        """
        Compute the MVDR beamformer output for the given signal.
        :param signal: The mic signals of shape (n_mics, n_samples).
        :return: The MVDR beamformed output.
        """
        # fft the signal
        n_mics, n_samples = np.shape(signal)
        n_fft = 2 ** np.ceil(np.log2(n_samples)).astype(int)
        signal_fft = fft.rfft(signal, n=n_fft, axis=1)
        # Compute the covariance matrix
        covariance_matrix = np.cov(signal_fft)
        # Compute the inverse of the covariance matrix
        covariance_matrix_inv = np.linalg.inv(covariance_matrix + self.noise_covariance)
        # Compute the steering vector
        steering_vector = self.rtf / np.linalg.norm(self.rtf)
        # Compute the MVDR weights
        weights = covariance_matrix_inv @ steering_vector / (steering_vector.conj().T @ covariance_matrix_inv @ steering_vector)
        # Apply the weights to the signal
        beamformed_signal = np.sum(weights * signal_fft, axis=0)
        # Apply the inverse FFT to get the time-domain signal
        beamformed_signal_time = fft.irfft(beamformed_signal, n=n_fft)
        return beamformed_signal_time[:n_samples]
import numpy as np
from Utils import Utils
import pyroomacoustics as pra

class MVDR():
    """ 
    [WARNING] NOT IMPLEMENTED YET
    """
    def __init__(self, mic_pos, fs: int, taps:int, rtf: np.array):
        self.mic_pos = mic_pos
        self.fs = fs
        self.tap_length = taps
        self.rtf = rtf

    def compute_mvdr(self, signal: np.array):
        """
        Compute the MVDR beamformer output for the given signal.
        :param signal: The mic signals of shape (n_mics, n_samples).
        :return: The MVDR beamformed output.
        """
        # Placeholder for MVDR computation logic
        # This should include the actual MVDR algorithm implementation
        
        raise NotImplementedError("MVDR computation is not implemented yet.")
from Estimators import AdaptiveFilters
import Evaluate as Evaluate
import numpy as np
from scipy import signal
from scipy import fft
import logging
import matplotlib.pyplot as plt
from Visualisations.vis import fft_default_plot

log = logging.getLogger(__name__)

MSE_THRESHOLD = 10

def test_lms_lowpass_awgn():
    # Sample freq of 8kHz
    log.info("Starting LMS test with lowpass filter")
    input_noise = np.random.normal(0, 1, 16000) # 16000 (2s) zero mean WGN samples 
    test_noise = np.random.normal(0,1,16000)
    # Reference FIR filter from http://t-filter.engineerjs.com/ with corner freq at 400Hz
    reference_filter_taps = [   
                                -0.02010411882885732, -0.05842798004352509, -0.061178403647821976,
                                -0.010939393385338943, 0.05125096443534972, 0.033220867678947885,
                                -0.05655276971833928, -0.08565500737264514, 0.0633795996605449,
                                0.31085440365663597, 0.4344309124179415, 0.31085440365663597,
                                0.0633795996605449, -0.08565500737264514, -0.05655276971833928,
                                0.033220867678947885, 0.05125096443534972, -0.010939393385338943,
                                -0.061178403647821976, -0.05842798004352509, -0.02010411882885732
                            ]
    
    reference_signal = signal.lfilter(reference_filter_taps, 1, input_noise)
    test_filter = AdaptiveFilters.LMS(len(reference_filter_taps), 0.00001)
    filter_output, error = test_filter.full_simulate(test_noise,reference_signal)
    mse = Evaluate.meansquared_error(reference_signal,filter_output)
    log.info("MSE: %d",mse)
    #log.info("Tap difference: %s", [x[0]-x[1] for x in zip(reference_filter_taps, test_filter.w)])
    assert mse <= MSE_THRESHOLD
    
def test_nlms_lowpass_awgn():
    # Sample freq of 8kHz
    log.info("Starting NLMS test with lowpass filter")
    input_noise = np.random.normal(0, 1, 16000) # 16000 (2s) zero mean WGN samples 
    test_noise = np.random.normal(0,1,16000)
    # Reference FIR filter from http://t-filter.engineerjs.com/ with corner freq at 400Hz
    reference_filter_taps = [   
                                -0.02010411882885732, -0.05842798004352509, -0.061178403647821976,
                                -0.010939393385338943, 0.05125096443534972, 0.033220867678947885,
                                -0.05655276971833928, -0.08565500737264514, 0.0633795996605449,
                                0.31085440365663597, 0.4344309124179415, 0.31085440365663597,
                                0.0633795996605449, -0.08565500737264514, -0.05655276971833928,
                                0.033220867678947885, 0.05125096443534972, -0.010939393385338943,
                                -0.061178403647821976, -0.05842798004352509, -0.02010411882885732
                            ]
    
    reference_signal = signal.lfilter(reference_filter_taps, 1, input_noise)
    fft_default_plot(reference_signal,8000)
    test_filter = AdaptiveFilters.NLMS(len(reference_filter_taps), 1)
    filter_output, error = test_filter.full_simulate(test_noise,reference_signal)
    fft_default_plot(filter_output,8000)
    mse = Evaluate.meansquared_error(reference_signal,filter_output)
    log.info("MSE: %d",mse)
    #log.info("Tap difference: %s", [x[0]-x[1] for x in zip(reference_filter_taps, test_filter.w)])
    assert mse <= MSE_THRESHOLD
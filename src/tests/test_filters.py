from Estimators import AdaptiveFilters
import Evaluate as Evaluate
import numpy as np
from scipy import signal
from scipy import fft
import logging
import matplotlib.pyplot as plt
import pytest
from Visualisations.vis import fft_default_plot

log = logging.getLogger(__name__)

MSE_THRESHOLD = 1.5

@pytest.mark.parametrize("filter_class, mu", [
    (AdaptiveFilters.LMS, 0.001),
    (AdaptiveFilters.NLMS, 1),
    (AdaptiveFilters.PNLMS, 1),
])
class TestAdaptiveFilters:
    def test_filter_lowpass_awgn(self, filter_class: AdaptiveFilters.AdaptiveFilter, mu):
        log.info("Starting %s test with lowpass filter", filter_class.__name__)
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
        test_filter = filter_class(len(reference_filter_taps)+50, mu)
        filter_output, error = test_filter.full_simulate(test_noise,reference_signal)
        mse = Evaluate.meansquared_error(reference_signal,filter_output)
        log.info("MSE: %d",mse)
        #log.info("Tap difference: %s", [x[0]-x[1] for x in zip(reference_filter_taps, test_filter.w)])
        assert mse <= MSE_THRESHOLD
        
    def test_filter_highpass_awgn(self, filter_class: AdaptiveFilters.AdaptiveFilter, mu):
        log.info("Starting %s test with highpass filter", filter_class.__name__)
        input_noise = np.random.normal(0, 1, 16000) # 16000 (2s) zero mean WGN samples 
        test_noise = np.random.normal(0,1,16000)
        # Reference FIR filter from http://t-filter.engineerjs.com/ with corner freq at 400Hz
        reference_filter_taps = [   
                                    0.02857983994169657,  -0.07328836181028245,  0.04512928732568175,
                                    0.03422632401030237,  -0.034724262386629436,  -0.05343090761376418,
                                    0.032914528649623416,  0.09880818246272206,  -0.034135422078843417,
                                    -0.3160339484471911,  0.5341936566511765,  -0.3160339484471911,
                                    -0.034135422078843417,  0.09880818246272206,  0.032914528649623416,
                                    -0.05343090761376418,  -0.034724262386629436,  0.03422632401030237,
                                    0.04512928732568175,  -0.07328836181028245,  0.02857983994169657
                                ]
        
        reference_signal = signal.lfilter(reference_filter_taps, 1, input_noise)
        test_filter = filter_class(len(reference_filter_taps), mu)
        filter_output, error = test_filter.full_simulate(test_noise,reference_signal)
        mse = Evaluate.meansquared_error(reference_signal,filter_output)
        log.info("MSE: %d",mse)
        #log.info("Tap difference: %s", [x[0]-x[1] for x in zip(reference_filter_taps, test_filter.w)])
        assert mse <= MSE_THRESHOLD
        
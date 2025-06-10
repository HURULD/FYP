from Estimators import AdaptiveFilters
import Evaluate as Evaluate
import numpy as np
from scipy import signal
from scipy import fft
import logging
import matplotlib
# Use the pgf backend (must be set before pyplot imported)
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",  # or "lualatex" or "xelatex" if you're using those
    "text.usetex": True,
    "font.family": "serif",       # Match your document font
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
import matplotlib.pyplot as plt
import pytest
from Visualisations.vis import fft_default_plot
import random
from Utils import Utils

log = logging.getLogger(__name__)

MSE_THRESHOLD = 2

@pytest.mark.parametrize("filter_class, mu", [
    (AdaptiveFilters.LMS, 0.0005),
    (AdaptiveFilters.NLMS, 0.5),
    (AdaptiveFilters.PNLMS, 0.5),
    (AdaptiveFilters.IPNLMS,0.5)
])
class TestAdaptiveFilters:
    def test_filter_lowpass_awgn(self, filter_class: AdaptiveFilters.AdaptiveFilter, mu):
        log.info("Starting %s test with lowpass filter", filter_class.__name__)
        input_noise = Utils.GenSignal("noise",4,2000)
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
        filter_output, error = test_filter.full_simulate(input_noise,reference_signal)
        mse = Evaluate.meansquared_error(reference_signal,filter_output)
        log.info("MSE: %d",mse)
        log.info("Tap difference (learning curve): %s", np.mean([(x[0]-x[1])**2 for x in zip(reference_filter_taps, test_filter.w)]))
        
        assert mse <= MSE_THRESHOLD
        
    def test_filter_highpass_awgn(self, filter_class: AdaptiveFilters.AdaptiveFilter, mu):
        log.info("Starting %s test with highpass filter", filter_class.__name__)
        input_noise = Utils.GenSignal('noise', 4, 2000) # 16000 (8s) zero mean WGN samples 
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
        filter_output, error = test_filter.full_simulate(input_noise,reference_signal)
        mse = Evaluate.meansquared_error(reference_signal,filter_output)
        log.info("MSE: %d",mse)
        #log.info("Tap difference: %s", [x[0]-x[1] for x in zip(reference_filter_taps, test_filter.w)])
        assert mse <= MSE_THRESHOLD
    
    def test_filter_4tap_awgn(self, filter_class: AdaptiveFilters.AdaptiveFilter, mu):
        log.info("Starting %s test with arbitrary 4 tap filter", filter_class.__name__)
        input_noise = Utils.GenSignal('noise',4,2000)
        # Reference filter
        reference_filter_taps = [(random.random() * -10) + 5 for _ in range(4)]
        reference_signal = signal.lfilter(reference_filter_taps, 1, input_noise)
        test_filter:AdaptiveFilters.AdaptiveFilter = filter_class(len(reference_filter_taps), mu)
        tap_mse = [ [] for _ in range(len(reference_filter_taps))]
        y_hat = []
        e = []
        for x,y in zip(input_noise, reference_signal):
            y_h, error = test_filter.step_update(x, y)
            y_hat.append(y_h)
            e.append(error)
            for i in range(len(reference_filter_taps)):
                tap_mse[i].append((test_filter.w[i]))
        mse = Evaluate.meansquared_error(reference_signal,y_hat)
        # for i in range(len(reference_filter_taps)):
        #     plt.plot(np.arange(16000), [reference_filter_taps[i]]*16000, c=plt.get_cmap('tab10').colors[i])
        #     plt.plot(tap_mse[i], linestyle='dashed')
        # plt.show()
        assert mse <= MSE_THRESHOLD
        
    def test_filter_learning_curve(self, filter_class: AdaptiveFilters.AdaptiveFilter, mu):
        log.info("Testing %s learning curve, error should generally decrease", filter_class.__name__)
        input_noise = Utils.GenSignal('noise',4,2000)
        # Reference filter
        reference_filter_taps = [(random.random() * -2) + 1 for _ in range(1024)]
        reference_signal = signal.lfilter(reference_filter_taps, 1, input_noise)
        test_filter:AdaptiveFilters.AdaptiveFilter = filter_class(len(reference_filter_taps), mu)
        tap_mse = np.zeros(len(input_noise))
        j = 0
        for i, (x,y) in enumerate(zip(input_noise, reference_signal)):
            y_h, error = test_filter.step_update(x, y)
            tap_mse[i] = Evaluate.meansquared_error(reference_filter_taps, test_filter.w)
            if tap_mse[i] > 1e5:
                j = j+1
                log.warning(f"High tap MSE at sample {i}: {tap_mse[i]}")
        log.warning(f"High tap MSE count: {j}")
        fig, ax = plt.subplots(figsize=(2.8, 2.3))
        ax.plot(Utils.MovingAverage(tap_mse))
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlim(left=0, right=len(input_noise))
        ax.set_ylim(bottom=0, top=0.35)
        ax.set_xlabel("Sample")
        ax.set_ylabel("MSE")
        plt.tight_layout()
        plt.savefig(f'learning-curve-{filter_class.__name__}.pgf', format='pgf')
        
        
class TestCovariance:
    
    def test_covariance_whitening(self):
        log.info("Testing covariance whitening")
        # Generate a random noise signal
        noisy_cpsd = np.random.rand(10, 10, 5, 5)
        raise NotImplementedError("Covariance whitening test not implemented yet")
    
    def test_covariance_subtraction(self):
        raise NotImplementedError("Covariance subtraction test not implemented yet")
from abc import ABC, abstractmethod
import numpy as np
import numpy as np
from scipy import signal
import pyroomacoustics as pra
import config_handler as conf

class AdaptiveFilter(ABC):
    @abstractmethod
    def step_update(self, x_sample:float, y_sample:float) -> tuple[float, float]:
        """Update filter weights with a new input sample."""
        pass

    @abstractmethod
    def apply_filter(self, x:np.ndarray) -> np.ndarray:
        """Apply the trained filter to an input signal."""
        pass


class LMS(AdaptiveFilter):
    def __init__(self, tap_count, mu):
        self.w = np.zeros(tap_count)
        self.tap_count = tap_count
        self.mu = mu
        self._delay_line = np.zeros(tap_count)
        
    def full_simulate(self, x, y):
        self.reset()
        y_hat = np.zeros(len(y))
        error = np.zeros(len(y))
        for i in range(len(x)):
            y_hat[i], error[i] = self.step_update(x[i], y[i])
        return y_hat, error
    
    def step_update(self, x_sample, y_sample):
        self._delay_line = np.roll(self._delay_line,1)
        self._delay_line[0] = x_sample
        y_hat = np.dot(self.w, self._delay_line)
        error = y_sample - y_hat
        self.w = np.add(self.w, (self.mu * error * self._delay_line))
        return y_hat, error
    
    def apply_filter(self, x):
        return signal.lfilter(self.w, 1, x)
    
    def reset(self):
        self.w = np.zeros(self.tap_count)
        self._delay_line = np.zeros(self.tap_count)
        
class NLMS(LMS):
    def __init__(self, tap_count, mu):
        super().__init__(tap_count=tap_count,mu=mu)

    def step_update(self, x_sample, y_sample):
        self._delay_line = np.roll(self._delay_line,1)
        self._delay_line[0] = x_sample
        y_hat = np.dot(np.conjugate(self.w), self._delay_line)
        error = y_sample - y_hat
        norm_factor = np.dot(np.conjugate(self._delay_line), self._delay_line) + 1e-10 # Normalise based on signal power (+ a little bit to avoid errors)
        self.w = np.add(self.w, ((self.mu / norm_factor) * error * self._delay_line))
        return y_hat, error
from abc import ABC, abstractmethod
import numpy as np
import numpy as np
from scipy import signal
import pyroomacoustics as pra
import config_handler as conf

class AdaptiveFilter(ABC):
    @abstractmethod
    def update(self, x_sample:float, y_sample:float) -> tuple[float, float]:
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
        self.delay_line = np.zeros(tap_count)
        
    def full_simulate(self, x, y):
        self.reset()
        y_hat = np.zeros(len(y))
        error = np.zeros(len(y))
        for i in range(len(self.x)):
            y_hat[i], error[i] = self.step_update(x[i], y[i])
        return y_hat, error
    
    def step_update(self, x_sample, y_sample):
        self.delay_line = np.roll(self.delay_line, 1)
        self.delay_line[0] = x_sample
        y_hat = np.dot(self.w, self.delay_line)
        error = y_sample - y_hat
        self.w += self.mu * error * self.delay_line
        return y_hat, error
    
    def apply_filter(self, x):
        return signal.lfilter(self.w, 1, x)
    
    def reset(self):
        self.w = np.zeros(self.tap_count)
        self.delay_line = np.zeros(self.tap_count)
from abc import ABC, abstractmethod
import numpy as np
import numpy as np
from scipy import signal
import pyroomacoustics as pra
import config_handler as conf
import logging

log = logging.getLogger(__name__)
class AdaptiveFilter(ABC):
    
    w = np.array([])
    
    @abstractmethod
    def step_update(self, x_sample:float, y_sample:float) -> tuple[float, float]:
        """Update filter weights with a new input sample."""
        pass

    @abstractmethod
    def apply_filter(self, x:np.ndarray) -> np.ndarray:
        """Apply the trained filter to an input signal."""
        pass

    def apply_filter(self, x):
        return signal.lfilter(self.w, 1, x)

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
        if abs(error) > 1e2 or np.any(np.abs(self.w) > 1e3):
            print(f"Huge update at sample")
            print(f"x: {x_sample}, error: {error}, w: {self.w}")
        return y_hat, error
    
    def reset(self):
        self.w = np.zeros(self.tap_count)
        self._delay_line = np.zeros(self.tap_count)
        
class NLMS(LMS):
    def __init__(self, tap_count, mu):
        super().__init__(tap_count=tap_count,mu=mu)
        
    @property
    def norm_factor(self):
        return np.dot(np.conjugate(self._delay_line), self._delay_line) + 1e-12 # Normalise based on signal power (+ a little bit to avoid errors)

    def step_update(self, x_sample, y_sample):
        self._delay_line = np.roll(self._delay_line,1)
        self._delay_line[0] = x_sample
        y_hat = np.dot(np.conjugate(self.w), self._delay_line)
        error = y_sample - y_hat
        self.w = np.add(self.w, ((self.mu / self.norm_factor) * error * self._delay_line))
        return y_hat, error
    
class PNLMS(NLMS):
    def __init__(self, tap_count, mu, delta=0.01, p=0.01):
        self._delta = delta
        self._p = p
        super().__init__(tap_count=tap_count,mu=mu)
        
    def step_update(self, x_sample, y_sample):
        self._delay_line= np.roll(self._delay_line, 1)
        self._delay_line[0] = x_sample
        y_hat = np.dot(np.conjugate(self.w), self._delay_line)
        error = y_sample - y_hat
        w_max = np.amax(np.abs(self.w))
        gamma = (lambda w_i: (np.maximum(self._p * max(self._delta, w_max), np.abs(w_i)))) (self.w)
        g = gamma / np.sum(gamma)
        self.w = np.add(self.w, (self.mu * g * self._delay_line * error) / (np.sum(g * self._delay_line**2) + self._delta))
        return y_hat, error

class IPNLMS(PNLMS):
    def __init__(self, tap_count, mu, delta=0.01, p=0.01, alpha = -0.5):
        self.alpha = alpha
        super().__init__(tap_count=tap_count,mu=mu,delta=delta,p=p)
        self._delta = 1-self.alpha / (2*self.tap_count) *self._delta
    
    def step_update(self, x_sample, y_sample):
        self._delay_line = np.roll(self._delay_line, 1)
        self._delay_line[0] = x_sample
        y_hat = np.dot(np.conjugate(self.w), self._delay_line)
        error = y_sample - y_hat
        w_norm = np.sum(np.abs(self.w))
        k = (lambda w_i : ((1-self.alpha) / (2*self.tap_count)) + (1+self.alpha)*((np.abs(w_i))/(2*w_norm + 1e-12))) (self.w)
        self.w = np.add(self.w, (self.mu * k * self._delay_line * error) / (np.sum(k * self._delay_line**2) + self._delta))
        return y_hat, error
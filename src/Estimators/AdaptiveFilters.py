from abc import ABC, abstractmethod
import numpy as np
import numpy as np
from scipy import signal
import pyroomacoustics as pra
import config_handler as conf

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
        return y_hat, error
    
    def reset(self):
        self.w = np.zeros(self.tap_count)
        self._delay_line = np.zeros(self.tap_count)
        
class NLMS(LMS):
    def __init__(self, tap_count, mu):
        super().__init__(tap_count=tap_count,mu=mu)
        
    @property
    def norm_factor(self):
        return np.dot(np.conjugate(self._delay_line), self._delay_line) + 1e-10 # Normalise based on signal power (+ a little bit to avoid errors)

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
        l = np.amax(np.abs(self.w))
        l = max(l, self._delta)
        g = np.array([max((self._p*l),np.abs(w_n)) for w_n in self.w])
        g_mean = np.mean(g)
        sigma = np.mean(np.square(self._delay_line))
        self.w = np.add(self.w, (self.mu / self.tap_count)*(g/g_mean)*((error*self._delay_line)/sigma))
        return y_hat, error

class IPNLMS(PNLMS):
    def __init__(self, tap_count, mu, delta=0.01, p=0.01, alpha = -0.5):
        self.alpha = alpha
        super().__init__(tap_count=tap_count,mu=mu,delta=delta,p=p)
    
    def step_update(self, x_sample, y_sample):
        self._delay_line = np.roll(self._delay_line, 1)
        self._delay_line[0] = x_sample
        y_hat = np.dot(np.conjugate(self.w), self._delay_line)
        error = y_sample - y_hat
        w_norm_mean = np.sum(np.abs(self.w)) / self.tap_count
        K = np.diag([(1-self.alpha) * w_norm_mean + (1+self.alpha) * np.abs(w_i) for w_i in self.w])
        self.w = np.add(self.w, (self.mu * K * self._delay_line * error) / (self._delay_line.T * K * self._delay_line + self._delta))
        return y_hat, error
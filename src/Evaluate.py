import numpy as np
import Estimators.AdaptiveFilters as AdaptiveFilters
import scipy.signal
import config_handler

def meansquared_error(x, y):
    # Handle case where x and y are not the same length
    if len(x) != len(y):
        #Compute MSE for the overlapping region
        overlap = min(len(x), len(y))
        x = x[:overlap]
        y = y[:overlap]
    return np.mean((x - y) ** 2)

def meansquared_error_delay_corrected(x, y):
    correlation = scipy.signal.correlate(x, y, mode='full')
    delay = np.argmax(correlation) - (len(y) - 1)
    x = np.roll(x, -delay)
    print(f"Estimated delay: {delay} samples, {delay / config_handler.get_config().audio.sample_rate} seconds")
    return meansquared_error(x, y)

def filter_total_mse(x, y, filter:AdaptiveFilters.AdaptiveFilter):
    y_hat = filter.apply_filter(x)
    return meansquared_error_delay_corrected(y, y_hat)

def filter_step_error(x, y, filter:AdaptiveFilters.AdaptiveFilter):
    y_hat = np.zeros(len(y))
    error = np.zeros(len(y))
    for i in range(len(x)):
        y_hat[i], error[i] = filter.step_update(x[i], y[i])
    return y_hat, error

def filter_learning_curve(x,y,filter:AdaptiveFilters.AdaptiveFilter, reference_taps:np.ndarray):
    tap_mse = np.zeros(len(x))
    for i in range(len(x)):
        filter.step_update(x[i], y[i])
        tap_mse[i] = meansquared_error(filter.w, reference_taps)
    return tap_mse
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

def filter_learning_curve_mse(x,y,filter:AdaptiveFilters.AdaptiveFilter, reference_taps:np.ndarray):
    tap_mse = np.zeros(len(x))
    for i in range(len(x)):
        filter.step_update(x[i], y[i])
        tap_mse[i] = meansquared_error(filter.w, reference_taps)
    return tap_mse

def npm(h,h_hat):
    return normalised_projection_misalignment(h, h_hat)

def normalised_projection_misalignment(h, h_hat):
    """Normalized Projection Misalignment 

    Args:
        h (np.NDArray): true impulse responses [M x L]
        h_hat (np.NDArray): estimated impulse responses [M x L]
    Returns:
        npm_val : Normalize Projection Misalignment
        
    References:
        [1] D. R. Morgan, J. Benesty and M. M. Sondhi, "On the evaluation of
            estimated impulse responses," IEEE Signal Processing Lett., Vol. 5, No.
            7, pp. 174-176 Jul 1998.

        [2] Y. Huang and J. Benesty, "Frequency-Domain adaptive approaches to
            blind multi-channel identification," IEEE Trans. Sig. Process. Vol. 51
            No. 1, pp/ 11-24, Jan 2003.

    Authors:
        N.D. Gaubitch and E.A.P. Habets 

    History:
        2004-04-26 - Initial version by NG
        
        2009-10-28 - reshape when the size of h_hat differs from h
        
        2025-06-03 - Translated into Python by Harry Griffiths

    Copyright (C) Imperial College London 2009-2010
    Version: $Id: npm.m 425 2011-08-12 09:15:01Z mrt102
    """
    if np.shape(h_hat)[1]-np.shape(h)[1] > 0:
        h = np.concat((h, np.zeros((np.shape(h)[0], np.shape(h_hat)[1]-np.shape(h)[1]))), axis=1)
    elif np.shape(h_hat)[1]-np.shape(h)[1] < 0:
        h_hat = np.concat((h_hat, np.zeros((np.shape(h_hat)[0], np.shape(h)[1]-np.shape(h_hat)[1]))), axis=1)
    
    if np.shape(h_hat)[0] <= np.shape(h)[0]:
        h_v = np.reshape(h[0:np.shape(h_hat)[0]],(-1,1),'F')
        h_hat_v = np.reshape(h_hat,(-1,1),'F')
    else:
        h_v = np.reshape(h,(-1,1),'F')
        h_hat_v = np.reshape(h_hat,(-1,1),'F')

    epsilon = h_v-np.squeeze((h_v.T@h_hat_v)/(h_hat_v.T@h_hat_v))*h_hat_v
    npm_val = np.linalg.norm(epsilon)/np.linalg.norm(h_v)
    
    return npm_val



def filter_learning_curve_npm(x,y,filter:AdaptiveFilters.AdaptiveFilter, reference_taps:np.ndarray):
    tap_npm = np.zeros(len(x))
    for i in range(len(x)):
        filter.step_update(x[i], y[i])
        tap_npm[i] = npm(filter.w, reference_taps)
    return tap_npm

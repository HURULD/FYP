# Adapted from https://github.com/Screeen/SVD-direct/tree/main
# G. Bologni, R. C. Hendriks and R. Heusdens, "Wideband Relative Transfer Function (RTF) 
# Estimation Exploiting Frequency Correlations," in IEEE Transactions on Audio, Speech and 
# Language Processing, vol. 33, pp. 731-747, 2025, doi: 10.1109/TASLPRO.2025.3533371.

import scipy.linalg
import scipy
import numpy as np
import logging

logger = logging.getLogger(__name__)

eps = 1e-12  # small value to avoid division by zero

def estimate_rtf_covariance_whitening(noise_cpsd, noisy_cpsd, use_cholesky=True) -> np.array:
    """
    1) Markovich, Shmulik, Sharon Gannot, and Israel Cohen. ‘Multichannel Eigenspace Beamforming in a Reverberant Noisy
    Environment With Multiple Interfering Speech Signals’. 2009

    2) Markovich-Golan, Shmulik, and Sharon Gannot. ‘Performance Analysis of the Covariance Subtraction Method for
    Relative Transfer Function Estimation and Comparison to the Covariance Whitening Method’. 2015
    """

    num_mics, _, num_freqs, num_time_frames = noise_cpsd.shape
    
    rtfs = np.ones((num_mics, num_freqs, num_time_frames), dtype=np.complex128)
    time_frames = range(num_time_frames)
    if use_cholesky:
        for tt in time_frames:
            if num_time_frames > 1:
                print(f"Processing time frame {tt + 1}/{num_time_frames}", end='\r')
            for kk in range(num_freqs):
                rtfs[..., kk, tt] = covariance_whitening_cholesky(noise_cpsd[..., kk, tt],
                                                                                noisy_cpsd[..., kk, tt])
    else:
        for tt in time_frames:
            if num_time_frames > 1:
                print(f"Processing time frame {tt + 1}/{num_time_frames}", end='\r')
            for kk in range(num_freqs):
                rtfs[..., kk, tt] = covariance_whitening_generalized_eig(noise_cpsd[..., kk, tt],
                                                                                        noisy_cpsd[..., kk, tt],
                                                                                        hermitian_matrices=True)

    return rtfs

def covariance_whitening_generalized_eig(noise_cpsd, noisy_cpsd, hermitian_matrices=False, normalize_rtf=True):

    if not hermitian_matrices:
        eigenvals, eigves = \
            scipy.linalg.eig(noisy_cpsd, noise_cpsd, check_finite=False, left=False, right=True)
    else:
        try:
            eigenvals, eigves = \
                scipy.linalg.eigh(noisy_cpsd, noise_cpsd, check_finite=False, driver='gvd')
        except np.linalg.LinAlgError:
            logger.warn(f"covariance_whitening_generalized_eig: LinAlgError, returning zeros")
            return np.zeros_like(noise_cpsd[0, ...])

    _, max_right_eigve = sort_eigenvectors_get_major(eigenvals, eigves)
    rtf = noise_cpsd @ max_right_eigve
    if normalize_rtf:
        rtf = normalize_to_1(rtf)

    return np.squeeze(rtf)

def covariance_whitening_generalized_eig_explicit_inversion(noise_cpsd, noisy_cpsd):

    # even if A, B are Hermitian, B^{-1} A is NOT hermitian!
    eigenvals, eigves = \
        scipy.linalg.eig(np.linalg.inv(noise_cpsd) @ noisy_cpsd, check_finite=False)

    _, max_eigve = sort_eigenvectors_get_major(eigenvals, eigves)
    max_eigve = noise_cpsd @ max_eigve
    rtf = normalize_to_1(max_eigve)

    return np.squeeze(rtf)

def covariance_whitening_cholesky(noise_cpsd, noisy_cpsd) -> np.array:
    # `noise_cpsd` must be Hermitian (symmetric if real-valued) and positive-definite

    _, maj_eigve_noisy_whitened, noise_cpsd_sqrt = get_eigenvectors_whitened_noisy_cov(noise_cpsd, noisy_cpsd)
    rtf = noise_cpsd_sqrt @ maj_eigve_noisy_whitened  # transform back from whitened domain
    rtf = normalize_to_1(rtf)

    return np.squeeze(rtf)

def normalize_to_1(eigve_single_column, idx_ref_mic=0):
    # normalize vector to get 1 at reference microphone
    if np.abs(eigve_single_column[idx_ref_mic]) < eps:
        eigve_normalized = np.zeros_like(eigve_single_column)
    else:
        eigve_normalized = eigve_single_column / eigve_single_column[idx_ref_mic]

    return eigve_normalized

def get_eigenvectors_whitened_noisy_cov(noise_cpsd, noisy_cpsd, how_many=1):
    noise_cpsd_sqrt, noisy_cpsd_whitened = whiten_covariance(noise_cpsd, noisy_cpsd)
    eigva_noisy_whitened, eigve_noisy_whitened = np.linalg.eigh(noisy_cpsd_whitened)
    maj_eigva, maj_eigve_whitened = sort_eigenvectors_get_major(eigva_noisy_whitened,
                                                                                eigve_noisy_whitened, how_many)

    return maj_eigva, maj_eigve_whitened, noise_cpsd_sqrt

def whiten_covariance(noise_cpsd, noisy_cpsd):
    """
    1) Perform Cholesky decomposition on noise_cpsd: noise_cpsd = L @ L.conj().T
    2) Calculate whitened covariance R_white = L^-1 @  noisy_cpsd @ (L^(-1))^H
    :param noise_cpsd: noise spatial covariance
    :param noisy_cpsd: noisy spatial covariance
    :return: Cholesky factor L, whitened noisy spatial covariance
    """
    noise_cpsd_sqrt = np.linalg.cholesky(noise_cpsd)
    noise_cpsd_sqrt_inv = np.linalg.inv(noise_cpsd_sqrt)
    noisy_cpsd_whitened = noise_cpsd_sqrt_inv @ noisy_cpsd @ noise_cpsd_sqrt_inv.conj().T
    # assert u.is_hermitian(noisy_cpsd_whitened)
    return noise_cpsd_sqrt, noisy_cpsd_whitened

def sort_eigenvectors_get_major(eigva, eigve, num_to_keep=1, squeeze=True):
    """
    Return eigenvector corresponding to eigenvalue with maximum norm. if eigenvalues are not ALL finite, return NaN
    """

    if num_to_keep == -1:
        num_to_keep = len(eigva)  # keep all eigenvectors

    if not np.all(np.isfinite(eigva)):
        return np.ones_like(eigva)[:num_to_keep] * np.nan, np.ones_like(eigve)[:, :num_to_keep] * np.nan

    # Sort eigenvalues and eigenvectors in ascending order
    idx_largest_eigvas_sorted = np.argsort(np.real(eigva))
    eigva, eigve = eigva[idx_largest_eigvas_sorted], eigve[:, idx_largest_eigvas_sorted]

    if squeeze:
        return np.squeeze(eigva[-num_to_keep:]), np.squeeze(eigve[:, -num_to_keep:])
    else:
        return eigva[-num_to_keep:], eigve[:, -num_to_keep:]
    

def covariance_subtraction_first_column(phi_xx_tt, reference_mic=0):
    return np.squeeze(phi_xx_tt[:, reference_mic] / (eps + phi_xx_tt[reference_mic, reference_mic]))

# for single source, RTF corresponds to eigenvector corresponding to largest eigenvalue
def covariance_subtraction_eigve(phi_xx_tt):
    eigva, eigve = scipy.linalg.eigh(phi_xx_tt, check_finite=False)
    _, rtf = sort_eigenvectors_get_major(eigva, eigve)
    rtf = normalize_to_1(rtf)
    return rtf

def estimate_rtf_covariance_subtraction(clean_speech_cpsd, use_first_column=True) -> np.array:

    print("estimate_rtf_covariance_subtraction...")

    num_mics, _, num_freqs, num_time_frames = clean_speech_cpsd.shape
    rtfs = np.ones((num_mics, num_freqs, num_time_frames), dtype=complex)
    for tt in range(num_time_frames):
        for kk in range(num_freqs):
            rtfs[..., kk, tt] = covariance_subtraction_internal(clean_speech_cpsd[..., kk, tt],
                                                                        use_first_column)

    return rtfs

def covariance_subtraction_internal(clean_speech_cpsd, use_first_column=False) -> np.array:
    if not np.alltrue(np.diag(clean_speech_cpsd) >= 0):
        return np.ones((clean_speech_cpsd.shape[0],), dtype=complex) * np.nan
    else:
        if use_first_column:
            return covariance_subtraction_first_column(clean_speech_cpsd)
        else:
            return covariance_subtraction_eigve(clean_speech_cpsd)
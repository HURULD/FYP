import math
from Estimators import RTF_Estimator, Covariance
import Evaluate as Eval
import numpy as np
from Utils import Utils
import csv
import config_handler


def rtf_accuracy_filter(adaptive_filter, rtf, mics, args):
    rtf_estimator = RTF_Estimator.RTFEstimator(adaptive_filter)
    
    h_hat = rtf_estimator.estimate_rtf(mics, reference_idx=0)
    
    npm = Eval.npm(rtf, h_hat)
    print(f"Normalized Projection Misalignment: {npm}")
    mses = np.array([Eval.meansquared_error_delay_corrected(rtf[i], h_hat[i]) for i in range(h_hat.shape[0])])
    print(f"Mean Squared Errors per mic: {np.abs(mses)}")
    with open(f"{args.output}/rtf_accuracy_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([f"{args.filter[0]}_{args.filter[1]}_{args.filter[2]}", np.mean(np.abs(mses)), npm])
    # Save the estimated RTF
    np.save(f"{args.output}/estimated_rtf_filter_{args.filter[0]}_{args.filter[1]}_{args.filter[2]}", h_hat)
    print("Estimated RTF saved to", f"{args.output}/estimated_rtf_filter_{args.filter[0]}_{args.filter[1]}_{args.filter[2]}.npy")
    
    return h_hat

def rtf_accuracy_covariance_whitening_identity(rtf, mics, args):
    noisy_cpsd = Utils.compute_cpsd_matrices(Utils.compute_multichannel_stft(mics))
    # Identity matrix of same shape, with same frequency bins and time frames (all ident) (Isotropic noise)
    noise_cpsd = np.zeros_like(noisy_cpsd)
    for i in range(noisy_cpsd.shape[2]):
        for j in range(noisy_cpsd.shape[3]):
            noise_cpsd[:,:, i, j] = np.identity(noisy_cpsd.shape[0])
    # Covariance whitening step
    h_hat = Covariance.estimate_rtf_covariance_whitening(noise_cpsd, noisy_cpsd)
    # Interpolate to the same shape as rtf
    h_hat = Utils.interpolate_stft_to_fft(h_hat, rtf.shape[1])
    
    npm = Eval.npm(rtf, h_hat)
    print(f"Normalized Projection Misalignment: {npm}")
    mses = np.array([Eval.meansquared_error_delay_corrected(rtf[i], h_hat[i]) for i in range(h_hat.shape[0])])
    print(f"Mean Squared Errors: {np.abs(mses)}")
    with open(f"{args.output}/rtf_accuracy_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([f"covariance_whitening", np.mean(np.abs(mses)), npm])
    # Save the estimated RTF
    np.save(f"{args.output}/estimated_rtf_covariance_whitening.npy", h_hat)
    print("Estimated RTF saved to", f"{args.output}/estimated_rtf_covariance_whitening.npy")
    return h_hat
    
def rtf_accuracy_covariance_subtraction_identity(rtf, mics, args):
    
    raise NotImplementedError("Covariance subtraction is not implemented yet.")

    noisy_cpsd = Utils.compute_cpsd_matrices(Utils.compute_multichannel_stft(mics))
    # Identity matrix of same shape, with same frequency bins and time frames (all ident)
    noise_cpsd = np.zeros_like(noisy_cpsd)
    for i in range(noisy_cpsd.shape[2]):
        for j in range(noisy_cpsd.shape[3]):
            noise_cpsd[:,:, i, j] = np.identity(noisy_cpsd.shape[0])
    # Covariance whitening step
    h_hat = Covariance.estimate_rtf_covariance_subtraction(noise_cpsd, noisy_cpsd)
    # Interpolate to the same shape as rtf
    h_hat = Utils.interpolate_stft_to_fft(h_hat, rtf.shape[1])
    print(f"Estimated RTF shape: {h_hat.shape}")
    print(f"RTF shape: {rtf.shape}")
    mses = np.array([Eval.meansquared_error_delay_corrected(rtf[i], h_hat[i]) for i in range(h_hat.shape[0])])
    print(f"Mean Squared Errors: {np.abs(mses)}")
    npm = Eval.npm(rtf, h_hat)
    print(f"Normalized Projection Misalignment: {npm}")
    # Save the estimated RTF
    np.save(f"{args.output}/estimated_rtf_cw.npy", h_hat)
    print("Estimated RTF saved to", f"{args.output}/estimated_rtf_covariance_whitening.npy")
    return h_hat
    
def rtf_accuracy_covariance_whitening_noisy(rtf, mics, noise_len, args):
    # Noise sample is from 0 up to noise_len (s)
    noise_idx = math.floor(noise_len* config_handler.get_config().audio.sample_rate)
    noise_sample = mics[:,:noise_idx]
    noise_cpsd = Utils.compute_cpsd_matrices(Utils.compute_multichannel_stft(noise_sample))
    noisy_cpsd = Utils.compute_cpsd_matrices(Utils.compute_multichannel_stft(mics[:,noise_idx:]))
    
    # Add Isotropic noise (ident) to noise and noisy cpsd
    for i in range(noise_cpsd.shape[2]):
        for j in range(noise_cpsd.shape[3]):
            noise_cpsd[:,:, i, j] = noise_cpsd[:,:,i,j] + np.identity(noise_cpsd.shape[0])
    for i in range(noisy_cpsd.shape[2]):
        for j in range(noisy_cpsd.shape[3]):
            noisy_cpsd[:,:, i, j] = noisy_cpsd[:,:,i,j] + np.identity(noisy_cpsd.shape[0])
    
    h_hat = Covariance.estimate_rtf_covariance_whitening(noise_cpsd, noisy_cpsd)
    # Interpolate to the same shape as rtf
    h_hat = Utils.interpolate_stft_to_fft(h_hat, rtf.shape[1])
    
    npm = Eval.npm(rtf, h_hat)
    print(f"Normalized Projection Misalignment: {npm}")
    mses = np.array([Eval.meansquared_error_delay_corrected(rtf[i], h_hat[i]) for i in range(h_hat.shape[0])])
    print(f"Mean Squared Errors: {np.abs(mses)}")
    with open(f"{args.output}/rtf_accuracy_results.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([f"covariance_whitening", np.mean(np.abs(mses)), npm])
    # Save the estimated RTF
    np.save(f"{args.output}/estimated_rtf_covariance_whitening_noisy.npy", h_hat)
    print("Estimated RTF saved to", f"{args.output}/estimated_rtf_covariance_whitening_noisy.npy")
    return h_hat
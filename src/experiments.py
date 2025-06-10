from Estimators import RTF_Estimator, Covariance
import Evaluate as Eval
import numpy as np
from Utils import Utils


def rtf_accuracy_filter(adaptive_filter, rtf, mics, args):
    rtf_estimator = RTF_Estimator.RTFEstimator(adaptive_filter)
    h_hat = rtf_estimator.estimate_rtf(mics, reference_idx=0)
    npm = Eval.npm(rtf, h_hat)
    print(f"Normalized Projection Misalignment: {npm}")
    mses = np.array([Eval.meansquared_error_delay_corrected(rtf[i], h_hat[i]) for i in range(h_hat.shape[0])])
    print(f"Mean Squared Errors: {np.abs(mses)}")
    # Save the estimated RTF
    np.save(f"{args.output}/estimated_rtf_{args.filter[0]}.npy", h_hat)
    print("Estimated RTF saved to", f"{args.output}/estimated_rtf_filter_{args.filter[0]}.npy")

def rtf_accuracy_covariance_whitening_identity(rtf, mics, args):
    noisy_cpsd = Utils.compute_cpsd_matrices(Utils.compute_multichannel_stft(mics))
    print(noisy_cpsd.shape)
    # Identity matrix of same shape, with same frequency bins and time frames (all ident)
    noise_cpsd = np.zeros_like(noisy_cpsd)
    for i in range(noisy_cpsd.shape[2]):
        for j in range(noisy_cpsd.shape[3]):
            noise_cpsd[:,:, i, j] = np.identity(noisy_cpsd.shape[0])
    # Covariance whitening step
    h_hat = Covariance.estimate_rtf_covariance_whitening(noise_cpsd, noisy_cpsd)
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
    
def rtf_accuracy_covariance_subtraction_identity(rtf, mics, args):
    noisy_cpsd = Utils.compute_cpsd_matrices(Utils.compute_multichannel_stft(mics))
    print(noisy_cpsd.shape)
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
import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import config_handler
import Visualisations.vis as vis
import Evaluate as Eval
import scipy as sp
from Utils import Utils
from Estimators import RTF_Estimator, AdaptiveFilters, Covariance

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='RTF Estimation simulation frontend')
    parser.add_argument('room_file', type=str, help='Room specification file')
    parser.add_argument('--config_file', type=str, help='Configuration file to be used', default='config.yaml')
    parser.add_argument('--audio_out', type=str, help='Output audio file to be saved')
    parser.add_argument('-p','--print_format', type=str, help='Output format for measurements')
    parser.add_argument("-v","--visualise", type=str, help="Chart type to visualise the data", default="basic")
    parser.add_argument("-e", "--experiment", type=str, help="Experiment to run")
    parser.add_argument("-o", "--output", type=str, help="Output directory for the experiment")
    parser.add_argument("-f", "--filter", nargs="+", help="Filter to use for RTF estimation", default=["IPNLMS", "1024", "0.1"])
    args = parser.parse_args()
    # Load config
    config_handler.load_config(args.config_file)
    cfg = config_handler.get_config()
    import RIRGen.RIRGen as RIRGen
    
    # Load room specifications
    room_spec = yaml.safe_load(open(args.room_file))
    rirgen = RIRGen.RIRGenerator.from_room_spec(room_spec)
    
    # Compute the relative transfer function (mic 0 is default reference)
    h_mics = rirgen.get_acoustic_transfer_functions()
    rtf = Utils.compute_rtf(h_mics)
    rrir = [np.fft.irfft(rtf_n) for rtf_n in rtf]
    
    mic_0_audio = rirgen.room.room.mic_array.signals[0, :]
    mic_1_audio = rirgen.room.room.mic_array.signals[1, :]
    
    mic_1_recovered = sp.signal.convolve(mic_0_audio, rrir[1])
    mse = Eval.meansquared_error_delay_corrected(mic_1_recovered, mic_1_audio)
    print(f"Mean Squared Error: {mse}")
    
    # Use an adaptive filter to estimate the RTF
    # import Estimators.AdaptiveFilters as AdaptiveFilters
    # Adaptivefilter = AdaptiveFilters.IPNLMS(1024, 0.1)
    # Adaptivefilter.full_simulate(mic_0_audio, mic_1_audio)  
    # mic_1_estimated, filter_error = Eval.filter_step_error(mic_0_audio, mic_1_audio, Adaptivefilter)
    # mic_1_recovered = sp.signal.convolve(mic_0_audio,rrir[1])
    # mse = Eval.meansquared_error_delay_corrected(mic_1_estimated, mic_1_audio)
    # print(f"Mean Squared Error: {mse}")
    # plt.plot(rrir[1])
    # plt.plot(Adaptivefilter.w, label='Adaptive Filter IR')
    # plt.show()
    
    # Do full RTF sim
    if args.filter is not None:
        if args.filter[0] == "IPNLMS":
            logger.info("Using IPNLMS filter for RTF estimation")
            adaptive_filter = AdaptiveFilters.IPNLMS(int(args.filter[1]), float(args.filter[2]))
        elif args.filter[0] == "PNLMS":
            logger.info("Using PNLMS filter for RTF estimation")
            adaptive_filter = AdaptiveFilters.PNLMS(int(args.filter[1]), float(args.filter[2]))
        elif args.filter[0] == "NLMS":
            logger.info("Using NLMS filter for RTF estimation")
            adaptive_filter = AdaptiveFilters.NLMS(int(args.filter[1]), float(args.filter[2]))  
        elif args.filter[0] == "LMS":
            logger.info("Using LMS filter for RTF estimation")
            adaptive_filter = AdaptiveFilters.LMS(int(args.filter[1]), float(args.filter[2]))
        else:
            logger.error(f"Filter {args.filter} is not implemented.")
            raise NotImplementedError(f"Filter {args.filter} is not implemented.")
    
    

    
    
    if args.experiment is not None:
        if args.output is None:
            logger.error('Output directory must be specified for experiments.')
            raise ValueError('Output directory must be specified for experiments.')
        if args.experiment == "all":
            raise NotImplementedError("Experiment 'all' is not implemented yet.")
        elif args.experiment == "rtf_accuracy_filter":
            print(f"Running RTF accuracy experiment with adaptive filter {args.filter[0]}")
            # Copy room config to output directory
            Utils.copy_room_spec_to_output(args.room_file, args.output)
            rtf_estimator = RTF_Estimator.RTFEstimator(adaptive_filter)
            h_hat = rtf_estimator.estimate_rtf(rirgen.room.room.mic_array.signals, reference_idx=0)
            mic_1_recovered = adaptive_filter.apply_filter(mic_0_audio)
            npm = Eval.npm(rtf, h_hat)
            print(f"Normalized Projection Misalignment: {npm}")
            mses = np.array([Eval.meansquared_error_delay_corrected(rrir[i], h_hat[i]) for i in range(h_hat.shape[0])])
            print(f"Mean Squared Errors: {np.abs(mses)}")
            # Save the estimated RTF
            np.save(f"{args.output}/estimated_rtf.npy", h_hat)
            print("Estimated RTF saved to", f"{args.output}/estimated_rtf_filter_{args.filter[0]}.npy")
            
        elif args.experiment == "rtf_accuracy_covariance_whitening_identity":
            print("Running RTF accuracy experiment with covariance whitening")
            # Copy room config to output directory
            Utils.copy_room_spec_to_output(args.room_file, args.output)
            noisy_cpsd = Utils.compute_cpsd_matrices(Utils.compute_multichannel_stft(rirgen.room.room.mic_array.signals))
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
            mic_1_recovered = sp.signal.convolve(rirgen.room.room.mic_array.signals[0, :], np.fft.irfft(h_hat[1]))
            print(f"Estimated RTF shape: {h_hat.shape}")
            print(f"RTF shape: {rtf.shape}")
            mses = np.array([Eval.meansquared_error_delay_corrected(rrir[i], h_hat[i]) for i in range(h_hat.shape[0])])
            print(f"Mean Squared Errors: {np.abs(mses)}")
            npm = Eval.npm(rtf, h_hat)
            print(f"Normalized Projection Misalignment: {npm}")
            # Save the estimated RTF
            np.save(f"{args.output}/estimated_rtf.npy", h_hat)
            print("Estimated RTF saved to", f"{args.output}/estimated_rtf_covariance_whitening.npy")
            
        else:
            logger.error(f"Experiment {args.experiment} does not exist.")
            raise NotImplementedError(f"Experiment {args.experiment} does not exist.")
            

    

    if args.audio_out is not None:
        rirgen.save_audio(args.audio_out)

    if args.print_format is not None:
        # Set up matplotlib when using PGF.
        if args.print_format == 'pgf':
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })
        #vis.defaultPlot(rirgen, rrir[1], mic_1_recovered)
        if args.visualise == 'basic':
            vis.defaultRIRPlot(rirgen, rrir[1], mic_1_recovered)
        elif args.visualise == 'all':
            vis.plotAllRir(rrir)
        elif args.visualise == 'learning_curve':
            print("Simulating learning curve")
            vis.filter_performance(Eval.filter_learning_curve_mse(mic_0_audio, mic_1_audio, adaptive_filter, rrir[1][0:len(adaptive_filter.w)]),cfg.audio.sample_rate)
        else:
            logger.error('Invalid visualisation type. Please use either "basic" or "all"')
            raise ValueError('Invalid visualisation type. Please use either "basic" or "all"')


        if args.print_format == 'show':
            plt.show()
        elif args.print_format == 'png':
            plt.savefig('output.png')
        elif args.print_format == 'pgf':
            plt.savefig('output.pgf')
        else:
            logger.error('Invalid print format. Please use either "show", "png" or "pgf"')
            raise ValueError('Invalid print format. Please use either "show", "png" or "pgf"')
        
    if args.visualise == 'pygame':
        vis.draw_room_from_spec(room_spec)

if __name__ == '__main__':
    main()
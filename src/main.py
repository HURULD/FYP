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
import experiments
import csv

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='RTF Estimation simulation frontend')
    parser.add_argument('room_file', type=str, help='Room specification file')
    parser.add_argument('--config_file', type=str, help='Configuration file to be used', default='config.yaml')
    parser.add_argument('--audio_out', type=str, help='Output audio file to be saved')
    parser.add_argument('-p','--print_format', type=str, help='Output format for measurements')
    parser.add_argument("-v","--visualise", type=str, help="Chart type to visualise the data")
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
    #mse = Eval.meansquared_error_delay_corrected(mic_1_recovered, mic_1_audio)
    #print(f"Mean Squared Error: {mse}")
    
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
        
        Utils.save_room_spec_to_output(room_spec, args.output)
        vis.draw_room_from_spec_matplotlib(room_spec)
        plt.savefig(f"{args.output}/room_spec.png")
        np.savez(f"{args.output}/mic_array_signals.npz", signals=rirgen.room.room.mic_array.signals)
        np.savez(f"{args.output}/rtf.npz", rtf=rtf, rrir=rrir)
        Utils.save_config_to_output(config_handler.get_config(), f"{args.output}/config.yaml")
        
        # Each result is written from the experiment function call
        with open(f"{args.output}/rtf_accuracy_results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["estimator", "mic_index", "mse"])
        
        if args.experiment == "all":
            print(f"Running RTF accuracy experiment with adaptive filter {args.filter[0]}")
            rtf_est = experiments.rtf_accuracy_filter(adaptive_filter, rtf, rirgen.room.room.mic_array.signals, args)
            
            print("Running RTF accuracy experiment with covariance whitening")
            rtf_est = experiments.rtf_accuracy_covariance_whitening_identity(rtf, rirgen.room.room.mic_array.signals, args)
            
        elif args.experiment == "rtf_accuracy_filter":
            print(f"Running RTF accuracy experiment with adaptive filter {args.filter[0]}")
            rtf_est = experiments.rtf_accuracy_filter(adaptive_filter, rtf, rirgen.room.room.mic_array.signals, args)

        elif args.experiment == "rtf_accuracy_covariance_whitening_identity":
            print("Running RTF accuracy experiment with covariance whitening")
            rtf_est = experiments.rtf_accuracy_covariance_whitening_identity(rtf, rirgen.room.room.mic_array.signals, args)
            
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
                "text.usetex": True,
                "font.family": "serif", 
                "axes.labelsize": 10,
                "legend.fontsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            })
        if args.visualise is not None:
            #vis.defaultPlot(rirgen, rrir[1], mic_1_recovered)
            if args.visualise == 'basic':
                vis.defaultRIRPlot(rirgen, rrir[1], mic_1_recovered)
            elif args.visualise == 'all':
                vis.plotAllRir(rrir)
            elif args.visualise == 'learning_curve':
                print("Simulating learning curve")
                vis.filter_performance(Eval.filter_learning_curve_mse(mic_0_audio, mic_1_audio, adaptive_filter, rrir[1][0:len(adaptive_filter.w)]),cfg.audio.sample_rate)
            
            elif args.visualise == "room_render":
                print("Rendering room from specification")
                vis.draw_room_from_spec_matplotlib(room_spec)
                
            elif args.visualise == "rtf_learning_curve":
                vis.filter_performance(Eval.rtf_learning_curve_mse(rirgen.room.room.mic_array.signals, adaptive_filter, rtf), cfg.audio.sample_rate, adaptive_filter.__class__.__name__)

            elif args.visualise == "rrir":
                print("Rendering RRIR")
                vis.plot_rrir(rrir, rtf_est, cfg.audio.sample_rate)
                
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
        vis.draw_room_from_spec_pygame(room_spec)

if __name__ == '__main__':
    main()
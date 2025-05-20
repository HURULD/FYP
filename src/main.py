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

logger = logging.getLogger(__name__)  

def main():
    parser = argparse.ArgumentParser(description='RTF Estimation simulation frontend')
    parser.add_argument('room_file', type=str, help='Room specification file')
    parser.add_argument('--config_file', type=str, help='Configuration file to be used', default='config.yaml')
    parser.add_argument('--audio_out', type=str, help='Output audio file to be saved')
    parser.add_argument('-p','--print_format', type=str, help='Output format for measurements')
    parser.add_argument("-v","--visualise", type=str, help="Chart type to visualise the data", default="basic")
    args = parser.parse_args()
    # Load config
    config_handler.load_config(args.config_file)
    import RIRGen.RIRGen as RIRGen
    
    # Load room specifications
    room_spec = yaml.safe_load(open(args.room_file))
    rirgen = RIRGen.RIRGenerator.from_room_spec(room_spec)
    
    # Compute the relative transfer function from mic 0 to mic 1
    
    h_mics = rirgen.get_acoustic_transfer_functions()
    mic_0_audio = rirgen.room.room.mic_array.signals[0, :]
    mic_1_audio = rirgen.room.room.mic_array.signals[1, :]
    rtf = Utils.compute_rtf(h_mics)
    rrir = [np.fft.irfft(rtf_n) for rtf_n in rtf]
    
    # Use an adaptive filter to estimate the RTF
    import Estimators.AdaptiveFilters as AdaptiveFilters
    Adaptivefilter = AdaptiveFilters.PNLMS(1024, 1)
    Adaptivefilter.full_simulate(mic_0_audio, mic_1_audio)
    mic_1_estimated, filter_error = Eval.filter_step_error(mic_0_audio, mic_1_audio, Adaptivefilter)
    mic_1_recovered = sp.signal.convolve(mic_0_audio,rrir[1])
    mse = Eval.meansquared_error_delay_corrected(mic_1_estimated, mic_1_audio)
    print(f"Mean Squared Error: {mse}")
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
        elif args.visualise == 'filter':
            vis.filter_performance(filter_error)
        elif args.visualise == 'learning_curve':
            print("Simulating learning curve")
            vis.filter_performance(Eval.filter_learning_curve(mic_0_audio, mic_1_audio, Adaptivefilter, rrir[1][0:len(Adaptivefilter.w)]))
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
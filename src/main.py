import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import config_handler
import Visualisations.vis as vis
import RIRGen.Evaluate as Eval
import scipy as sp

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
    rs = yaml.safe_load(open(args.room_file))
    rirgen = RIRGen.RIRGenerator.from_room_spec(rs)
    
    # Compute the relative transfer function from mic 0 to mic 1
    
    h_mics = rirgen.get_acoustic_transfer_functions()
    mic_0_audio = rirgen.room.room.mic_array.signals[0, :]
    mic_1_audio = rirgen.room.room.mic_array.signals[1, :]
    rtf = Eval.compute_rtf(h_mics)
    rrir = [np.fft.irfft(rtf_n) for rtf_n in rtf]
    
    
    mic_1_recovered = sp.signal.fftconvolve(mic_0_audio,rrir[1])
    mse = Eval.meansquared_error(mic_1_recovered, mic_1_audio)
    print(f"Mean Squared Error: {mse}")
    print(f"min MSE: {min(mse)}")
    print(f"Delay needed: {np.where(mse == min(mse))[0][0]}")
    
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
            vis.defaultPlot(rirgen, rrir[1], mic_1_recovered)
        elif args.visualise == 'all':
            rrir_shifted = np.fft.fftshift(rrir)
            vis.plotAllRir(rrir_shifted)
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

if __name__ == '__main__':
    main()
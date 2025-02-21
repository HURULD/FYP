import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import config_handler
import helpers
import Visualisations.vis as vis

logger = logging.getLogger(__name__)  

def main():
    parser = argparse.ArgumentParser(description='RTF Estimation simulation frontend')
    parser.add_argument('room_file', type=str, help='Room specification file')
    parser.add_argument('--config_file', type=str, help='Configuration file to be used', default='config.yaml')
    parser.add_argument('--audio_out', type=str, help='Output audio file to be saved')
    parser.add_argument('--print_format', type=str, help='Output format for measurements')
    args = parser.parse_args()
    # Load config
    config_handler.load_config(args.config_file)
    import RIRGen.RIRGen as RIRGen
    
    # Load room specifications
    rs = yaml.safe_load(open(args.room_file))
    rirgen = RIRGen.RIRGenerator.from_room_spec(rs)
    rirgen.simulate()
    
    # Compute the relative transfer function from mic 0 to mic 1
    h_mic0 = rirgen.room.room.rir[0][0]
    h_mic1 = rirgen.room.room.rir[1][0]
    mic_0_audio = rirgen.room.room.mic_array.signals[0, :]
    mic_1_audio = rirgen.room.room.mic_array.signals[1, :]
    rtf_01 = helpers.compute_rtf(h_mic0, h_mic1)
    rrir_01 = np.fft.irfft(rtf_01)
    
    mic_1_recovered = np.convolve(mic_0_audio, rrir_01)
    mse = helpers.meansquared_error(mic_1_recovered, mic_1_audio)
    print(f"Mean Squared Error: {mse}")
    
    
    if args.audio_out is not None:
        rirgen.save_audio(args.audio_out)
    
    if args.print_format is not None:
        if args.print_format == 'pgf':
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })
        vis.defaultPlot(rirgen, rrir_01, mic_1_recovered)
        
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
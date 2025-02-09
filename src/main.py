import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
import config_handler
import helpers

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
        
        # Create a plot
        plt.figure()
        
        # plot one of the RIR. both can also be plotted using room.plot_rir()
        rir_1_0 = rirgen.room.room.rir[1][0]
        plt.subplot(4, 1, 1)
        plt.plot(np.arange(len(rir_1_0)) / rirgen.room.room.fs, rir_1_0)
        plt.title("The RIR from source 0 to mic 1")
        plt.xlabel("Time [s]")

        # plot signal at microphone 1
        plt.subplot(4, 1, 2)
        plt.plot(np.arange(len(rirgen.room.room.mic_array.signals[1, :])) / rirgen.room.room.fs, rirgen.room.room.mic_array.signals[1, :])
        plt.title("Microphone 1 signal")
        plt.xlabel("Time [s]")

        plt.subplot(4, 1, 3)
        plt.plot(np.arange(len(rrir_01)) / rirgen.room.room.fs, rrir_01)
        plt.title("The RRIR from mic 0 to mic 1")
        plt.xlabel("Time [s]")
        
        plt.subplot(4, 1, 4)
        plt.plot(np.arange(len(mic_1_recovered)) / rirgen.room.room.fs, mic_1_recovered)
        plt.title("Recovered mic 1 signal")
        plt.xlabel("Time [s]")

        plt.tight_layout()    # # Create a plot

        if args.print_format == 'show':
            plt.show()
        elif args.print_format == 'png':
            plt.savefig('output.png')
        elif args.print_format == 'pgf':
            plt.savefig('output.pgf')
        else:
            logger.error('Invalid print format. Please use either "show", "png" or "pgf"')
            raise ValueError('Invalid print format. Please use either "show", "png" or "pgf"')

    # _, audio = wavfile.read(args.input_file)
    # # The desired reverberation time and dimensions of the room
    # rt60_tgt = 0.5  # seconds
    # room_dim = [9, 7.5, 3.5]  # meters

    # # We invert Sabine's formula to obtain the parameters for the ISM simulator
    # e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # # Create the room
    # room = pra.ShoeBox(
    #                     room_dim,
    #                     fs=44100, 
    #                     materials=pra.Material(e_absorption), 
    #                     max_order=max_order,
    #                     use_rand_ism=True,
    #                     max_rand_disp=0.5
    #                 )
    # room.add_source([2.5,3.73,1.76], signal=audio, delay=1.3)
    # mics = np.c_[
    #     [6.3,4.87,1.2],
    #     [6.3,4.93,1.2],
    # ]
    # room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
    # # Run the simulation (this will also build the RIR automatically)
    # room.simulate()

    # room.mic_array.to_wav(
    #     f"output.wav",
    #     norm=True,
    #     bitdepth=np.int16,
    # )

    # # measure the reverberation time
    # rt60 = room.measure_rt60()
    # print("The desired RT60 was {}".format(rt60_tgt))
    # print("The measured RT60 is {}".format(rt60[1, 0]))

    # # Create a plot
    # plt.figure()

    # # plot one of the RIR. both can also be plotted using room.plot_rir()
    # rir_1_0 = room.rir[1][0]
    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
    # plt.title("The RIR from source 0 to mic 1")
    # plt.xlabel("Time [s]")

    # # plot signal at microphone 1
    # plt.subplot(2, 1, 2)
    # plt.plot(room.mic_array.signals[1, :])
    # plt.title("Microphone 1 signal")
    # plt.xlabel("Time [s]")

    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()
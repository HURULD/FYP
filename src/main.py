import argparse
import logging
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
from scipy.io import wavfile


def main():
    parser = argparse.ArgumentParser(description='RTF Estimation simulation frontend')
    parser.add_argument('input_file', type=str, help='Input file to be processed')
    #parser.add_argument('output_file', type=str, help='Output file to be saved')
    args = parser.parse_args()
    _, audio = wavfile.read(args.input_file)
    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.5  # seconds
    room_dim = [9, 7.5, 3.5]  # meters

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    room = pra.ShoeBox(
                        room_dim,
                        fs=44100, 
                        materials=pra.Material(e_absorption), 
                        max_order=max_order,
                        use_rand_ism=True,
                        max_rand_disp=0.5
                    )
    room.add_source([2.5,3.73,1.76], signal=audio, delay=1.3)
    mics = np.c_[
        [6.3,4.87,1.2],
        [6.3,4.93,1.2],
    ]
    room.add_microphone_array(pra.MicrophoneArray(mics, room.fs))
    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    room.mic_array.to_wav(
        f"output.wav",
        norm=True,
        bitdepth=np.int16,
    )

    # measure the reverberation time
    rt60 = room.measure_rt60()
    print("The desired RT60 was {}".format(rt60_tgt))
    print("The measured RT60 is {}".format(rt60[1, 0]))

    # Create a plot
    plt.figure()

    # plot one of the RIR. both can also be plotted using room.plot_rir()
    rir_1_0 = room.rir[1][0]
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
    plt.title("The RIR from source 0 to mic 1")
    plt.xlabel("Time [s]")

    # plot signal at microphone 1
    plt.subplot(2, 1, 2)
    plt.plot(room.mic_array.signals[1, :])
    plt.title("Microphone 1 signal")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import numpy as np

def BeamPatternPolar(arrayShape:np.array, arrayWeights:np.array, thetaRange = np.linspace(0, 2*np.pi, 1000)):
    raise NotImplementedError
    # magnitudes = []
    # weightedArray = arrayWeights.conj().T @ arrayShape
    # for theta in thetaRange:
    
def defaultPlot(rirgen,rrir_01,mic_1_recovered):
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
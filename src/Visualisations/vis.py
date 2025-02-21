import matplotlib.pyplot as plt
import numpy as np
import config_handler as conf

def BeamPatternPolar(arrayShape:np.array, arrayWeights:np.array, thetaRange = np.linspace(0, 2*np.pi, 1000)):
    raise NotImplementedError
    # magnitudes = []
    # weightedArray = arrayWeights.conj().T @ arrayShape
    # for theta in thetaRange:
    
def defaultPlot(rirgen,rrir,mic_recovered):
    # TODO: Make this more general
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
        plt.plot(np.arange(len(rrir)) / rirgen.room.room.fs, rrir)
        plt.title("The RRIR from mic 0 to mic 1")
        plt.xlabel("Time [s]")
        
        plt.subplot(4, 1, 4)
        plt.plot(np.arange(len(mic_recovered)) / rirgen.room.room.fs, mic_recovered)
        plt.title("Recovered mic 1 signal")
        plt.xlabel("Time [s]")

        plt.tight_layout()    # # Create a plot
        
def plotAllRir(rrir):
    plt.figure()
    for i, rrir_i, in enumerate(rrir):
        
        #RRIR time domain
        plt.subplot(len(rrir), 3, (3*i)+1)
        plt.plot((np.arange(len(rrir_i))-(0.5*len(rrir_i))) / conf.get_config().audio.sample_rate, rrir_i)
        plt.title("The RRIR from mic 0 to mic " + str(i))
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        
        # RTF Magnitude
        plt.subplot(len(rrir), 3, (3*i)+2)
        rtf_i = np.fft.fft(rrir_i)
        plt.plot(np.arange(len(rtf_i))-(0.5*len(rrir_i)), np.abs(rtf_i))
        plt.title("The RTF from mic 0 to mic " + str(i))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        
        # RTF Phase
        plt.subplot(len(rrir), 3, (3*i)+3)
        plt.plot(np.arange(len(rtf_i))-(0.5*len(rrir_i)), np.angle(rtf_i))
        plt.title("The RTF from mic 0 to mic " + str(i))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Phase [rad]")
        
    plt.tight_layout()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import config_handler as conf
from scipy.fft import fft, fftfreq, fftshift
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from typing import Optional, Literal

def BeamPatternPolar(arrayShape:np.array, arrayWeights:np.array, thetaRange = np.linspace(0, 2*np.pi, 1000)):
    raise NotImplementedError
    # magnitudes = []
    # weightedArray = arrayWeights.conj().T @ arrayShape
    # for theta in thetaRange:

def customPlot(data:list[list], range: Optional[list[float]]):
    raise NotImplementedError

def defaultRIRPlot(rirgen,rrir,mic_recovered):
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

        plt.tight_layout()    # Create a plot

def plotAllRir(rrir):
    plt.figure()
    for i, rrir_i, in enumerate(rrir):
        
        #RRIR time domain
        plt.subplot(len(rrir), 3, (3*i)+1)
        plt.plot((np.arange(len(rrir_i))-(0.5*len(rrir_i))) / conf.get_config().audio.sample_rate, rrir_i)
        plt.title("The RRIR from mic 0 to mic " + str(i))
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        
        N = len(rrir)
        rtf_i = fftshift(fft(rrir_i))
        xf = fftfreq(len(rrir_i), conf.get_config().audio.sample_rate)
        xf = fftshift(xf)
        
        # RTF Magnitude
        plt.subplot(len(rrir), 3, (3*i)+2)
        plt.plot(xf, 1.0/N * np.abs(rtf_i))
        plt.title("The RTF from mic 0 to mic " + str(i))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        
        # RTF Phase
        plt.subplot(len(rrir), 3, (3*i)+3)
        plt.plot(xf, 1.0/N * np.unwrap(np.angle(rtf_i)))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase')
        
    plt.tight_layout()
    
def draw_room_from_spec_pygame(room_spec:dict):
    dimensions = room_spec['room']['dimensions']
    sources = room_spec.get('sources', [])
    microphones = room_spec.get('microphones', [])
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Room Visualisation")
    scale = 40 # 1m = 40px
    offsetx, offsety = 100, 100
    def render():
        # Draw the room from the dimensions
        screen.fill((0,0,0))
        pygame.draw.rect(screen, (255, 255, 255), (offsetx,offsety, int(dimensions[0]*scale), int(dimensions[1]*scale)),5,4)
        for source in sources:
            pygame.draw.circle(screen, (255, 0, 0), (offsetx+int(source['position'][0]*scale), offsety+int(source['position'][1]*scale)), 5)
        for mic in microphones:
            pygame.draw.circle(screen, (0,255,0), (offsetx+int(mic[0]*scale), offsety+int(mic[1]*scale)), 5)
        pygame.draw.line(screen, (255,255,255), (10,10), (10+scale, 10), 5)
        pygame.display.flip()
    running = True
    drag = False
    while running:
        for event in pygame.event.get():
            # change scale with scroll wheel
            if event.type == pygame.MOUSEWHEEL:
                scale += event.y
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    drag = True
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drag = False
            if event.type == pygame.MOUSEMOTION:
                if drag:
                    offsetx += event.rel[0]
                    offsety += event.rel[1]
            if event.type == pygame.QUIT:
                running = False
            render()
    pygame.quit()
    
def draw_room_from_spec_matplotlib(room_spec:dict):
    fig, ax = plt.subplots()
    dimensions = room_spec['room']['dimensions']
    sources = room_spec.get('sources', [])
    microphones = room_spec.get('microphones', [])
    # Draw the room from the dimensions
    ax.add_patch(patches.Rectangle((0, 0), dimensions[0], dimensions[1], linewidth=4, edgecolor='black', facecolor='none'))
    # Draw sources as red circles
    for source in sources:
        if 'noise' in source:
            ax.add_patch(patches.Circle((source['position'][0], source['position'][1]), 0.05, color='red'))
        elif 'file' in source:
            ax.add_patch(patches.Circle((source['position'][0], source['position'][1]), 0.05, color='blue'))
    # Draw microphones as green circles
    for mic in microphones:
        ax.add_patch(patches.Circle((mic[0], mic[1]), 0.05, color='green'))
    ax.set_xlim(-0.5, dimensions[0] + 0.5)
    ax.set_ylim(-0.5, dimensions[1] + 0.5)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Room Visualisation")
    plt.grid()
    

def filter_performance(filter_error, sample_rate:Optional[int]=None, filter_name:Optional[str]=None):
    print(filter_error[~np.isnan(filter_error)])
    plt.figure()
    if sample_rate is not None:
        plt.plot(np.arange(len(filter_error)) / sample_rate, filter_error)
        plt.xlabel("Time (s)")
    else:
        plt.plot(filter_error)
        plt.xlabel("Sample Index")
    if filter_name is not None:
        plt.title(f"Filter Performance: {filter_name}")
    else:
        plt.title("Filter Performance")
    plt.ylabel("MSE")
    
def fft_default_plot(signal,sample_rate, scale:Literal['log','linear']='log'):
    
    N = len(signal) # N samples
    T = 1 / sample_rate # Sample period
    
    yf = fft(signal)
    xf = fftfreq(N, T)
    yplot = fftshift(yf)
    
    magnitude = np.abs(yplot[:N//2]) * 2/N
    magnitude_db = 20 * np.log10(magnitude + 1e-12) # Avoid log(0)
    
    plt.figure()
    plt.subplot(1,2,1)
    if scale == 'linear':
        plt.plot(xf[:N//2], magnitude)
        plt.ylabel('Magnitude')
    elif scale == 'log':
        plt.plot(xf[:N//2], magnitude_db)
        plt.ylabel('Magnitude (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.grid()
    
    plt.subplot(1,2,2)
    xf = fftshift(xf)
    plt.plot(xf, 1.0/N * np.unwrap(np.angle(yplot)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase')
    plt.grid()
    
    plt.tight_layout() 
    plt.show()
    
def plot_rrir(rrir, rtf_est, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Plot true RRIR for mic 0 to mic 1 (assumes ref mic is 0)
    ax.plot(np.arange(len(rrir[0])) / sample_rate, rrir[1], label='True RRIR')
    
    # Plot estimated RRIR for mic 0 to mic 1
    rrir_est = np.fft.irfft(rtf_est[1])
    ax.plot(np.arange(len(rrir_est)) / sample_rate, rrir_est, label='Estimated RRIR', color='orange')
    ax.set_title('RRIR Comparison')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    plt.grid()
    ax.legend()
    plt.tight_layout()

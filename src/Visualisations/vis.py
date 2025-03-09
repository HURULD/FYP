import matplotlib.pyplot as plt
import numpy as np
import config_handler as conf
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

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
    
def draw_room_from_spec(room_spec:dict):
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

def filter_performance(filter_error):
    print(filter_error[~np.isnan(filter_error)])
    plt.figure()
    plt.plot(filter_error)
    plt.title("Filter Performance")
    plt.xlabel("Sample")
    plt.ylabel("MSE")
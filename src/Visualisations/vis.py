import matplotlib.pyplot as plt
import numpy as np

def BeamPatternPolar(arrayShape:np.array, arrayWeights:np.array, thetaRange = np.linspace(0, 2*np.pi, 1000)):
    magnitudes = []
    weightedArray = arrayWeights.conj().T @ arrayShape
    for theta in thetaRange:
        
        results.append()
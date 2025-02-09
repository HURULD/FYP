from dataclasses import dataclass
import numpy as np
import logging
import cmath

logger = logging.getLogger(__name__)

@dataclass
class Position2D:
    x: float
    y: float

@dataclass
class Position3D:
    x: float
    y: float
    z: float

@dataclass
class ArrayElement2D:
    position: Position2D
    weight: complex
    
    def __str__(self):
        return f"Position: {self.position}, Gain: {abs(self.weight)}, Phase: {cmath.phase(self.weight)}"
    
@dataclass
class ArrayElement3D:
    position: Position3D
    weight: complex
    
    def __str__(self):
        return f"Position: {self.position}, Gain: {abs(self.weight)}, Phase: {cmath.phase(self.weight)}"
    
class SensorArray():
    def __init__(self, array_elements: list[ArrayElement2D] = []):
        self.array_elements = array_elements
        
    def __str__(self):
        return '\n'.join([str(element) for element in self.array_elements])
    
    def toPositionArray(self):
        return np.array([element.position for element in self.array_elements])
    
    def assignWeights(self, array_weights:np.array):
        if len(array_weights) != len(self.array_elements):
            raise ValueError("Array weights must be of the same length as the array elements")
        for i, element in enumerate(self.array_elements):
            element.weight = array_weights[i]
    
    @property
    def array_elements(self):
        return self._array_elements
    @array_elements.setter
    def array_elements(self, array_elements: list[ArrayElement2D]):
        if len(array_elements) == 0:
            self._array_elements = []
            logger.warning("Array elements is empty, may cause simulation problems")
            return
        assumed_type = type(array_elements[0])
        for element in array_elements:
            if not isinstance(element, ArrayElement2D):
                raise TypeError(f"Array elements must be of type ArrayElement2D, not {type(element)}")
            if type(element) != assumed_type:
                raise TypeError(f"Array elements must all be of the same type")
        self._array_elements = array_elements
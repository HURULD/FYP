import numpy as np
import pyroomacoustics as pra
import yaml
import config_handler
from scipy.io import wavfile
config = config_handler.get_config()

class Room:
    def __init__(self, dimensions:list[float], rt60_tgt:float, absorption:float|None=None):
        self.dimensions = dimensions
        self.rt60_tgt = rt60_tgt
        if absorption is None:
            e_absorption, max_order = pra.inverse_sabine(rt60_tgt, dimensions)
        else:
            e_absorption = float(absorption)
        self.room = pra.ShoeBox(
            dimensions,
            fs=config.audio.sample_rate, 
            materials=pra.Material(e_absorption), 
            use_rand_ism=False,
            max_rand_disp=0.5
        )
    
    @classmethod
    def from_room_spec(cls, room_spec:dict):
        return cls(room_spec['dimensions'], room_spec['rt60_tgt'], room_spec.get('absorption', None))


class RIRGenerator:
    '''
    # RIRGenerator
    Generate a room impulse response to be used in the simulation pipeline
    '''
    def __init__(self, room:Room):
        self.room = room
        
    def add_microphones(self, microphones:list[list[float]]):
        mics = np.array(microphones).transpose()
        self.room.room.add_microphone_array(pra.MicrophoneArray(mics, self.room.room.fs))
        
    def add_source(self, position:list[float], signal:np.ndarray, delay:float):
        self.room.room.add_source(position, signal=signal, delay=delay)

    @classmethod
    def from_room_spec(cls, room_spec:dict):
        room = Room.from_room_spec(room_spec['room'])
        rir_gen = cls(room)
        if 'source' in room_spec:
            _, audio = wavfile.read(room_spec['source']['file'])
            rir_gen.add_source(room_spec['source']['position'], audio, room_spec['source']['delay'])
        if 'microphones' in room_spec:
            rir_gen.add_microphones(room_spec['microphones'])
        return rir_gen
    
    def simulate(self):
        self.room.room.simulate()
        
    def measure_rt60(self):
        return self.room.measure_rt60()
    def save_audio(self, output_file:str):
        self.room.mic_array.to_wav(
            output_file,
            norm=True,
            bitdepth=np.int16,
        )
import numpy as np
import pyroomacoustics as pra
import yaml
import config_handler
from scipy.io import wavfile
from scipy import signal
import logging
from Utils import Utils
logger = logging.getLogger(__name__)
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
        if 'sources' in room_spec:
            for source in room_spec['sources']:
                if source.get('file') is not None:
                    sr, audio = wavfile.read(source['file'])
                    if sr != config.audio.sample_rate:
                        logger.warning("Source %s has mismatched sample rate, resampling to %dHz",source['file'],config.audio.sample_rate) 
                        n_samples = round(len(audio) * float(config.audio.sample_rate) / sr)
                        audio = signal.resample(audio, n_samples)
                    rir_gen.add_source(source['position'], audio, source['delay'])
                elif source.get('noise') is not None:
                    noise = Utils.GenSignal('noise', source['noise']['length'], config.audio.sample_rate, format='real')
                    rir_gen.add_source(source['position'], noise, source['delay'])
                else:
                    logger.warning("Source %s has no valid audio file or noise definition, skipping...", source)
        if 'microphones' in room_spec:
            rir_gen.add_microphones(room_spec['microphones'])
        return rir_gen
    
    def _simulate(self):
        self.room.room.simulate()
        
    def measure_rt60(self):
        return self.room.measure_rt60()
    def save_audio(self, output_file:str):
        self.room.mic_array.to_wav(
            output_file,
            norm=True,
            bitdepth=np.int16,
        )
    def get_acoustic_transfer_functions(self):
        self._simulate()
        # a[0] represents the first sim result (if using multiple sim techniques, default is ISM)
        h = [a[0] for a in self.room.room.rir]
        # offset h to allow for negative time indexes
        h_offset = [np.pad(h_i, (len(h_i), 0), 'constant', constant_values=(0,0)) for h_i in h]
        return h_offset
    
    def get_mic_audio(self):
        return [mic for mic in self.room.mic_array]
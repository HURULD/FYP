import pyroomacoustics as pra

class RIRGenerator():
    '''
    # RIRGenerator
    Generate a room impulse response to be used in thee simulation pipeline
    '''
    def __init__(self,
                 dimensions: list[float],):
        self.room = pra.ShoeBox(
            dimensions,
            fs=44100, 
            materials=pra.Material(e_absorption), 
            use_rand_ism=True,
            max_rand_disp=0.5
        )
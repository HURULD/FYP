from typing import Literal
import numpy as np

def GenSignal(sig_type:Literal['noise','sine','cosine','impulse'],len:float,sample_rate:int,frequency,format:Literal['real','complex']='complex'):
    output=np.zeros(round(len*sample_rate))
    if sig_type == 'noise':
        if format == 'real' or format == 'complex':
            output = np.random.normal(0,1,size=round(len*sample_rate))
            return output
    if sig_type == 'sine':
        output = np.sin(np.arange(len*sample_rate))
    else:
        raise NotImplementedError()
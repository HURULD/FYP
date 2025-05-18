from typing import Literal, Optional
import numpy as np

def GenSignal(sig_type:Literal['noise','sine','cosine','impulse'],len:float,sample_rate:int,frequency:Optional[int]=0,format:Literal['real','complex']='complex'):
    output=np.zeros(round(len*sample_rate))
    if sig_type == 'noise':
        if format == 'real' or format == 'complex':
            output = np.random.normal(0,1,size=round(len*sample_rate))
            return output
    if sig_type == 'sine':
        output = np.sin(np.arange(len*sample_rate)*frequency)
    else:
        raise NotImplementedError()
    
def MovingAverage(x:np.ndarray, window_size=10):
    out = np.array([])
    bins = np.array_split(x, np.floor(len(x)/window_size))
    bins = np.apply_over_axes(np.mean,np.array(bins),1)
    for bin in bins:
        np.append(out, bin)
    return out
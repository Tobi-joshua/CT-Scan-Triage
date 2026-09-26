import numpy as np
from src.ct import window_hu, sample_indices

def test_window_hu_bounds():
    x=np.array([-2000.,-1350.,-600.,150.,1000.])
    y=window_hu(x,center=-600,width=1500)
    assert np.all(y>=0) and np.all(y<=1)
    assert y[0]==0 and y[-1]==1

def test_sample_indices():
    idx=sample_indices(100,32)
    assert idx[0]==0 and idx[-1]==99
    assert len(idx)==32
    assert np.all(np.diff(idx)>0)

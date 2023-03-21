import numpy as np
import matplotlib.pyplot as plt
from utils import *
import pandas as pd
import bnlearn
from os.path import dirname, abspath, join

if __name__=='__main__':
    path = dirname(abspath(__file__))
    data = pd.read_stata(join(path, 'close_college.dta'))
    data['lwage'] = data['lwage'].apply(lambda x: int(x))
    data_scaled = (data - data.min()) / (data.max() - data.min())
    structure = bnlearn.structure_learning.fit(data_scaled, methodtype='hc', verbose=True)
    print(structure)


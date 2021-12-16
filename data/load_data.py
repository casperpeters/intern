import h5py
import numpy as np
import os
import sys

def load_data():
    path = os.path.dirname(os.getcwd())
    sys.path.append(path)
    data_obj = h5py.File(path + '/data/real_zebrafish_data/subject_1_reconv_spikes.h5', 'r')
    behavior = np.array(data_obj['Data']['behavior'])
    coordinates = np.array(data_obj['Data']['coords'])
    df = np.array(data_obj['Data']['df'])
    spikes = np.array(data_obj['Data']['spikes'])
    stimulus = np.array(data_obj['Data']['stimulus'])

    return spikes, behavior, coordinates, df, stimulus


def load_data_thijs():
    path = os.path.dirname(os.getcwd())
    sys.path.append(path)

    data_obj = h5py.File(path + '/data/real_zebrafish_data/thijs.h5', 'r')
    spikes = np.transpose(np.array(data_obj['Data']['Brain']['Analysis']['ThresholdedSpikes']))
    coordinates = np.transpose(np.array(data_obj['Data']['Brain']['Coordinates']))
    times = np.array(data_obj['Data']['Brain']['Times'])

    return spikes, coordinates, times

import csv
import numpy as np
import os

data_dir = os.path.dirname(os.path.abspath(__file__))
def load_data(path, nominal=True):
    path = os.path.join(data_dir, path)
    with open(path, 'r', encoding='utf8') as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data_type = 'a25' if nominal else np.int
        data = np.empty((n_samples, n_features), data_type)
        target = np.empty((n_samples,), dtype=data_type)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=data_type)
            target[i] = np.asarray(ir[-1], dtype=data_type)

    return data, target, target_names

def load_contact_lenses():
    return load_data('contact_lenses.csv')

def load_will_wait():
    return load_data('will_wait.csv')

def load_weather():
    return load_data('weather.csv')

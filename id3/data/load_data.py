import csv
import numpy as np
import os
import io

data_dir = os.path.dirname(os.path.abspath(__file__))


def load_data(path, nominal=True):
    """Loads data from path

    Parameters
    ----------
    path : String
        Name of file to be loaded.

    nominal : Boolean
        Flag if data should be loaded as nominal

    Returns
    -------
    data : np.array
        A 2D array with each column representing a feature of the data and
        each row representing a sample.

    target : np.array
        A 1D array with the target variables for the samples in
        the loaded data.

    target_names : np.array
        A 1D array with the classification names for the targets
        in the loaded data.

    """
    path = os.path.join(data_dir, path)
    with io.open(path, 'r', encoding='utf-8') as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data_type = 'a25' if nominal else np.int
        data = np.empty((n_samples, n_features), data_type)
        target = np.empty((n_samples, ), dtype=data_type)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=data_type)
            target[i] = np.asarray(ir[-1], dtype=data_type)

    return data, target, target_names

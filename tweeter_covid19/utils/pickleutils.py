import os
import pickle

from tweeter_covid19.utils import get_file_name


def data_split(path=None):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, 'rb') as fid:
        dictionary = pickle.load(fid)
        fid.close()
    x = []
    y = []
    for key, value in dictionary.items():
        y.append(value['class'])
        x.append(value['vector'])
    return x, y


def write_pickle_data(path=None, data=None):
    if path is None or data is None:
        return None
    with open(path, "wb") as fid:
        pickle.dump(data, fid)
        fid.close()


def read_pickle_data(path=None):
    if path is None or not os.path.isfile(path):
        return None
    with open(path, 'rb') as fid:
        data = pickle.load(fid)
        fid.close()
        return data


# TODO : This function is not tested yet.
def backup_pickle_file(backup_name, write_path):
    if os.path.exists(write_path):
        write_dir = get_file_name(write_path, 1, directory_only=True)
        if os.path.exists(os.path.join(write_dir, backup_name)):
            os.remove(os.path.join(write_dir, backup_name))
        os.rename(write_path, os.path.join(write_dir, backup_name))

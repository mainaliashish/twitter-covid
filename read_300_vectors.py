import pickle
import os

if __name__ == '__main__':
    logger_path = os.path.join('data', '300_vectors', 'logger.pkl')
    tokens = pickle.load(open(logger_path, "rb"))
    path = tokens['path'][0]
    vector = pickle.load(open(path, "rb"))
    print(vector)



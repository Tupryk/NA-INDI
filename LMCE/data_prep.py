import os
import rowan
import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

import LMCE.cfusdlog as cfusdlog


def feature_label_from_data(data: dict, residual_func, payload: bool=False):
    features = []
    labels = []

    payload_char = "P" if payload else ""

    r = []
    for j in range(0, len(data['timestamp'])):
        q = rowan.from_euler(data[f"ctrlLee{payload_char}.rpyx"][j],
                             data[f"ctrlLee{payload_char}.rpyy"][j],
                             data[f"ctrlLee{payload_char}.rpyz"][j], "xyz", "extrinsic")
        R = rowan.to_matrix(q)[:, :2]
        R = R.reshape(1, 6)
        r.append(R[0])

    r = np.array(r)
    
    accs = np.array([data["stateEstimateZ.ax"],
                     data["stateEstimateZ.ay"],
                     data["stateEstimateZ.az"]], dtype=np.float64).T * 1e-3
    
    vels = np.array([data["stateEstimateZ.vx"],
                     data["stateEstimateZ.vy"],
                     data["stateEstimateZ.vz"]], dtype=np.float64).T * 1e-3
    
    gyro = np.array([data[f"ctrlLee{payload_char}.omegax"],
                     data[f"ctrlLee{payload_char}.omegay"],
                     data[f"ctrlLee{payload_char}.omegaz"]]).T
    
    motors = np.array([data["motor.m1"],
                       data["motor.m2"],
                       data["motor.m3"],
                       data["motor.m4"]]).T * 1e-4
    
    if payload:
        payload_acc = np.array([
            data["ctrlLeeP.plAccx"],
            data["ctrlLeeP.plAccy"],
            data["ctrlLeeP.plAccz"],
        ]).T

        payload_vel = np.array([
            data["ctrlLeeP.plVelx"],
            data["ctrlLeeP.plVelx"],
            data["ctrlLeeP.plVelx"],
        ]).T

        payload_pos = np.array([
            data["stateEstimateZ.px"],
            data["stateEstimateZ.py"],
            data["stateEstimateZ.pz"],
        ]).T * 1e-3

    features = np.append(r, accs, axis=1)
    features = np.append(features, vels, axis=1)
    features = np.append(features, gyro, axis=1)
    features = np.append(features, motors, axis=1)

    # if payload:
    #     features = np.append(features, payload_acc, axis=1)
    #     features = np.append(features, payload_vel, axis=1)
    #     features = np.append(features, payload_pos, axis=1)

    f_a, tau_a = residual_func(data)
    labels = np.append(f_a, tau_a, axis=1)

    return features, labels


def prepare_data(file_paths: list,
                 residual_func,
                 save_as: str="",
                 shuffle_data: bool=True,
                 overwrite: bool=True,
                 verbose: int=0,
                 cutoffs: dict={},
                 payload: bool=False):
    
    if os.path.exists(f"./data/{save_as}.npz") and not overwrite:
        print("Data already exists, loading from files...")
        loaded_arrays = np.load(f"./data/{save_as}.npz")
        X = loaded_arrays['X']
        y = loaded_arrays['y']
        return X, y
    
    X = []
    y = []
    
    if verbose:
        pbar = tqdm(file_paths, unit="files")
    else:
        pbar = file_paths

    for file_path in pbar:
        try:
            data_usd = cfusdlog.decode(file_path)
        except:
            if verbose > 1:
                print(f"Failed to load {file_path}!")
            continue

        data = data_usd['fixedFrequency']
        idx = int(file_path[-2:])
        if idx in cutoffs:
            for k in data.keys():
                data[k] = data[k][:cutoffs[idx]]

        features, labels = feature_label_from_data(data, residual_func, payload=payload)
        X.extend(features)
        y.extend(labels)

    X = np.array(X)
    y = np.array(y)
    if shuffle_data:
        X, y = shuffle(X, y)

    if save_as:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        np.savez(f'./data/{save_as}.npz', X=X, y=y)

    return X, y

def create_dataloader(X: np.ndarray, y: np.ndarray):

    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64)

    return dataloader

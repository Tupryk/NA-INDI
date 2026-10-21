import os
import rowan
import torch
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader


def feature_label_from_data(
    data: dict, residual_func, use_history: bool=True,
    payload: bool=False, kappas: np.ndarray=np.array([])):
    
    features = []
    labels = []

    r = []
    for i in range(0, len(data["timestamp"])):
        q = data["rot"]
        R = rowan.to_matrix(q[i])[:, :2]
        R = R.reshape(1, 6)
        r.append(R[0])

    r = np.array(r)
    
    accs = data["acc"]
    vels = data["vel"]
    gyro = data["ang_vel"]
    motors = data["pwm"].astype(np.float64)
    if len(kappas):
        rpm2pwmA = -0.12128823778162669
        rpm2pwmB = 4.2310782971594264e-05
        pwm_normalized = motors / 65535.0
        rpm = (pwm_normalized - rpm2pwmA) / rpm2pwmB
        motors = rpm*rpm * kappas
    
    features = np.append(r[1:], accs[1:], axis=1)
    features = np.append(features, vels[1:], axis=1)
    features = np.append(features, gyro[1:], axis=1)
    features = np.append(features, motors[1:], axis=1)
    
    if use_history:
        features = np.append(features, r[:-1], axis=1)
        features = np.append(features, accs[:-1], axis=1)
        features = np.append(features, vels[:-1], axis=1)
        features = np.append(features, gyro[:-1], axis=1)
        features = np.append(features, motors[:-1], axis=1)

    if payload:
        payload_acc = data["p_acc"]
        payload_vel = data["p_vel"]
        payload_dir = data["p_dir"]
        # payload_pos = data["p_pos"]
        
        features = np.append(features, payload_acc[1:], axis=1)
        features = np.append(features, payload_vel[1:], axis=1)
        # features = np.append(features, payload_pos, axis=1)
        features = np.append(features, payload_dir[1:], axis=1)
        
        l = np.linalg.norm(data["p_pos"] - data["pos"], axis=1)
        qidot = (data["p_vel"] - data["vel"])
        qidot /= l.reshape(-1, 1)
        
        features = np.append(features, qidot[1:], axis=1)

    f_a, tau_a = residual_func(data)
    labels = np.append(f_a, tau_a, axis=1)[1:]

    return features, labels


def prepare_data(file_paths: list,
                 residual_func,
                 save_as: str="",
                 shuffle_data: bool=True,
                 overwrite: bool=True,
                 verbose: int=0,
                 cutoffs: dict={},
                 use_history: bool=False,
                 kappas: np.ndarray=np.array([]),
                 payload: bool=False):
    
    if save_as and os.path.exists(f"./data/{save_as}_X.npz") and not overwrite:
        print("Data already exists, loading from files...")
        X = np.load(f"./data/{save_as}_X.npz")["arr_0"]
        y = np.load(f"./data/{save_as}_y.npz")["arr_0"]
        return X, y
    
    X = []
    y = []
    
    if verbose:
        pbar = tqdm(file_paths, unit="files")
    else:
        pbar = file_paths

    for file_path in pbar:
        try:
            data = np.load(file_path, allow_pickle=True).item()
        except:
            if verbose:
                print(f"Failed to load {file_path}!")
            continue
        
        if len(cutoffs.keys()):
            idx = int(file_path[-2:])
            if idx in cutoffs:
                for k in data.keys():
                    data[k] = data[k][:cutoffs[idx]]

        try:
            features, labels = feature_label_from_data(data, residual_func, use_history=use_history, kappas=kappas, payload=payload)
            X.extend(features)
            y.extend(labels)
        except Exception as e:
            print(f"Error while calculating residuals for {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    if shuffle_data:
        X, y = shuffle(X, y)

    if save_as:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        np.savez(f'./data/{save_as}_X.npz', X)
        np.savez(f'./data/{save_as}_y.npz', y)

    return X, y

def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int=256):

    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

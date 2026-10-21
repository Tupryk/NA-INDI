import os
import shutil
import numpy as np
from omegaconf import OmegaConf
from LMCE.networks import MLP, ResnetMLP

from LMCE.model_to_c_conversion import exportNet, nn_c_model_test

model_path = "./outputs/2025-11-06/21-38-43"


config_path = os.path.join(model_path, ".hydra/config.yaml")
cfg = OmegaConf.load(config_path)

X_train = np.load(f"./data/{cfg.train_data_path}_X.npz")["arr_0"]
y_train = np.load(f"./data/{cfg.train_data_path}_y.npz")["arr_0"]
X_test = np.load(f"./data/{cfg.test_data_path}_X.npz")["arr_0"]
y_test = np.load(f"./data/{cfg.test_data_path}_y.npz")["arr_0"]

X_maxs = np.load(os.path.join(model_path, "min_max.npz"))["X_maxs"]
y_maxs = np.load(os.path.join(model_path, "min_max.npz"))["y_maxs"]
X_mins = np.load(os.path.join(model_path, "min_max.npz"))["X_mins"]
y_mins = np.load(os.path.join(model_path, "min_max.npz"))["y_mins"]

if cfg.model_type == "mlp":
    model = MLP(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_layers=cfg.net_arch)
elif cfg.model_type == "resnet":
    model = ResnetMLP(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_layers=cfg.net_arch)
else:
    raise Exception(f"Model type {cfg.model_type} does not exist!")

model_path = os.path.join(model_path, "model.pth")
model.load(path=model_path)

# Generate the model c code
exportNet(model_path, X_mins=X_mins, X_maxs=X_maxs, y_mins=y_mins, y_maxs=y_maxs, info="Neural network for the 2.1 Crazyflie", resnet_arch=True)

# Test if the generated model gives the same outputs as the original
model.double()
passed = nn_c_model_test(model, X_mins=X_mins, X_maxs=X_maxs, y_mins=y_mins, y_maxs=y_maxs, in_dim=X_train.shape[1], test_data=X_test, verbose=0)

if passed:
    try:
        shutil.copyfile("./LMCE/c_utils/nn.c", "../crazyflie-firmware/src/modules/src/controller/nn.c")
        shutil.copyfile("./LMCE/c_utils/nn.h", "../crazyflie-firmware/src/modules/interface/controller/nn.h")
        print("Successfully copied nn to firmware.")
    except:
        print("Crazyflie firmware not found.")
import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

from LMCE.networks import MLP
from LMCE.utils import min_max
from LMCE.trainers import train
from LMCE.residual_calculation import indi_residuals
from LMCE.data_prep import prepare_data, create_dataloader
from LMCE.model_to_c_conversion import exportNet, nn_c_model_test

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']

kappas = np.array([
    2.0938753372837369e-10,
    2.2766702598220073e-10,
    1.906494181367591e-10,
    2.4578364131636854e-10
])


# Get train data paths
training_data_paths = [f"./data/single/train_clean/{i:02}.npy" for i in range(53)]
testing_data_paths = [f"./data/single/test_clean/00.npy"]

residual_func = lambda data: indi_residuals(data, make_smooth=True, verbose=0, spline_segments=200)

X_train, y_train_ = prepare_data(training_data_paths,
                                 save_as="train_data_hist",
                                 residual_func=residual_func,
                                 overwrite=False,
                                 payload=False,
                                 kappas=kappas,
                                 use_history=True,
                                 verbose=1)

X_test, y_test_ = prepare_data(testing_data_paths,
                               save_as="test_data_hist",
                               shuffle_data=False,
                               residual_func=residual_func,
                               overwrite=False,
                               payload=False,
                               kappas=kappas,
                               use_history=True,
                               verbose=1)

print("X_train.shape: ", X_train.shape)
print("y_train_.shape: ", y_train_.shape)
print("X_test.shape: ", X_test.shape)
print("y_test_.shape: ", y_test_.shape)

_, X_mins, X_maxs = min_max(X_train.copy())
X_mins[:6] = -1.
X_maxs[:6] = 1.
X_train, _, _ = min_max(X_train, X_mins, X_maxs)
X_test, _, _ = min_max(X_test, X_mins, X_maxs)

y_train, y_mins, y_maxs = min_max(y_train_)
y_test, _, _ = min_max(y_test_, y_mins, y_maxs)

train_loader = create_dataloader(X_train, y_train)
val_loader = create_dataloader(X_test, y_test)

print("Train size: ", X_train.shape[0])
print("Test size: ", X_test.shape[0])
print("X dim: ", X_train.shape[1], "y dim: ", y_train.shape[1])

# Create and train neural network
model = MLP(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_layers=[24, 24, 24])
model = train(model, train_loader, val_loader, lr=3e-4, scheduler_gamma=.92, scheduler_step_size=10, device="cpu", epochs=42, verbose=1)
model.save("mlp")
model.load("mlp")

residual_func = lambda data: indi_residuals(data, make_smooth=False, verbose=0)

X_test, y_test = prepare_data([testing_data_paths[0]],
                               save_as="test_data",
                               shuffle_data=False,
                               residual_func=residual_func,
                               overwrite=True,
                               payload=False,
                               kappas=kappas,
                               use_history=True,
                               verbose=1)

X_test, _, _ = min_max(X_test, X_mins, X_maxs)
y_test, _, _ = min_max(y_test, y_mins, y_maxs)

tensor_input = torch.from_numpy(X_test).type(torch.float32)
pred = model.forward(tensor_input).detach().numpy()
error = np.abs(y_test-pred)
print(f"Average force error: (x, y, z) N -> ({np.mean(error[:, 0]):.4f}, {np.mean(error[:, 1]):.4f}, {np.mean(error[:, 2]):.4f}) N")
print(f"Average torque error: (x, y, z) N -> ({np.mean(error[:, 3]):.4f}, {np.mean(error[:, 4]):.4f}, {np.mean(error[:, 5]):.4f}) N")

y_test_scaled = (y_test + 1) * .5 * (y_maxs - y_mins) + y_mins
pred_scaled = (pred + 1) * .5 * (y_maxs - y_mins) + y_mins

dims = ['x', 'y', 'z']
force_type = ["Force", "Torque"]
for i in range(6):
    fig, ax = plt.subplots()
    ax.set_title(f"{force_type[i//3]} {dims[i%3]}")
    ax.plot(y_test_scaled[:, i], label="INDI")
    ax.plot(pred_scaled[:, i], label="NN")
    ax.legend()
    
    # axins = zoomed_inset_axes(ax, 3, loc=3)
    # axins.plot(y_test_scaled[:, i])
    # axins.plot(pred_scaled[:, i])

    # x1, x2, y1, y2 = 200, 240, .027, .037
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)

    # plt.xticks(visible=False)
    # plt.yticks(visible=False)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.draw()
    plt.show()
    
# Generate the model c code
model_path = "./models/mlp.pth"
exportNet(model_path, X_mins=X_mins, X_maxs=X_maxs, y_mins=y_mins, y_maxs=y_maxs, resnet_arch=True, info="Neural network for the 2.1 Crazyflie")

# Test if the generated model gives the same outputs as the original
model.double()
nn_c_model_test(model, X_mins=X_mins, X_maxs=X_maxs, y_mins=y_mins, y_maxs=y_maxs, in_dim=X_train.shape[1], test_data=X_test, verbose=0)

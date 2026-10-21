import os
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from LMCE.utils import min_max
from LMCE.trainers import train
from LMCE.networks import MLP, ResnetMLP
from LMCE.data_prep import create_dataloader

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    
    new_train_X = []
    new_train_y = []
    
    X_train = np.load(f"./data/{cfg.train_data_path}_X.npz")["arr_0"]
    y_train = np.load(f"./data/{cfg.train_data_path}_y.npz")["arr_0"][:, :6]
    
    # for i in range(X_train.shape[0]):
    #     vel = np.linalg.norm(X_train[i][9:12])
    #     if vel > 0.6 or vel > 0.15 or np.random.random() < 0.05:
    #         new_train_X.append(X_train[i])
    #         new_train_y.append(y_train[i])
    
    # X_train = np.array(new_train_X)
    # y_train = np.array(new_train_y)
    
    X_test = np.load(f"./data/{cfg.test_data_path}_X.npz")["arr_0"][10:]
    y_test = np.load(f"./data/{cfg.test_data_path}_y.npz")["arr_0"][10:][:, :6]

    _, X_mins, X_maxs = min_max(X_train.copy())
    X_mins[:6] = -1.
    X_maxs[:6] = 1.
    X_train, _, _ = min_max(X_train, X_mins, X_maxs)
    X_test, _, _ = min_max(X_test, X_mins, X_maxs)

    y_train, y_mins, y_maxs = min_max(y_train)
    y_test, _, _ = min_max(y_test, y_mins, y_maxs)
    
    min_max_path = os.path.join(cfg.output_dir, "min_max.npz")
    np.savez(min_max_path, X_maxs=X_maxs, y_maxs=y_maxs, X_mins=X_mins, y_mins=y_mins)

    train_loader = create_dataloader(X_train, y_train, batch_size=cfg.batch_size)
    val_loader = create_dataloader(X_test, y_test, batch_size=cfg.batch_size)

    print("Train size: ", X_train.shape[0])
    print("Test size: ", X_test.shape[0])
    print("X dim: ", X_train.shape[1], "y dim: ", y_train.shape[1])
    
    trial_count = 1
    out_dim = y_train.shape[1]
    mean_error = np.zeros(out_dim)
    for _ in range(trial_count):
        if cfg.model_type == "mlp":
            model = MLP(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_layers=cfg.net_arch)
        elif cfg.model_type == "resnet":
            model = ResnetMLP(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_layers=cfg.net_arch)
        else:
            raise Exception(f"Model type {cfg.model_type} does not exist!")

        model, train_losses, val_losses = train(
            model,
            train_loader,
            val_loader,
            lr=cfg.lr,
            loss_type=cfg.loss_type,
            scheduler_gamma=cfg.scheduler_gamma,
            scheduler_step_size=cfg.scheduler_step_size,
            epochs=cfg.epochs,
            device="cpu"
        )
    
        tensor_input = torch.from_numpy(X_test).type(torch.float32)
        pred = model.forward(tensor_input).detach().numpy()
        error = np.abs(y_test-pred)
        mean_error += np.mean(error, axis=0)
    
    mean_error /= trial_count
    
    print(f"Average force error: (x, y, z) N -> ({mean_error[0]:.4f}, {mean_error[1]:.4f}, {mean_error[2]:.4f}) N")
    print(f"Average torque error: (x, y, z) N -> ({mean_error[3]:.4f}, {mean_error[4]:.4f}, {mean_error[5]:.4f}) N")
    if out_dim > 6:
        print(f"Average payload error: (x, y, z) N -> ({mean_error[6]:.4f}, {mean_error[7]:.4f}, {mean_error[8]:.4f}) N")
        print(f"Average cable error: (x, y, z) N -> ({mean_error[9]:.4f}, {mean_error[10]:.4f}, {mean_error[11]:.4f}) N")
    
    # plt.plot(train_losses, label="Train loss")
    # plt.plot(val_losses, label="Val loss")
    # plt.legend()
    # plt.savefig(os.path.join(cfg.output_dir, "training.png"))
    
    model.save(path=os.path.join(cfg.output_dir, "model.pth"))

    with open(os.path.join(cfg.output_dir, "eval.txt"), "w") as f:
        f.write(f"{mean_error.tolist()}".replace(",", "").replace("[", "").replace("]", ""))

    y_test_scaled = (y_test + 1) * .5 * (y_maxs - y_mins) + y_mins
    pred_scaled = (pred + 1) * .5 * (y_maxs - y_mins) + y_mins

    force_type = ["Force", "Torque"]
    dims = ["x", "y", "z"]

    if out_dim > 5:
        things = [
            "Force X", "Force Y", "Force Z",
            "Payload X", "Payload Y", "Payload Z",
            "Cable X", "Cable Y", "Cable Z",
            "Torque X", "Torque Y", "Torque Z"
        ]
        # fig, axes = plt.subplots(6, 2, figsize=(12, 12))
        # axes = axes.flatten()

        # for i in range(out_dim):
        #     axes[i].set_title(things[i])
        #     axes[i].plot(y_test_scaled[:, i], label="INDI")
        #     axes[i].plot(pred_scaled[:, i], label="LINDI")
        #     axes[i].legend()
        #     axes[i].grid(True)
        plt.figure(dpi=300)
        plt.plot(y_test_scaled[:, 1], label="INDI")
        plt.plot(pred_scaled[:, 1], label="LINDI")
        plt.xlabel("Timestep")
        plt.ylabel(r"Residual Accelerations (m/s$^2$)")
        plt.grid(True)
        plt.legend()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes = axes.flatten()

        for i in range(6):
            axes[i].set_title(f"{force_type[i // 3]} {dims[i % 3]}")
            axes[i].plot(y_test_scaled[:, i], label="INDI")
            axes[i].plot(pred_scaled[:, i], label="NN")
            axes[i].legend()
            axes[i].grid(True)

    # fig.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "results.png"))


if __name__ == "__main__":
    main()

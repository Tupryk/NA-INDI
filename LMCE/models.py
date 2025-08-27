import os
import torch
import pickle
import numpy as np
from torch import nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_layers: list[int]=[24, 24, 24]):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size

        self.test_lossess = []
        self.train_lossess = []
        
        layers = [nn.Linear(self.input_size, hidden_layers[0]), nn.LeakyReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_layers[-1], self.output_size))
        self.layers = nn.Sequential(*layers)

        for layer in self.modules(): # Improves Torque Z output
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        pred = self.layers(x)
        return pred
    
    def save(self, name: str="mlp"):
        if not os.path.exists("./models"):
            os.makedirs("./models")
        torch.save(self.state_dict(), f"./models/{name}.pth")


class DTE():
    def __init__(self, n_estimators=5,
                       max_depth=10,
                       min_samples_split=400,
                       ccp_alpha=0.0
                       ):
        
        self.model = MultiOutputRegressor(
                        RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            ccp_alpha=ccp_alpha
                        )
                    )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def forward(self, x: np.ndarray):
        return self.model.predict(x)

    def save(self, name: str="dte"):
        with open(f"./models/{name}.pkl", 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, name: str):
        with open(f"./models/{name}.pkl", 'rb') as f:
            self.model = pickle.load(f)

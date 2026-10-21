import numpy as np

from LMCE.data_prep import prepare_data
from LMCE.residual_calculation import indi_residuals


# Get train data paths
training_data_paths = [f"./data/single/train_clean/{i:02}.npy" for i in range(25)]
testing_data_paths = [f"./data/single/test_clean/00.npy"]
USE_PAYLOAD = "payload" in training_data_paths[0]
print(f"Using payload: {USE_PAYLOAD}")

residual_func = lambda data: indi_residuals(data, make_smooth=True, payload_extras=False, verbose=0, spline_segments=200)
kappas = np.array([])

X_train, y_train_ = prepare_data(training_data_paths,
                                 save_as="train_data_single_smooth",
                                 residual_func=residual_func,
                                 overwrite=False,
                                 payload=USE_PAYLOAD,
                                 kappas=kappas,
                                 verbose=1)

X_test, y_test_ = prepare_data(testing_data_paths,
                               save_as="test_data_single_smooth",
                               shuffle_data=False,
                               residual_func=residual_func,
                               overwrite=True,
                               payload=USE_PAYLOAD,
                               kappas=kappas,
                               verbose=1)

print("X_train.shape: ", X_train.shape)
print("y_train_.shape: ", y_train_.shape)
print("X_test.shape: ", X_test.shape)
print("y_test_.shape: ", y_test_.shape)
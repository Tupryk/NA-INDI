import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    filenames = [f"./data/data_clean/{i:02}.npy" for i in range(64)]
    
    pwm = None
    rpm = None
    for f in filenames:
        try:
            data = np.load(f, allow_pickle=True).item()
            if pwm is None:
                pwm = data["pwm"]
                rpm = data["rpm"]
            else:
                pwm = np.vstack((pwm, data["pwm"]))
                rpm = np.vstack((rpm, data["rpm"]))
        except:
            pass

    kappa_f = np.array([2.139974655714972e-10, 2.3783777845095615e-10, 1.9693330742680727e-10, 2.559402652634741e-10])
    force = kappa_f * rpm**2
    force = np.mean(force, axis=1)
    
    pwm = np.mean(pwm, axis=1)

    p = np.polyfit(pwm, force, deg=3)
    print(p)
    
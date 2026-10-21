import rowan
import numpy as np
import matplotlib.pyplot as plt

from LMCE.utils import SplineFitter


t2t = 0.006  # thrust-to-torque ratio
g2N = 0.00981
GRAVITY = 9.81
ARM_LENGTH = 0.046  # m


def project(this, on_that):
    projections = []
    for j in range(len(on_that)):
        on_that[j] /= np.linalg.norm(on_that[j])
        dot_product = np.dot(this[j], on_that[j])

        projected = (dot_product / np.dot(on_that[j], on_that[j])) * on_that[j]

        projections.append(projected)

    projections = np.array(projections)
    return projections


def project_onto_plane(v, n):
    projections = []
    for j in range(len(v)):
        n[j] = n[j] / np.linalg.norm(n[j])
        dot_product = np.dot(v[j], n[j])
        projection_on_normal = dot_product * n[j]
        projection_on_plane = v[j] - projection_on_normal

        projections.append(projection_on_plane)

    projections = np.array(projections)
    return projections


def payload_fa(data_usd: dict,
               mass: float=.0366,
               mass_payload: float=.005):

    start_time = data_usd["timestamp"][0]
    t = (data_usd["timestamp"] - start_time) / 1e3

    pos = data_usd["pos"]
    
    pos_payload = data_usd["p_pos"]
    
    acc = data_usd["acc"] - np.array([0., 0., 9.81])
    
    sf = SplineFitter()

    acc_payload_spline = []
    for k, axis in enumerate(["x", "y", "z"]):
        sf.fit(t, pos_payload[:,k], 200)
        spline = sf.evaldd(t)
        acc_payload_spline.append(spline)

    acc_payload_spline = np.array(acc_payload_spline).T

    # fa vs tau_a
    rpm = data_usd["rpm"]

    q = data_usd["rot"]

    kappa_f = np.array([2.139974655714972e-10, 2.3783777845095615e-10, 1.9693330742680727e-10, 2.559402652634741e-10])
    force = kappa_f * rpm**2
    f = force.sum(axis=1)
    u = np.zeros_like(rpm)
    u[:,2] = f

    g = np.array([0, 0, -9.81])

    cable_q = (pos_payload - pos) / np.tile(np.linalg.norm(pos_payload - pos, axis=1), [3, 1]).T

    fa = []
    for i in range(force.shape[0]):
        T = (-mass_payload * cable_q[i]).dot(acc_payload_spline[i] - g)
        fa.append(mass * acc[i] - mass * g - rowan.rotate(q[i], u[i]) - T*cable_q[i])
    fa = np.array(fa)

    return fa


def residual(data_usd: dict,
             use_rpm: bool = True,
             verbose: int=0):

    # Set mass
    total_mass = .0366
    start_time = data_usd["timestamp"][0]
    t = (data_usd["timestamp"] - start_time) / 1e3

    q = data_usd["rot"]
    acc_world = data_usd["acc"]

    # Get total force
    if use_rpm:
        rpm = data_usd["rpm"]
        kappa_f = np.array([2.139974655714972e-10, 2.3783777845095615e-10, 1.9693330742680727e-10, 2.559402652634741e-10])
        force = kappa_f * rpm**2

    else:  # pwm
        pwm = data_usd["pwm"]
        # coeffs = [6.47418985e-11, -5.03191974e-06, 1.85109791e-01]
        coeffs = [5.95534920e-15, -8.25263291e-10, 3.91162380e-05, -5.41800603e-01]
        force = coeffs[0] * pwm**3 + coeffs[1] * pwm**2 + coeffs[2] * pwm + coeffs[3]

    eta = np.empty((force.shape[0], 4))
    f_u = np.empty((force.shape[0], 3))
    tau_u = np.empty((force.shape[0], 3))

    # Get residual forces
    arm = 0.707106781 * ARM_LENGTH

    B0 = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
    ])

    for k in range(force.shape[0]):
        eta[k] = np.dot(B0, force[k])
        f_u[k] = np.array([0, 0, eta[k, 0]])
        tau_u[k] = np.array([eta[k, 1], eta[k, 2], eta[k, 3]])

    f_a = (acc_world * total_mass - rowan.rotate(q, f_u)) / 0.0366

    omega = data_usd["ang_vel"]
    sf = SplineFitter()
    omega_dot_spline = []

    for k in range(3):
        sf.fit(t, omega[:,k], 50)
        spline = sf.evald(t)
        omega_dot_spline.append(spline)
        
    omega_dot_spline = np.array(omega_dot_spline).T
    J = np.diag([16.571710e-6, 16.655602e-6, 29.261652e-6])

    tau_a = []
    for i in range(force.shape[0]):
        eta = B0 @ force[i]
        tau_a.append(J @ omega_dot_spline[i] - np.cross(J @ omega[i], omega[i]) - eta[1:])

    tau_a = np.array(tau_a)

    return f_a, tau_a


def indi_residuals(data_usd: dict,
                   make_smooth: bool=True,
                   verbose: int=0,
                   spline_segments: int=50,
                   payload_extras: bool=False):

    f_a = data_usd["a_imu"] - data_usd["a_rpm"]
    tau_a = data_usd["tau_imu"] - data_usd["tau_rpm"]
    
    if payload_extras:
        f_a = np.append(f_a, data_usd["fpayload_imu"] - data_usd["fpayload_rpm"], axis=1)
        f_a = np.append(f_a, data_usd["fcable_imu"] - data_usd["fcable_rpm"], axis=1)

    if make_smooth:
        start_time = data_usd["timestamp"][0]
        t = (data_usd["timestamp"] - start_time) / 1e3

        f_a_noisy = f_a.copy()
        tau_a_noisy = tau_a.copy()
        sf = SplineFitter()

        for i in range(f_a.shape[1]):
            sf.fit(t, f_a[:, i], spline_segments)
            f_a[:, i] = sf.eval(t)
        
        for i in range(tau_a.shape[1]):
            sf.fit(t, tau_a[:, i], spline_segments)
            tau_a[:, i] = sf.eval(t)

    if verbose:
        axis = ['x', 'y', 'z']
        if payload_extras:
            fig, ax = plt.subplots(6, 2, figsize=(10, 15))
        else:
            fig, ax = plt.subplots(6, figsize=(10, 30))
        for i, f in enumerate(["Force", "Torque"]):
            for j, v in enumerate(axis):
                idx = j+i*3
                if make_smooth:
                    ax[idx, 0].plot(f_a_noisy[:, j] if f == "Force" else tau_a_noisy[:, j], label="Noisy")
                    ax[idx, 0].plot(f_a[:, j] if f == "Force" else tau_a[:, j], label="Smooth")
                else:
                    ax[idx, 0].plot(f_a[:, j] if f == "Force" else tau_a[:, j])
                ax[idx, 0].set_title(f'{f} {v}')
                ax[idx, 0].set_xlabel("Timestamp")
                ax[idx, 0].set_ylabel(f"{f} prediction")
                if make_smooth:
                    ax[idx, 0].legend()
            
            if f == "Force":
                for j in range(3, 9):
                    v = "Cable" if j > 5 else "Payload"
                    a = axis[j%3]
                    if make_smooth:
                        ax[j-3, 1].plot(f_a_noisy[:, j], label="Noisy")
                        ax[j-3, 1].plot(f_a[:, j], label="Smooth")
                    else:
                        ax[j-3, 1].plot(f_a[:, j] if f == "Force" else tau_a[:, j])
                    ax[j-3, 1].set_title(f'{f} {v} {a}')
                    ax[j-3, 1].set_xlabel("Timestamp")
                    ax[j-3, 1].set_ylabel(f"{f} prediction")
                    if make_smooth:
                        ax[j-3, 1].legend()
                
        plt.show()

    return f_a, tau_a

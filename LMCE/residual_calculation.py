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


def residual(data_usd: dict,
             is_brushless: bool = False,
             has_payload: bool = False,
             use_rpm: bool = True,
             force_per_rotor: bool=False,
             world_acc: bool=False,
             projection_payload_calc: bool=False,
             payload_mass: float=.0047,
             payload_acc_method: list[float]=["ctrlLeeP.plAccx_tq", "ctrlLeeP.plAccy_tq", "ctrlLeeP.plAccz_tq"]):

    # Set mass
    if is_brushless:
        total_mass = .0444
    else:
        total_mass = .0347

    # q = np.array([
    #     data_usd['stateEstimate.qw'],
    #     data_usd['stateEstimate.qx'],
    #     data_usd['stateEstimate.qy'],
    #     data_usd['stateEstimate.qz']]).T
    
    q = rowan.from_euler(data_usd["ctrlLee.rpyx"],
                         data_usd["ctrlLee.rpyy"],
                         data_usd["ctrlLee.rpyz"])

    # Get acceleration in world frame
    if not world_acc:
        # acc_body = np.array([
        #     data_usd['acc.x'],
        #     data_usd['acc.y'],
        #     data_usd['acc.z']]).T
        
        acc_body = np.array([
            data_usd['ctrlLee.a_imux'],
            data_usd['ctrlLee.a_imuy'],
            data_usd['ctrlLee.a_imuz']]).T

        acc_world = rowan.rotate(q, acc_body)

    else:
        acc_world = np.array([
            data_usd['stateEstimate.ax'],
            data_usd['stateEstimate.ay'],
            data_usd['stateEstimate.az']]).T

    acc_world *= 9.81

    # Get total forces
    if use_rpm:
        rpm = np.array([
            data_usd['rpm.m1'],
            data_usd['rpm.m2'],
            data_usd['rpm.m3'],
            data_usd['rpm.m4']]).T
        if is_brushless:
            force_in_grams = 4.310657321921365e-08 * rpm**2
        else:
            if force_per_rotor:
                force = rpm**2 * np.array([2.0938753372837369e-10,
                                           2.2766702598220073e-10,
                                           1.906494181367591e-10,
                                           2.4578364131636854e-10])
            else:
                force_in_grams = 2.40375893e-08 * rpm**2 + - \
                    3.74657423e-05 * rpm + -7.96100617e-02

    else:  # pwm
        pwm = np.array([
            data_usd['pwm.m1_pwm'],
            data_usd['pwm.m2_pwm'],
            data_usd['pwm.m3_pwm'],
            data_usd['pwm.m4_pwm']]).T
        if is_brushless:
            force_in_grams = -5.360718677769569 + pwm * 0.0005492858445116151
        else:
            force_in_grams = 1.65049399e-09 * pwm**2 + \
                9.44396129e-05 * pwm + -3.77748052e-01

    if not force_per_rotor:
        force = force_in_grams * g2N

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

    if has_payload:
        if projection_payload_calc:
            payload_dir = np.array([data_usd["stateEstimateZ.px"] - data_usd["stateEstimate.x"],
                                    data_usd["stateEstimateZ.py"] - data_usd["stateEstimate.y"],
                                    data_usd["stateEstimateZ.pz"] - data_usd["stateEstimate.z"]], dtype=np.float32).T
            f_a = total_mass * acc_world - rowan.rotate(q, f_u)
            f_a = project_onto_plane(f_a, payload_dir)

        else:
            payload_acc = np.array([data_usd[payload_acc_method[0]],
                                    data_usd[payload_acc_method[1]],
                                    data_usd[payload_acc_method[2]]]).T
            T_p = -payload_mass * payload_acc - payload_mass * np.array([0., 0., 9.81])
            f_a = total_mass * acc_world - rowan.rotate(q, f_u) - T_p
    else:
        f_a = total_mass * acc_world - rowan.rotate(q, f_u)

    return f_a, tau_u


def payload_fa(data_usd: dict,
               mass: float=.0366,
               mass_payload: float=.005):

    start_time = data_usd['timestamp'][0]
    t = (data_usd['timestamp'] - start_time) / 1e3

    pos = np.array([
        data_usd['stateEstimate.x'],
        data_usd['stateEstimate.y'],
        data_usd['stateEstimate.z']]).T
    
    pos_payload = np.array([
        data_usd['stateEstimateZ.px'],
        data_usd['stateEstimateZ.py'],
        data_usd['stateEstimateZ.pz']]).T / 1000.0
    
    acc = np.array([
        data_usd['stateEstimateZ.ax'],
        data_usd['stateEstimateZ.ay'],
        data_usd['stateEstimateZ.az'] - 9810]).T / 1000.0
    
    sf = SplineFitter()

    acc_payload_spline = []
    for k, axis in enumerate(["x", "y", "z"]):
        sf.fit(t, pos_payload[:,k], 200)
        spline = sf.evaldd(t)
        acc_payload_spline.append(spline)

    acc_payload_spline = np.array(acc_payload_spline).T

    # fa vs tau_a
    rpm = np.array([
        data_usd['rpm.m1'],
        data_usd['rpm.m2'],
        data_usd['rpm.m3'],
        data_usd['rpm.m4'],
    ]).T

    rpy = np.array([
        data_usd[f'ctrlLeeP.rpyx'],
        data_usd[f'ctrlLeeP.rpyy'],
        data_usd[f'ctrlLeeP.rpyz']]).T

    q = rowan.from_euler(rpy[:,0], rpy[:,1], rpy[:,2], "xyz", "extrinsic")

    kappa_f = np.array([2.139974655714972e-10, 2.3783777845095615e-10, 1.9693330742680727e-10, 2.559402652634741e-10])
    force = kappa_f * rpm**2
    f = force.sum(axis=1)
    u = np.zeros_like(rpy)
    u[:,2] = f

    g = np.array([0, 0, -9.81])

    cable_q = (pos_payload - pos) / np.tile(np.linalg.norm(pos_payload - pos, axis=1), [3, 1]).T

    fa = []
    for i in range(force.shape[0]):
        T = (-mass_payload * cable_q[i]).dot(acc_payload_spline[i] - g)
        fa.append(mass * acc[i] - mass * g - rowan.rotate(q[i], u[i]) - T*cable_q[i])
    fa = np.array(fa)

    return fa


def spline_tau_a(data_usd: dict):

    start_time = data_usd['timestamp'][0]
    t = (data_usd['timestamp'] - start_time) / 1e3
    
    omega = np.array([
        data_usd[f'ctrlLeeP.omegax'],
        data_usd[f'ctrlLeeP.omegay'],
        data_usd[f'ctrlLeeP.omegaz']
    ]).T
    
    rpm = np.array([
        data_usd['rpm.m1'],
        data_usd['rpm.m2'],
        data_usd['rpm.m3'],
        data_usd['rpm.m4'],
    ]).T
    kappa_f = np.array([2.139974655714972e-10, 2.3783777845095615e-10, 1.9693330742680727e-10, 2.559402652634741e-10])
    force = kappa_f * rpm**2

    J = np.diag([16.571710e-6, 16.655602e-6, 29.261652e-6])
    arm_length = 0.046
    arm = 0.707106781 * arm_length
    t2t = 0.006
    allocation_matrix = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
    ])

    sf = SplineFitter()
    omega_dot_spline = []

    for k in range(3):
        sf.fit(t, omega[:,k], 50)
        spline = sf.evald(t)
        omega_dot_spline.append(spline)
        
    omega_dot_spline = np.array(omega_dot_spline).T

    tau_a = []
    for i in range(force.shape[0]):
        eta = allocation_matrix @ force[i]
        tau_a.append(J @ omega_dot_spline[i] - np.cross(J @ omega[i], omega[i]) - eta[1:])

    tau_a = np.array(tau_a)
    return tau_a


def indi_residuals(data_usd: dict,
                   total_mass: float=.0366,
                   payload: bool=False,
                   make_smooth: bool=True,
                   show_fit: bool=False,
                   spline_segments: int=50):

    payload_char = "P" if payload else ""

    data_len = len(data_usd["timestamp"])

    if payload:
        f_a = np.zeros((data_len, 3))
    else:
        if payload and make_smooth:
            f_a = payload_fa(data_usd, total_mass)
        else:
            f_a = np.zeros((data_len, 3))
            f_a[:, 0] = (data_usd[f"ctrlLee{payload_char}.a_imu_fx"] - data_usd[f"ctrlLee{payload_char}.a_rpm_fx"]) * total_mass
            f_a[:, 1] = (data_usd[f"ctrlLee{payload_char}.a_imu_fy"] - data_usd[f"ctrlLee{payload_char}.a_rpm_fy"]) * total_mass
            f_a[:, 2] = (data_usd[f"ctrlLee{payload_char}.a_imu_fz"] - data_usd[f"ctrlLee{payload_char}.a_rpm_fz"]) * total_mass
    
    tau_a = np.zeros((data_len, 3))
    tau_a[:, 0] = data_usd[f"ctrlLee{payload_char}.tau_imu_fx"] - data_usd[f"ctrlLee{payload_char}.tau_rpm_fx"]
    tau_a[:, 1] = data_usd[f"ctrlLee{payload_char}.tau_imu_fy"] - data_usd[f"ctrlLee{payload_char}.tau_rpm_fy"]
    tau_a[:, 2] = data_usd[f"ctrlLee{payload_char}.tau_imu_fz"] - data_usd[f"ctrlLee{payload_char}.tau_rpm_fz"]
    if make_smooth:
        start_time = data_usd['timestamp'][0]
        t = (data_usd['timestamp'] - start_time) / 1e3
        tau_a_noisy = tau_a.copy()
        # tau_a = spline_tau_a(data_usd)
        for i in range(3):
            sf = SplineFitter()
            sf.fit(t, tau_a[:, i], spline_segments)
            tau_a[:, i] = sf.eval(t)

    if make_smooth:
        start_time = data_usd['timestamp'][0]
        t = (data_usd['timestamp'] - start_time) / 1e3

        f_a_noisy = f_a.copy()

        if make_smooth and not payload:
            for i in range(3):
                sf = SplineFitter()
                sf.fit(t, f_a[:, i], spline_segments)
                f_a[:, i] = sf.eval(t)

        if show_fit:
            fig, ax = plt.subplots(6, figsize=(10, 30))
            for i, f in enumerate(["Force", "Torque"]):
                for j, v in enumerate(["x", "y", "z"]):
                    idx = j+i*3
                    ax[idx].plot(f_a_noisy[:, j] if f == "Force" else tau_a_noisy[:, j], label="Noisy")
                    ax[idx].plot(f_a[:, j] if f == "Force" else tau_a[:, j], label="Smooth")
                    ax[idx].set_title(f'{f} {v}')
                    ax[idx].set_xlabel('Timestamp')
                    ax[idx].set_ylabel(f'{f} prediction')
                    ax[idx].legend()
            plt.show()

    return f_a, tau_a

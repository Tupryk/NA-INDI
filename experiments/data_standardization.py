"""
Data from the crazyflies always has different labels
and conventions. This script gives a base to read the
files and transform them to the same json format with
the same keys and standards.
"""
import os
import rowan
import numpy as np
import matplotlib.pyplot as plt

from LMCE import cfusdlog


def transform(data: dict, has_payload: bool=False) -> dict:
    
    extra_char = "P" if has_payload else ""
    
    timestamp = np.array(data["timestamp"])

    pos = np.array([data["stateEstimateZ.x"],
                    data["stateEstimateZ.y"],
                    data["stateEstimateZ.z"]]).T * 1e-3
    
    vel = np.array([data["stateEstimateZ.vx"],
                    data["stateEstimateZ.vy"],
                    data["stateEstimateZ.vz"]]).T * 1e-3
    
    acc = np.array([data["stateEstimateZ.ax"],
                    data["stateEstimateZ.ay"],
                    data["stateEstimateZ.az"]]).T * 1e-3
    
    # rot = np.array([data["stateEstimate.qw"],
    #                 data["stateEstimate.qx"],
    #                 data["stateEstimate.qy"],
    #                 data["stateEstimate.qz"]]).T
    rot = rowan.from_euler(data[f"ctrlLee{extra_char}.rpyx"],
                           data[f"ctrlLee{extra_char}.rpyy"],
                           data[f"ctrlLee{extra_char}.rpyz"], "xyz", "extrinsic")
    
    rpm = np.array([data["rpm.m1"],
                    data["rpm.m2"],
                    data["rpm.m3"],
                    data["rpm.m4"]]).T
    
    ang_vel = np.array([
        data[f"ctrlLee{extra_char}.omegax"],
        data[f"ctrlLee{extra_char}.omegay"],
        data[f"ctrlLee{extra_char}.omegaz"]
    ]).T
    
    # pwm = np.array([data_usd["pwm.m1_pwm"],
    #                 data_usd["pwm.m2_pwm"],
    #                 data_usd["pwm.m3_pwm"],
    #                 data_usd["pwm.m4_pwm"]]).T
    # pwm = np.array([
    #     data["powerDist.m1d"],
    #     data["powerDist.m2d"],
    #     data["powerDist.m3d"],
    #     data["powerDist.m4d"]
    # ]).T
    pwm = np.array([
        data["motor.m1"],
        data["motor.m2"],
        data["motor.m3"],
        data["motor.m4"]
    ]).T
    
    a_rpm_f = np.array([
        data[f"ctrlLee{extra_char}.a_rpm_fx"],
        data[f"ctrlLee{extra_char}.a_rpm_fy"],
        data[f"ctrlLee{extra_char}.a_rpm_fz"]
    ]).T
    
    a_imu_f = np.array([
        data[f"ctrlLee{extra_char}.a_imu_fx"],
        data[f"ctrlLee{extra_char}.a_imu_fy"],
        data[f"ctrlLee{extra_char}.a_imu_fz"]
    ]).T
    
    tau_rpm_f = np.array([
        data[f"ctrlLee{extra_char}.tau_rpm_fx"],
        data[f"ctrlLee{extra_char}.tau_rpm_fy"],
        data[f"ctrlLee{extra_char}.tau_rpm_fz"]
    ]).T
    
    tau_imu_f = np.array([
        data[f"ctrlLee{extra_char}.tau_imu_fx"],
        data[f"ctrlLee{extra_char}.tau_imu_fy"],
        data[f"ctrlLee{extra_char}.tau_imu_fz"]
    ]).T
    
    bat = data["pm.vbatMV"]
    
    data_dict = {
        "timestamp": timestamp,
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "rot": rot,
        "ang_vel": ang_vel,
        "rpm": rpm,
        "pwm": pwm,
        "bat": bat,
        "a_rpm": a_rpm_f,
        "a_imu": a_imu_f,
        "tau_rpm": tau_rpm_f,
        "tau_imu": tau_imu_f,
    }
    
    # Payload
    if has_payload:
        p_pos = np.array([
            data["stateEstimateZ.px"],
            data["stateEstimateZ.py"],
            data["stateEstimateZ.pz"]
        ]).T * 1e-3
        
        p_vel = np.array([
            data["stateEstimateZ.pvx"],
            data["stateEstimateZ.pvy"],
            data["stateEstimateZ.pvz"]
        ]).T * 1e-3
        
        p_acc = np.array([
            data["stateEstimateZ.pax"],
            data["stateEstimateZ.pay"],
            data["stateEstimateZ.paz"]
        ]).T * 1e-3
        
        p_dir = np.array([
            data["ctrlLeeP.qix"],
            data["ctrlLeeP.qiy"],
            data["ctrlLeeP.qiz"]
        ]).T
        
        data_dict["p_pos"] = p_pos
        data_dict["p_vel"] = p_vel
        data_dict["p_acc"] = p_acc
        data_dict["p_dir"] = p_dir
        
        data_dict["fpayload_rpm"] = np.array([
            data["ctrlLeeP.fpayload_rpm_fx"],
            data["ctrlLeeP.fpayload_rpm_fy"],
            data["ctrlLeeP.fpayload_rpm_fz"]
        ]).T
        
        data_dict["fpayload_imu"] = np.array([
            data["ctrlLeeP.fpayload_imu_fx"],
            data["ctrlLeeP.fpayload_imu_fy"],
            data["ctrlLeeP.fpayload_imu_fz"]
        ]).T
        
        data_dict["fcable_rpm"] = np.array([
            data["ctrlLeeP.fcable_rpm_fx"],
            data["ctrlLeeP.fcable_rpm_fy"],
            data["ctrlLeeP.fcable_rpm_fz"]
        ]).T
        
        data_dict["fcable_imu"] = np.array([
            data["ctrlLeeP.fcable_imu_fx"],
            data["ctrlLeeP.fcable_imu_fy"],
            data["ctrlLeeP.fcable_imu_fz"]
        ]).T
        
    if "nn_output.f_x" in data:
        data_dict["nn_output.f_x"] = data["nn_output.f_x"]
        data_dict["nn_output.f_y"] = data["nn_output.f_y"]
        data_dict["nn_output.f_z"] = data["nn_output.f_z"]
        data_dict["nn_output.tau_x"] = data["nn_output.tau_x"]
        data_dict["nn_output.tau_y"] = data["nn_output.tau_y"]
        data_dict["nn_output.tau_z"] = data["nn_output.tau_z"]
    
    for i in range(19):
        if f"nn_input.i{i}" in data:
            data_dict[f"nn_input.i{i}"] = data[f"nn_input.i{i}"]
    
    return data_dict


if __name__ == "__main__":
    
    data_dir = "./data/single/train_raw_all"
    target_dir = "./data/single/train_clean"
    
    os.makedirs(target_dir, exist_ok=True)
    
    for i, file_name in enumerate(os.listdir(data_dir)):
        
        file_path = os.path.join(data_dir, file_name)
        try:
            data_usd = cfusdlog.decode(file_path)
        except:
            print(f"Error on file: {file_name}")
            continue
        
        data = data_usd["fixedFrequency"]
        if not i:
            for k in data.keys():
                print(k)
                
        # vels = np.array([
        #     data["stateEstimateZ.vx"],
        #     data["stateEstimateZ.vy"],
        #     data["stateEstimateZ.vz"]
        # ]).T / 1000
        # plt.plot(np.linalg.norm(vels, axis=1))
        # plt.show()
        # if i == 3: exit()
        
        data_clean = transform(data, has_payload=("payload" in data_dir))
        print(f"{i:02}.npy Done.")
        target_path = os.path.join(target_dir, f"{i:02}.npy")
        np.save(target_path, data_clean, allow_pickle=True)
        
"""
Data from the crazyflies always has different labels
and conventions. This script gives a base to read the
files and transform them to the same json format with
the same keys and standards.
"""
import os
import json
import rowan
import numpy as np

from LMCE import cfusdlog


def tranform(data: dict, has_payload: bool=False) -> dict:
    
    timestamp = np.array(data["timestamp"])

    pos = np.array([data["stateEstimate.x"],
                    data["stateEstimate.y"],
                    data["stateEstimate.z"]]).T
    
    # rot = np.array([data["stateEstimate.qw"],
    #                 data["stateEstimate.qx"],
    #                 data["stateEstimate.qy"],
    #                 data["stateEstimate.qz"]]).T
    rot = rowan.from_euler(data["ctrlLee.rpyx"],
                           data["ctrlLee.rpyy"],
                           data["ctrlLee.rpyz"], "xyz", "extrinsic")
    
    vel = np.array([data["stateEstimateZ.vx"],
                    data["stateEstimateZ.vy"],
                    data["stateEstimateZ.vz"]]).T * 1e-3
    
    acc = np.array([data["stateEstimateZ.ax"],
                    data["stateEstimateZ.ay"],
                    data["stateEstimateZ.az"]]).T * 1e-3
    
    rpm = np.array([data["rpm.m1"],
                    data["rpm.m2"],
                    data["rpm.m3"],
                    data["rpm.m4"]]).T
    
    # pwm = np.array([data_usd["pwm.m1_pwm"],
    #                 data_usd["pwm.m2_pwm"],
    #                 data_usd["pwm.m3_pwm"],
    #                 data_usd["pwm.m4_pwm"]]).T
    pwm = np.array([
        data["powerDist.m1d"],
        data["powerDist.m2d"],
        data["powerDist.m3d"],
        data["powerDist.m4d"]
    ]).T
    pwm = np.array([
        data["motor.m1"],
        data["motor.m2"],
        data["motor.m3"],
        data["motor.m4"]
    ]).T
    
    a_rpm_f = np.array([
        data["ctrlLee.a_rpm_fx"],
        data["ctrlLee.a_rpm_fy"],
        data["ctrlLee.a_rpm_fz"]
    ]).T
    
    a_imu_f = np.array([
        data["ctrlLee.a_imu_fx"],
        data["ctrlLee.a_imu_fy"],
        data["ctrlLee.a_imu_fz"]
    ]).T
    
    tau_rpm_f = np.array([
        data["ctrlLee.tau_rpm_fx"],
        data["ctrlLee.tau_rpm_fy"],
        data["ctrlLee.tau_rpm_fz"]
    ]).T
    
    tau_imu_f = np.array([
        data["ctrlLee.tau_imu_fx"],
        data["ctrlLee.tau_imu_fy"],
        data["ctrlLee.tau_imu_fz"]
    ]).T
    
    ang_vel = np.array([
        data["ctrlLee.omegax"],
        data["ctrlLee.omegay"],
        data["ctrlLee.omegaz"]
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
            data["ctrlLeeP.plVelx"],
            data["ctrlLeeP.plVelx"],
            data["ctrlLeeP.plVelx"],
        ]).T
        
        p_acc = np.array([
            data["ctrlLeeP.plAccx_tq"],
            data["ctrlLeeP.plAccy_tq"],
            data["ctrlLeeP.plAccz_tq"]
        ]).T
        
        data_dict["p_pos"] = p_pos
        data_dict["p_vel"] = p_vel
        data_dict["p_acc"] = p_acc
    
    return data_dict


if __name__ == "__main__":
    
    data_dir = "data_old/indi_no_payload"
    target_dir = "data/data_clean"
    
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
        
        data_clean = tranform(data)
        print(f"{i:02}.npy Done.")
        target_path = os.path.join(target_dir, f"{i:02}.npy")
        np.save(target_path, data_clean, allow_pickle=True)
        
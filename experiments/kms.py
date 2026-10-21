import matplotlib.pyplot as plt
from LMCE import cfusdlog


data_usd = cfusdlog.decode("data/kms/lp_40")
data = data_usd["fixedFrequency"]

# plt.plot(data["nn_input.i19"], alpha=.5)
# plt.plot(data["stateEstimateZ.pax"] * 1e-3, alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i20"], alpha=.5)
# plt.plot(data["stateEstimateZ.pay"] * 1e-3, alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i21"], alpha=.5)
# plt.plot(data["stateEstimateZ.paz"] * 1e-3, alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i22"], alpha=.5)
# plt.plot(data["stateEstimateZ.pvx"] * 1e-3, alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i23"], alpha=.5)
# plt.plot(data["stateEstimateZ.pvy"] * 1e-3, alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i24"], alpha=.5)
# plt.plot(data["stateEstimateZ.pvz"] * 1e-3, alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i25"], alpha=.5)
# plt.plot(data["ctrlLeeP.qix"], alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i26"], alpha=.5)
# plt.plot(data["ctrlLeeP.qiy"], alpha=.5)
# plt.show()

# plt.plot(data["nn_input.i27"], alpha=.5, label="nn")
# plt.plot(data["ctrlLeeP.qiz"], alpha=.5, label="real")
# plt.legend()
# plt.show()

plt.plot(data["nn_output.f_x"], alpha=.5, label="nn")
plt.plot(data["ctrlLeeP.a_imu_fx"] - data["ctrlLeeP.a_rpm_fx"], alpha=.5, label="indi")
plt.legend()
plt.show()

plt.plot(data["nn_output.f_y"], alpha=.5, label="nn")
plt.plot(data["ctrlLeeP.a_imu_fy"] - data["ctrlLeeP.a_rpm_fy"], alpha=.5, label="indi")
plt.legend()
plt.show()

plt.plot(data["nn_output.f_z"], alpha=.5, label="nn")
plt.plot(data["ctrlLeeP.a_imu_fz"] - data["ctrlLeeP.a_rpm_fz"], alpha=.5, label="indi")
plt.legend()
plt.show()

plt.plot(data["nn_output.tau_x"], alpha=.5, label="nn")
plt.plot(data["ctrlLeeP.tau_imu_fx"] - data["ctrlLeeP.tau_rpm_fx"], alpha=.5, label="indi")
plt.legend()
plt.show()

plt.plot(data["nn_output.tau_y"], alpha=.5, label="nn")
plt.plot(data["ctrlLeeP.tau_imu_fy"] - data["ctrlLeeP.tau_rpm_fy"], alpha=.5, label="indi")
plt.legend()
plt.show()

plt.plot(data["nn_output.tau_z"], alpha=.5, label="nn")
plt.plot(data["ctrlLeeP.tau_imu_fz"] - data["ctrlLeeP.tau_rpm_fz"], alpha=.5, label="indi")
plt.legend()
plt.show()

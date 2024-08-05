import numpy as np
import matplotlib.pyplot as plt

solar_constant = 1361 * 4 * np.pi  # 太阳常数，单位：W/m^2

# 定义恒星和行星参数
mass_yellow_star = 0.93  # 黄星的质量，单位：太阳质量
mass_red_star = 0.713  # 红星的质量，单位：太阳质量
lum_yellow_star = 0.8  # 黄星的光度，单位：太阳光度
lum_red_star = 0.4  # 红星的光度，单位：太阳光度

# 引力常数
G = 4 * np.pi ** 2  # 天文单位、太阳质量、年为单位

# 轨道参数
a = 1.0  # 半长轴，单位：AU
l = 1.4  # 焦距长度，单位：AU

# 计算双星系统的重心位置
total_mass = mass_yellow_star + mass_red_star
p_y = mass_red_star / total_mass * a
p_r = - mass_yellow_star / total_mass * a

v = np.sqrt(G * total_mass / a)
v_y = v / total_mass * mass_red_star
v_r = - v / total_mass * mass_yellow_star
v_p = np.sqrt(G * total_mass / l)

# 初始条件
# 双星的初始位置和速度（单位：AU和AU/年）
pos_yellow_star = np.array([p_y, 0.0])
pos_red_star = np.array([p_r, 0.0])
vel_yellow_star = np.array([0.0, v_y])
vel_red_star = np.array([0.0, v_r])

# 行星初始位置和速度
pos_planet = np.array([0.0, l])
vel_planet = np.array([v_p, 0.0])

# 时间步长和总时间
dt = 0.001  # 时间步长（单位：年）
total_time = 10  # 总时间（单位：年）
num_steps = int(total_time / dt)
time = np.linspace(0, total_time, num_steps)

# 用于存储轨道数据
positions_yellow_star = np.zeros((num_steps, 2))
positions_red_star = np.zeros((num_steps, 2))
positions_planet = np.zeros((num_steps, 2))
energy_input = np.zeros(num_steps)  # 用于存储行星接收到的能量输入


def compute_acceleration(pos1, pos2, mass):
    r = pos2 - pos1
    r_mag = np.linalg.norm(r)
    return G * mass * r / r_mag ** 3


def compute_energy_input(distance, luminosity):
    return luminosity / (4 * np.pi * distance ** 2) * solar_constant


def compute_surface_temperature(luminosity, radius, albedo=0.3):
    sigma = 5.670374419e-8  # 斯特藩-玻尔兹曼常数, 单位：W/m^2/K^4
    absorbed_luminosity = (1 - albedo) * luminosity  # 考虑反射率后的吸收光度
    area_in = np.pi * (radius ** 2)  # 行星截面积
    area_out = 4 * np.pi * (radius ** 2)  # 行星表面积
    temperature = (absorbed_luminosity * area_in / sigma / area_out) ** 0.25  # 斯特藩-玻尔兹曼定律
    return temperature


# Verlet积分方法计算轨道
for i in range(num_steps):
    positions_yellow_star[i] = pos_yellow_star
    positions_red_star[i] = pos_red_star
    positions_planet[i] = pos_planet

    # 计算加速度
    acc_planet_yellow = compute_acceleration(pos_planet, pos_yellow_star, mass_yellow_star)
    acc_planet_red = compute_acceleration(pos_planet, pos_red_star, mass_red_star)
    acc_yellow_red = compute_acceleration(pos_yellow_star, pos_red_star, mass_red_star)
    acc_red_yellow = compute_acceleration(pos_red_star, pos_yellow_star, mass_yellow_star)

    # Verlet积分更新位置
    pos_yellow_star += vel_yellow_star * dt + 0.5 * acc_yellow_red * dt ** 2
    pos_red_star += vel_red_star * dt + 0.5 * acc_red_yellow * dt ** 2
    pos_planet += vel_planet * dt + 0.5 * (acc_planet_yellow + acc_planet_red) * dt ** 2

    # 计算新的加速度
    new_acc_planet_yellow = compute_acceleration(pos_planet, pos_yellow_star, mass_yellow_star)
    new_acc_planet_red = compute_acceleration(pos_planet, pos_red_star, mass_red_star)
    new_acc_yellow_red = compute_acceleration(pos_yellow_star, pos_red_star, mass_red_star)
    new_acc_red_yellow = compute_acceleration(pos_red_star, pos_yellow_star, mass_yellow_star)

    # Verlet积分更新速度
    vel_yellow_star += 0.5 * (acc_yellow_red + new_acc_yellow_red) * dt
    vel_red_star += 0.5 * (acc_red_yellow + new_acc_red_yellow) * dt
    vel_planet += 0.5 * (acc_planet_yellow + new_acc_planet_yellow + acc_planet_red + new_acc_planet_red) * dt

    # 计算行星接收到的能量输入
    distance_to_yellow_star = np.linalg.norm(pos_planet - pos_yellow_star)
    distance_to_red_star = np.linalg.norm(pos_planet - pos_red_star)
    energy_from_yellow_star = compute_energy_input(distance_to_yellow_star, lum_yellow_star)
    energy_from_red_star = compute_energy_input(distance_to_red_star, lum_red_star)
    energy_input[i] = energy_from_yellow_star + energy_from_red_star

# 可视化
plt.figure(figsize=(12, 12))

# 可视化
plt.figure(figsize=(15, 10))

# 轨道可视化
plt.subplot(3, 2, 1)
plt.plot(positions_yellow_star[:, 0], positions_yellow_star[:, 1], label="Yellow Star Orbit")
plt.plot(positions_red_star[:, 0], positions_red_star[:, 1], label="Red Dwarf Orbit")
plt.plot(positions_planet[:, 0], positions_planet[:, 1], label="Planet Orbit")
plt.xlabel('X (AU)')
plt.ylabel('Y (AU)')
plt.legend()
plt.title('Orbits of Stars and Planet')

# 能量输入可视化
plt.subplot(3, 2, 2)
plt.plot(time, energy_input)
plt.xlabel('Time (years)')
plt.ylabel('Energy Input (arbitrary units)')
plt.title('Energy Input to Planet')

# 行星表面温度（类似地球）可视化
planet_radius = 6371000  # 地球半径，单位：m
surface_temperature_earth = compute_surface_temperature(energy_input, planet_radius, albedo=0.3)
plt.subplot(3, 2, 3)
plt.plot(time, surface_temperature_earth)
plt.xlabel('Time (years)')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature of Earth-like Planet')

# 行星表面温度（超级地球）可视化
planet_radius = 1.1 * 6371000  # 超级地球半径，单位：m
surface_temperature_super_earth = compute_surface_temperature(energy_input, planet_radius, albedo=0.25)
plt.subplot(3, 2, 4)
plt.plot(time, surface_temperature_super_earth)
plt.xlabel('Time (years)')
plt.ylabel('Surface Temperature (K)')
plt.title('Surface Temperature of Super Earth')

# 行星表面温度（类似地球）可视化，考虑 10% 的温室效应和热容导致的 30 天的一个滞后效应（10 个步长）
planet_radius = 6371000  # 地球半径，单位：m
surface_temperature_earth = compute_surface_temperature(energy_input, planet_radius, albedo=0.3)
surface_temperature_earth = surface_temperature_earth * 1.1
# 使用 10 个步长滑动窗口平均温度
surface_temperature_earth_lagged = np.convolve(surface_temperature_earth, np.ones((10,))/10, mode='valid')
time_lagged = np.linspace(0, total_time, len(surface_temperature_earth_lagged))
plt.subplot(3, 2, 5)
plt.plot(time_lagged, surface_temperature_earth_lagged)
plt.xlabel('Time (years)')
plt.ylabel('Surface Temperature (K) with seasonal lag')
plt.title('Surface Temperature of Earth-like Planet')

# 行星表面温度（超级地球）可视化，考虑 10% 的温室效应和热容导致的 45 天的一个滞后效应（15 个步长）
planet_radius = 1.1 * 6371000  # 超级地球半径，单位：m
surface_temperature_super_earth = compute_surface_temperature(energy_input, planet_radius, albedo=0.25)
surface_temperature_super_earth = surface_temperature_super_earth * 1.1
surface_temperature_super_earth_lagged = np.convolve(surface_temperature_super_earth, np.ones((15,))/15, mode='valid')
time_lagged_super_earth = np.linspace(0, total_time, len(surface_temperature_super_earth_lagged))
plt.subplot(3, 2, 6)
plt.plot(time_lagged_super_earth, surface_temperature_super_earth_lagged)
plt.xlabel('Time (years)')
plt.ylabel('Surface Temperature (K) with seasonal lag')
plt.title('Surface Temperature of Super Earth')

plt.tight_layout()
plt.show()

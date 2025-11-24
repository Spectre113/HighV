from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
import csv

from regulators.pid_controller import PDController
from baseline_detection import MultiCameralineDetector
from post_processing import (
    compute_turns_vmax_amax,
    generate_new_trajectory_with_hermite,
    normalize_coordinates,
    plot_new_vs_original,
    plot_trajectory,
    print_turns_vmax_summary,
    smooth_coordinates,
    smooth_trajectory_with_window,
    calculate_curvature_global_spline,
    analyze_turns,
    add_scaled_radius_and_intersections,
    merge_overlapping_scaled_circles,
    plot_turns_on_trajectory,
    smooth_full_trajectory_global_spline,
)


robot = Robot()
timestep = 32

# Движение
left_motor = robot.getDevice("left_rear_wheel")
right_motor = robot.getDevice("right_rear_wheel")
left_steer = robot.getDevice("left_steer")
right_steer = robot.getDevice("right_steer")

# Сенсоры
gps = robot.getDevice("gps")
imu = robot.getDevice("inertial unit")
gyro = robot.getDevice("gyro")
lidar = robot.getDevice("lidar_on")

left_camera = robot.getDevice("left")
central_camera = robot.getDevice("central")
right_camera = robot.getDevice("right")

# Датчики вращения колёс
left_rear_sensor = robot.getDevice("left_rear_sensor")
right_rear_sensor = robot.getDevice("right_rear_sensor")
left_steer_sensor = robot.getDevice("left_steer_sensor")
right_steer_sensor = robot.getDevice("right_steer_sensor")

# ВКЛЮЧЕНИЕ СЕНСОРОВ
gps.enable(timestep)
imu.enable(timestep)
gyro.enable(timestep)
lidar.enable(timestep)

left_camera.enable(timestep)
central_camera.enable(timestep)
right_camera.enable(timestep)

left_rear_sensor.enable(timestep)
right_rear_sensor.enable(timestep)
left_steer_sensor.enable(timestep)
right_steer_sensor.enable(timestep)

# === НАСТРОЙКА МОТОРОВ ===
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# Setup visualization

width = central_camera.getWidth()
height = central_camera.getHeight()

cameras = {"left": left_camera, "central": central_camera, "right": right_camera}

speed = 10.0
angle = 0.0
err = 0.0

kp = 0.1
kd = 0.1
controller = PDController(MultiCameralineDetector(), kp=kp, kd=kd)
controller.start_processing()

positions = []
yaw_rates = []
interval = 0.2
log = 0
threshold = 0.25


def moving_max(data, window_size):
    max_vals = []
    half_w = window_size // 2
    n = len(data)
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        window = data[start:end]
        max_vals.append(max(window))
    return max_vals


# === ГЛАВНЫЙ ЦИКЛ ===
try:
    while robot.step(timestep) != -1:
        current_time = robot.getTime()

        # ----- GPS -----
        pos = gps.getValues()  # [x, y, z]
        print(f"GPS: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

        # ----- IMU -----
        roll, pitch, yaw = imu.getRollPitchYaw()
        print(f"IMU: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")

        # ----- Гироскоп -----
        gyro_values = gyro.getValues()
        print(
            f"Gyro: x={gyro_values[0]:.3f}, y={gyro_values[1]:.3f}, z={gyro_values[2]:.3f}"
        )

        if current_time - log >= interval:
            x = pos[0]
            y = pos[1]

            positions.append((x, y))
            yaw_rate = gyro.getValues()[2]
            yaw_rates.append(yaw_rate)
            print(f"Saved position: {x:.3f}, {y:.3f} at t = {current_time:.2f}")

            log = current_time
        else:
            print("Curr time: ", current_time)

        # ----- Лидар -----
        ranges = lidar.getRangeImage()
        print(f"Lidar (первые 5): {[round(r, 2) for r in ranges[:5]]}")

        # ----- Датчики вращения -----
        print(f"Left rear wheel: {left_rear_sensor.getValue():.3f}")
        print(f"Right rear wheel: {right_rear_sensor.getValue():.3f}")
        print(f"Left steer angle: {left_steer_sensor.getValue():.3f}")
        print(f"Right steer angle: {right_steer_sensor.getValue():.3f}")

        print("-" * 60)

        # ----- Управление движением -----

        controller.update_sensor(cameras, angle)

        output, err, state = controller.get_latest_result()

        if state and err:
            # img_display.set_data(output

            # Pull towards the line if too close to the edge of camera frame
            counter = 3
            if np.abs(err) > 0.7 and counter > 0:
                counter -= 1
                controller.kp = 0.3
                controller.kd = 0.1
            else:
                counter = 3

        angle = controller.get_current_command()

        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        left_steer.setPosition(angle)
        right_steer.setPosition(angle)
except KeyboardInterrupt:
    pass

xs = [p[0] for p in positions]
ys = [p[1] for p in positions]

xs_norm, ys_norm, min_x, max_x, min_y, max_y = normalize_coordinates(xs, ys)

plot_trajectory(xs_norm, ys_norm)

xs_smooth, ys_smooth = smooth_coordinates(xs, ys, window_length=15, polyorder=4)
smooth_xs_local, smooth_ys_local = smooth_trajectory_with_window(
    xs_smooth, ys_smooth, window_size=3, num_interp_per_segment=100
)

smooth_xs, smooth_ys = smooth_full_trajectory_global_spline(
    smooth_xs_local, smooth_ys_local, num_points=4 * len(smooth_xs_local)
)

curvature = calculate_curvature_global_spline(smooth_xs, smooth_ys)

turns_info, s = analyze_turns(
    smooth_xs, smooth_ys, curvature, threshold=0.08, min_length=18
)

turns_info = merge_overlapping_scaled_circles(turns_info, smooth_xs, smooth_ys)

plot_turns_on_trajectory(smooth_xs, smooth_ys, turns_info)

turns_info = compute_turns_vmax_amax(
    turns_info, mass=1900.0, mu=1.0, g=9.81, downforce=0.0
)
print_turns_vmax_summary(turns_info)

new_xs, new_ys = generate_new_trajectory_with_hermite(
    smooth_xs,
    smooth_ys,
    turns_info,
    samples_per_segment=300,
    blend_points=0,
)

trajectory_file = "trajectory_new.csv"
np.savetxt(
    trajectory_file,
    np.column_stack((new_xs, new_ys)),
    delimiter=",",
    header="x,y",
    comments="",
)
print(f"Saved new trajectory to {trajectory_file}")

turn_points_file = "turn_points_with_limits.csv"
with open(turn_points_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["turn_id", "point_type", "index", "x", "y", "a_max", "v_max"])

    n_points = len(smooth_xs)

    for i, t in enumerate(turns_info):
        for point_type, idx_key, a_key, v_key in [
            ("entry", "entry_idx", "entry_a_max", "entry_v_max"),
            ("apex", "apex_idx", "apex_a_max", "apex_v_max"),
            ("exit", "exit_idx", "exit_a_max", "exit_v_max"),
        ]:
            idx = t.get(idx_key, None)
            if idx is None:
                continue
            if not (0 <= idx < n_points):
                continue

            x = float(smooth_xs[idx])
            y = float(smooth_ys[idx])
            a_val = t.get(a_key, None)
            v_val = t.get(v_key, None)

            writer.writerow([i + 1, point_type, int(idx), x, y, a_val, v_val])

print(f"Saved turn points and limits to {turn_points_file}")

plot_new_vs_original(smooth_xs, smooth_ys, new_xs, new_ys)

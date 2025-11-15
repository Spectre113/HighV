from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
# from follow_model_line import LineFollowerController, LineFollowerController2
from pid_controller import PDController
from baseline_detection import MultiCameralineDetector
from scipy.interpolate import CubicSpline


robot = Robot()
timestep = 32

# Движение
left_motor = robot.getDevice('left_rear_wheel')
right_motor = robot.getDevice('right_rear_wheel')
left_steer = robot.getDevice('left_steer')
right_steer = robot.getDevice('right_steer')

# Сенсоры
gps = robot.getDevice('gps')
imu = robot.getDevice('inertial unit')
gyro = robot.getDevice('gyro')
lidar = robot.getDevice('lidar_on')

left_camera = robot.getDevice('left')
central_camera = robot.getDevice('central')
right_camera = robot.getDevice('right')

# Датчики вращения колёс
left_rear_sensor = robot.getDevice('left_rear_sensor')
right_rear_sensor = robot.getDevice('right_rear_sensor')
left_steer_sensor = robot.getDevice('left_steer_sensor')
right_steer_sensor = robot.getDevice('right_steer_sensor')

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
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# Setup visualization

width = central_camera.getWidth()
height = central_camera.getHeight()

cameras = {
    'left': left_camera,
    'central': central_camera,
    'right': right_camera
}

speed = 13
angle = 0.0
err = 0.0

kp = 0.1
kd = 0.01
controller = PDController(MultiCameralineDetector(), kp=kp, kd=kd)
controller.start_processing()

positions = []
yaw_rates = []
interval = 0.5
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
        print(f"Gyro: x={gyro_values[0]:.3f}, y={gyro_values[1]:.3f}, z={gyro_values[2]:.3f}")


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
                controller.kp = 0.008
                controller.kd = 0.01
            else:
                counter = 3

        angle = controller.get_current_command()

        # 2nd order control
        # line_angle_rad = -0.05
        # error_rad = left_steer_sensor.getValue() - line_angle_rad
        # acceleration = LineFollowerController2().compute_steering(velocity, error_rad)

        # dt = timestep / 1000.0
        # velocity = velocity + acceleration * dt
        # angle = left_steer_sensor.getValue() + velocity * dt

        # 1st order control
        # velocity = LineFollowerController().compute_steering(error_rad)

        # dt = timestep / 1000.0
        # angle = left_steer_sensor.getValue() + velocity * dt

        # angle = np.clip(angle, -1, 1)

        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        left_steer.setPosition(angle)
        right_steer.setPosition(angle)
except KeyboardInterrupt:
    pass


threshold = 0.1
window_size = 4

abs_yaw_rates = [abs(y) for y in yaw_rates]
smoothed_max = moving_max(abs_yaw_rates, window_size)

turn_peaks = []
for i in range(1, len(smoothed_max) - 1):
    if smoothed_max[i] > smoothed_max[i-1] and smoothed_max[i] >= smoothed_max[i+1] and smoothed_max[i] > threshold:
        turn_peaks.append(i)

turn_mask = np.array([val > threshold for val in abs_yaw_rates])

turn_starts = []
turn_ends = []

for i in range(1, len(turn_mask)):
    if turn_mask[i] and not turn_mask[i-1]:
        turn_starts.append(i)
    if not turn_mask[i] and turn_mask[i-1]:
        turn_ends.append(i)

if len(turn_mask) > 0 and turn_mask[-1]:
    turn_ends.append(len(turn_mask) - 1)

xs = [p[0] for p in positions]
ys = [p[1] for p in positions]

min_x = min(xs)
max_x = max(xs)
min_y = min(ys)
max_y = max(ys)

range_x = max_x - min_x if max_x != min_x else 1
range_y = max_y - min_y if max_y != min_y else 1

xs_norm = [(x - min_x) / range_x for x in xs]
ys_norm = [(y - min_y) / range_y for y in ys]

print("Green points (Turn Peaks) coordinates (normalized):")
for i in turn_peaks:
    print(f"Index: {i}, X: {xs[i]:.4f}, Y: {ys[i]:.4f}")


plt.figure(figsize=(8, 8))
plt.plot(xs_norm, ys_norm, marker='o', color='blue', linewidth=1, label='Trajectory')

for start, end in zip(turn_starts, turn_ends):
    plt.plot(xs_norm[start:end+1], ys_norm[start:end+1], marker='o', linestyle='', color='yellow', alpha=0.7)

plt.scatter([xs_norm[i] for i in turn_starts], [ys_norm[i] for i in turn_starts], color='red', s=60, label='Turn Start')
plt.scatter([xs_norm[i] for i in turn_ends], [ys_norm[i] for i in turn_ends], color='red', s=60, label='Turn End')

plt.scatter([xs_norm[i] for i in turn_peaks], [ys_norm[i] for i in turn_peaks], color='green', s=150, label='Turn Peaks')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectory with Turns Highlighted and Peaks")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.savefig("trajectory_with_turns_and_peaks.png", dpi=200)
print("Saved trajectory_with_turns_and_peaks.png")
plt.show()

xs = np.array([p[0] for p in positions])
ys = np.array([p[1] for p in positions])

peak_indices = np.array(turn_peaks)

peak_xs = xs[peak_indices]
peak_ys = ys[peak_indices]

cs_x = CubicSpline(peak_indices, peak_xs)
cs_y = CubicSpline(peak_indices, peak_ys)

interp_indices = np.linspace(peak_indices[0], peak_indices[-1], 500)
smooth_xs = cs_x(interp_indices)
smooth_ys = cs_y(interp_indices)

min_x, max_x = np.min(xs), np.max(xs)
min_y, max_y = np.min(ys), np.max(ys)
range_x = max_x - min_x if max_x != min_x else 1
range_y = max_y - min_y if max_y != min_y else 1

xs_norm = (xs - min_x) / range_x
ys_norm = (ys - min_y) / range_y

smooth_xs_norm = (smooth_xs - min_x) / range_x
smooth_ys_norm = (smooth_ys - min_y) / range_y

plt.figure(figsize=(10, 8))

plt.plot(xs_norm, ys_norm, marker='o', color='blue', linewidth=1, label='Original Trajectory')

plt.scatter(xs_norm[peak_indices], ys_norm[peak_indices], color='green', s=100, label='Turn Peaks')

plt.plot(smooth_xs_norm, smooth_ys_norm, color='red', linewidth=2, label='Smoothed Trajectory (Cubic Spline)')

plt.xlabel("Normalized X")
plt.ylabel("Normalized Y")
plt.title("Trajectory Smoothing Through All Turn Peaks")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
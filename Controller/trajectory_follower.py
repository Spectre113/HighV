from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline


robot = Robot()
timestep = 32  # ms
MAX_VELOCITY = 50.0  # m/s
RADIUS = 0.36  # m
MAX_SPEED = int(MAX_VELOCITY / RADIUS)  # rad / s

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

left_rear_sensor.enable(timestep)
right_rear_sensor.enable(timestep)
left_steer_sensor.enable(timestep)
right_steer_sensor.enable(timestep)

# === НАСТРОЙКА МОТОРОВ ===
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


def smooth_and_interpolate_trajectory(
    x_raw: np.ndarray, y_raw: np.ndarray, N: int
) -> tuple:
    """
    Smooth raw trajectory points using Savitzky-Golay filter and interpolate to N points.

    Args:
        x_raw: Raw x coordinates of path points
        y_raw: Raw y coordinates of path points
        N: Number of points for output trajectory

    Returns:
        tuple: (x_smooth, y_smooth) as numpy arrays of length N
    """

    # Apply Savitzky-Golay filter for smoothing
    window_length = min(21, len(x_raw) - 1)
    if window_length % 2 == 0:
        window_length -= 1

    x_smooth_raw = savgol_filter(x_raw, window_length=window_length, polyorder=3)
    y_smooth_raw = savgol_filter(y_raw, window_length=window_length, polyorder=3)

    # Create parameter (arc length) for interpolation
    dx = np.diff(x_smooth_raw)
    dy = np.diff(y_smooth_raw)
    distances = np.sqrt(dx**2 + dy**2)
    s_raw = np.concatenate(([0], np.cumsum(distances)))

    # Create cubic splines for smooth interpolation
    x_spline = CubicSpline(s_raw, x_smooth_raw)
    y_spline = CubicSpline(s_raw, y_smooth_raw)

    # Generate evenly spaced arc length points
    s_smooth = np.linspace(0, s_raw[-1], N)

    # Interpolate to get smooth trajectory
    x_smooth = x_spline(s_smooth)
    y_smooth = y_spline(s_smooth)

    return x_smooth, y_smooth


def compute_steering_angles(
    x_traj: np.ndarray, y_traj: np.ndarray, wheelbase: float = 2.94
) -> np.ndarray:
    """
    Compute steering angles for each trajectory point using curvature.

    Args:
        x_traj: X coordinates of trajectory points
        y_traj: Y coordinates of trajectory points
        wheelbase: Vehicle wheelbase (meters)

    Returns:
        steering_angles: Array of steering angles in radians
    """
    # Compute first derivatives (tangent vectors)
    dx = np.gradient(x_traj)
    dy = np.gradient(y_traj)

    # Compute second derivatives
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Compute curvature: κ = (x'y'' - y'x'') / (x'² + y'²)^(3/2)
    numerator = dx * d2y - dy * d2x
    denominator = (dx**2 + dy**2) ** 1.5
    curvature = numerator / (denominator + 1e-6)  # Avoid division by zero

    # Compute steering angles using bicycle model: δ = arctan(κ * L)
    steering_angles = -np.arctan(curvature * wheelbase)

    # Smooth steering angles for realistic vehicle behavior
    window_length = min(15, len(steering_angles) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length >= 3:
        steering_angles = savgol_filter(
            steering_angles, window_length=window_length, polyorder=2
        )

    return steering_angles


xs = None
ys = None
steering_angles = None

with open("trajectory_new.csv", "r") as f:
    pos_reader = csv.DictReader(f)
    xs = []
    ys = []
    for row in pos_reader:
        xs.append(float(row["x"]))
        ys.append(float(row["y"]))

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # Smooth and interpolate trajectory
    # xs, ys = smooth_and_interpolate_trajectory(xs, ys, 1000)

    # Compute steering angles for the trajectory
    steering_angles = compute_steering_angles(xs, ys, wheelbase=2.94)

x_turn = None
y_turn = None
v_turn = None
a_turn = None
turn_info = None
# turn_ids = Nones

with open("turn_points_with_limits.csv", "r") as f:
    turn_reader = csv.DictReader(f)
    x_turn = []
    y_turn = []
    v_turn = []
    a_turn = []
    turn_info = []
    # turn_ids = []

    for row in turn_reader:
        turn_id = int(row["turn_id"])
        point_type = row["point_type"]
        x = float(row["x"])
        y = float(row["y"])
        a_max = float(row["a_max"])
        v_max = float(row["v_max"])

        x_turn.append(x)
        y_turn.append(y)
        v_turn.append(v_max)
        a_turn.append(a_max)
        turn_info.append(point_type)

        # turn_ids.append(turn_id)

        # if turn_id == len(turn_info):
        #     turn_info.append(dict())
        # turn_info[-1][point_type] = [a_max, v_max]


def find_closest_point(
    current_x: float,
    current_y: float,
    xs: np.ndarray,
    ys: np.ndarray,
    last_index: int = 0,
    search_window: int = 2000,
) -> int:
    """
    Find the closest trajectory point to current position with directional bias.

    Args:
        current_x: Current x position
        current_y: Current y position
        xs: Trajectory x coordinates
        ys: Trajectory y coordinates
        last_index: Previous closest index to start search from
        search_window: Number of points to search ahead/behind

    Returns:
        closest_index: Index of closest trajectory point
    """
    # Search around last index to maintain direction
    start_idx = max(0, last_index - search_window // 2)
    end_idx = min(len(xs), last_index + search_window)

    min_distance = float("inf")
    closest_index = last_index

    for i in range(start_idx, end_idx):
        distance = np.sqrt((current_x - xs[i]) ** 2 + (current_y - ys[i]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index


def compute_weighted_cross_track_error(
    current_x: float,
    current_y: float,
    xs: np.ndarray,
    ys: np.ndarray,
    current_idx: int,
    lookahead_window: int = 50,
    discount_factor: float = 0.8,
) -> float:
    """
    Compute weighted average cross-track error with distance-based discounting.

    Args:
        current_x, current_y: Current vehicle position
        xs, ys: Path coordinates
        current_idx: Current closest path index
        lookahead_window: Number of points to consider ahead
        discount_factor: Weight reduction per point (0-1, higher = more discount)

    Returns:
        weighted_error: Weighted average cross-track error
    """

    # Define search window
    start_idx = current_idx + lookahead_window // 2
    end_idx = min(len(xs), current_idx + lookahead_window + 1)

    total_weight = 0.0
    weighted_sum = 0.0

    for i in range(start_idx, end_idx - 1):
        # Current path segment
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i + 1], ys[i + 1]

        target_yaw = np.arctan2(
            y2 - y1,
            x2 - x1,
        )

        signed_error = compute_cross_track_error(
            current_x, current_y, x1, y1, target_yaw
        )

        # Weight: discount based on how far ahead this point is
        distance_ahead = i - current_idx
        weight = discount_factor**distance_ahead

        weighted_sum += signed_error * weight
        total_weight += weight

    # Return weighted average
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def compute_cross_track_error(
    current_x: float,
    current_y: float,
    target_x: float,
    target_y: float,
    target_yaw: float,
) -> float:
    """
    Compute signed cross-track error (lateral distance to path).

    Args:
        current_x, current_y: Current vehicle position
        target_x, target_y: Target point on path
        target_yaw: Path tangent angle at target point (radians)

    Returns:
        cross_track_error: Signed lateral distance (positive if vehicle is to the left of path)
    """
    # Vector from target point to vehicle
    dx = current_x - target_x
    dy = current_y - target_y

    # Path normal vector (perpendicular to path direction, pointing left)
    path_normal_x = -np.sin(target_yaw)
    path_normal_y = np.cos(target_yaw)

    # Cross-track error = projection of (dx,dy) onto path normal
    cross_track_error = dx * path_normal_x + dy * path_normal_y

    return cross_track_error


def lin2ang(linear: float, radius: float = 0.36) -> float:
    """Convert linear velocity to angular velociry of rear wheels"""
    return linear / radius


def ang2lin(angular: float, radius: float = 0.36) -> float:
    """Convert angular velocity to linear velocity of rear wheels"""
    return angular / radius


last_closest_index_position = 0
last_closest_index_velocity = -1

dt = timestep / 1000.0

xs_new = []
ys_new = []
cross_track_errors = []
vel_errors = []
real_velocities = []
velocities = []

last_wheel_angle = 0.0

last_cross_track_error = 0.0
cross_track_error_int = 0.0

last_vel_error = 0.0
vel_error_int = 0.0

# === ГЛАВНЫЙ ЦИКЛ ===
try:
    while robot.step(timestep) != -1:
        current_time = robot.getTime()

        # ----- GPS -----
        pos = gps.getValues()  # [x, y, z]
        current_x, current_y = pos[0], pos[1]  # Using x and z coordinates
        print(f"GPS: x={current_x:.2f}, y={current_y:.2f}")

        xs_new.append(current_x)
        ys_new.append(current_y)

        # ----- IMU -----
        roll, pitch, yaw = imu.getRollPitchYaw()
        print(f"IMU: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")

        # ----- Гироскоп -----
        gyro_values = gyro.getValues()
        print(
            f"Gyro: x={gyro_values[0]:.3f}, y={gyro_values[1]:.3f}, z={gyro_values[2]:.3f}"
        )

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

        # Find closest trajectory point

        closest_index_position = find_closest_point(
            current_x, current_y, xs, ys, last_closest_index_position
        )
        last_closest_index_position = closest_index_position

        # Find closest velocity point
        closest_index_velocity = find_closest_point(
            current_x, current_y, x_turn, y_turn, last_closest_index_velocity
        )
        last_closest_index_velocity = closest_index_velocity

        # Cross track error
        # Compute path heading at target point
        if closest_index_position < len(xs) - 1:
            target_yaw = np.arctan2(
                ys[closest_index_position + 1] - ys[closest_index_position],
                xs[closest_index_position + 1] - xs[closest_index_position],
            )
        else:
            target_yaw = np.arctan2(
                ys[closest_index_position] - ys[closest_index_position - 1],
                xs[closest_index_position] - xs[closest_index_position - 1],
            )

        cross_track_error = compute_weighted_cross_track_error(
            current_x, current_y, xs, ys, last_closest_index_position, 50
        )

        cross_track_errors.append(cross_track_error)

        cross_track_error = np.clip(cross_track_error, -1.0, 1.0)

        if np.abs(cross_track_error) == 1.0:
            red_pnt = closest_index_position

        # cross_track_error = 0.0 if np.abs(cross_track_error) < 0.9 else cross_track_error

        velocity_ref = v_turn[closest_index_velocity]
        velocities.append(velocity_ref)

        current_wheel_angle = left_rear_sensor.getValue()
        current_speed = (current_wheel_angle - last_wheel_angle) / dt
        current_velocity = current_speed * RADIUS
        last_wheel_angle = current_wheel_angle

        real_velocities.append(current_velocity)

        velocity_error = velocity_ref - current_velocity

        vel_errors.append(velocity_error)

        # velocity_error = 0.0 if np.abs(velocity_error) < 0.5 else velocity_error

        # Get corresponding steering angle and velocity

        kp_ct = 0.1
        kd_ct = 0.01
        ki_ct = 0.01

        kp_v = 0.1
        kd_v = 0.01
        ki_v = 0.8

        cross_track_error_der = (cross_track_error - last_cross_track_error) / dt

        cross_track_error_int += cross_track_error * dt

        vel_error_der = (velocity_error - last_vel_error) / dt
        vel_error_int += velocity_error * dt

        if closest_index_position == len(xs) - 1:
            angle = 0.0
            speed = 0.0
        else:
            angle = (
                steering_angles[closest_index_position]
                + kp_ct * cross_track_error
                + kd_ct * cross_track_error_der
                + ki_ct * cross_track_error_int
            )
            speed = (
                lin2ang(velocity_ref)
                + kp_v * velocity_error
                + kd_v * vel_error_der
                + ki_v * vel_error_int
            )

        print(
            f"Closest point: {closest_index_position}/{len(xs)}, cross-track error: {cross_track_error}, velocity error: {velocity_error}"
        )

        last_cross_track_error = cross_track_error
        last_vel_error = velocity_error

        speed = np.clip(speed, -MAX_SPEED, MAX_SPEED)
        angle = np.clip(angle, -1, 1)

        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
        left_steer.setPosition(angle)
        right_steer.setPosition(angle)

        # Reset if we reach the end of trajectory
        if closest_index_position >= len(xs):
            last_closest_index_position = 0
        if closest_index_velocity >= len(xs):
            last_closest_index_velocity = 0

except KeyboardInterrupt:
    pass


plt.figure(figsize=(8, 8))
plt.plot(xs, ys, linestyle="--", color="red", linewidth=1, label="Reference")
plt.plot(xs_new, ys_new, color="blue", label="Real")

plt.scatter(x_turn, y_turn, marker="x", label="turn")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectory")
plt.grid(True)
plt.axis("equal")
plt.legend()

plt.figure(figsize=(8, 8))
plt.plot(np.arange(0, len(cross_track_errors)), cross_track_errors, label="cross-track")
plt.plot(np.arange(0, len(vel_errors)), vel_errors, label="velocity")
plt.xlabel("t")
plt.ylabel("Errors")
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 8))
plt.plot(
    np.arange(0, len(velocities)),
    velocities,
    linestyle="--",
    color="red",
    linewidth=1,
    label="Reference",
)
plt.plot(
    np.arange(0, len(real_velocities)), real_velocities, color="Blue", label="Real"
)
plt.xlabel("t")
plt.ylabel("Velocity profile")
plt.grid(True)
plt.legend()

plt.show()

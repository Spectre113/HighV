from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree


robot = Robot()
timestep = 32  # ms
MAX_VELOCITY = 50.0  # m/s
RADIUS = 0.36  # m
MAX_SPEED = int(MAX_VELOCITY / RADIUS)  # rad / s

ACCELERATION_VELOCITY = 45.0

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


def init_kd_tree(xs: np.ndarray, ys: np.ndarray) -> KDTree:
    """
    Initialize KD-tree with trajectory points.

    Args:
        xs: X coordinates of trajectory points
        ys: Y coordinates of trajectory points

    Returns:
        KDTree object for nearest neighbor searches
    """
    points = np.column_stack([xs, ys])
    return KDTree(points)


def find_closest_point(tree: KDTree, current_x: float, current_y: float) -> int:
    """
    Find the closest trajectory point using KD-tree.

    Args:
        tree: Pre-initialized KDTree
        current_x: Current x position
        current_y: Current y position

    Returns:
        Index of closest trajectory point
    """
    distance, index = tree.query([current_x, current_y])
    return index


def find_closest_velocity(s_fine: np.ndarray, path: float) -> int:
    """
    Find the closest point in s_fine to the given path coordinate using binary search.
    More efficient for large arrays.
    """
    if len(s_fine) == 0:
        return -1

    # Handle boundaries
    if path <= s_fine[0]:
        return 0
    if path >= s_fine[-1]:
        return len(s_fine) - 1

    # Use binary search to find the insertion point
    left = 0
    right = len(s_fine) - 1

    while left <= right:
        mid = (left + right) // 2
        if s_fine[mid] < path:
            left = mid + 1
        else:
            right = mid - 1

    # Now left is the insertion point, check left and left-1 for closest
    if left == 0:
        return 0
    if left == len(s_fine):
        return len(s_fine) - 1

    # Check which of the two candidates is closer
    if abs(s_fine[left] - path) < abs(s_fine[left - 1] - path):
        return left
    else:
        return left - 1


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


def create_velocity_bezier_interpolator(
    start_x: float,
    start_y: float,
    x_turn: list,
    y_turn: list,
    v_turn: list,
    num_points: int = 1000,
    max_segment_length: float = 15.0,
    intermediate_velocity: float = 20.0,
    tension: float = 0.3,
):
    """
    Creates a Bezier interpolation function for velocities along the path.
    Path coordinate s=0 corresponds to the vehicle's current position.

    Args:
        start_x, start_y: Current vehicle position (s=0)
        x_turn, y_turn: Coordinates of velocity points
        v_turn: Velocity values at each point
        num_points: Number of points in the output interpolation
        max_segment_length: Maximum allowed distance between consecutive points
        intermediate_velocity: Velocity value for added intermediate points
        tension: Controls smoothness (0.0 = linear, 0.5 = very smooth, 1.0 = sharp)

    Returns:
        Tuple: (s_fine, velocity_fine, s_orig, s_new, x_enhanced, y_enhanced, v_enhanced)
        - s_orig: Path coordinates of ORIGINAL v_turn points
        - s_new: Path coordinates of ADDED acceleration points
    """
    # Create enhanced point lists starting from vehicle position
    x_enhanced = [start_x]
    y_enhanced = [start_y]

    # Lists to track which points are original and which are new
    point_types = ["start"]  # 'start', 'original', 'new'

    # Determine initial velocity
    if len(x_turn) > 0:
        distance_to_first = np.linalg.norm([x_turn[0] - start_x, y_turn[0] - start_y])
        if distance_to_first > max_segment_length:
            v_enhanced = [intermediate_velocity]
        else:
            v_enhanced = [v_turn[0]]
    else:
        v_enhanced = [intermediate_velocity]

    # Store path coordinates for original and new points
    s_orig = []  # Path coordinates of original v_turn points
    s_new = []  # Path coordinates of added acceleration points

    # Add turn points with intermediate points if needed
    for i in range(len(x_turn)):
        if i == 0:
            prev_x, prev_y = start_x, start_y
            prev_v = v_enhanced[0]
        else:
            prev_x, prev_y = x_turn[i - 1], y_turn[i - 1]
            prev_v = v_turn[i - 1]

        distance = np.linalg.norm([x_turn[i] - prev_x, y_turn[i] - prev_y])

        # Only add intermediate points if segment is not too long
        if distance > max_segment_length and distance <= 2 * max_segment_length:
            # Add ONE intermediate point in the middle (not multiple)
            mid_x = (prev_x + x_turn[i]) / 2
            mid_y = (prev_y + y_turn[i]) / 2

            x_enhanced.append(mid_x)
            y_enhanced.append(mid_y)
            v_enhanced.append(intermediate_velocity)
            point_types.append("new")
            print(f"Added intermediate point at ({mid_x:.1f}, {mid_y:.1f})")

        # Add the original turn point
        x_enhanced.append(x_turn[i])
        y_enhanced.append(y_turn[i])
        v_enhanced.append(v_turn[i])
        point_types.append("original")

    # Calculate cumulative path distances s starting from vehicle position (s=0)
    s_enhanced = np.zeros(len(x_enhanced))
    for i in range(1, len(x_enhanced)):
        s_enhanced[i] = s_enhanced[i - 1] + np.linalg.norm(
            [x_enhanced[i] - x_enhanced[i - 1], y_enhanced[i] - y_enhanced[i - 1]]
        )

    # Extract s_orig and s_new from enhanced points
    s_orig = []
    s_new = []

    for i, point_type in enumerate(point_types):
        if point_type == "original":
            s_orig.append(s_enhanced[i])
        elif point_type == "new":
            s_new.append(s_enhanced[i])

    # Generate fine s values for interpolation
    s_fine = np.linspace(0, s_enhanced[-1], num_points + 1)
    velocity_fine = np.zeros(num_points + 1)

    # Perform Bezier interpolation between segments
    for seg in range(len(s_enhanced) - 1):
        # Get current and neighboring points for better control points
        v_prev = v_enhanced[max(0, seg - 1)]
        v_curr = v_enhanced[seg]
        v_next = v_enhanced[seg + 1]
        v_next2 = v_enhanced[min(len(v_enhanced) - 1, seg + 2)]

        segment_length = s_enhanced[seg + 1] - s_enhanced[seg]

        # Improved control point calculation with tension
        if seg == 0:
            # First segment - start from current position
            P0 = (s_enhanced[seg], v_curr)
            P1 = (s_enhanced[seg] + segment_length * tension, v_curr)
            P2 = (s_enhanced[seg] + segment_length * (1 - tension), v_next)
            P3 = (s_enhanced[seg + 1], v_next)
        elif seg == len(s_enhanced) - 2:
            # Last segment
            P0 = (s_enhanced[seg], v_curr)
            P1 = (s_enhanced[seg] + segment_length * tension, v_curr)
            P2 = (s_enhanced[seg] + segment_length * (1 - tension), v_next)
            P3 = (s_enhanced[seg + 1], v_next)
        else:
            # Middle segments
            tangent_in = (v_curr - v_prev) * tension
            tangent_out = (v_next2 - v_next) * tension

            P0 = (s_enhanced[seg], v_curr)
            P1 = (s_enhanced[seg] + segment_length * 0.3, v_curr + tangent_in * 0.3)
            P2 = (s_enhanced[seg] + segment_length * 0.7, v_next - tangent_out * 0.3)
            P3 = (s_enhanced[seg + 1], v_next)

        # Find indices in this segment
        seg_indices = np.where(
            (s_fine >= s_enhanced[seg]) & (s_fine <= s_enhanced[seg + 1])
        )[0]

        for idx in seg_indices:
            t = (s_fine[idx] - s_enhanced[seg]) / (
                s_enhanced[seg + 1] - s_enhanced[seg]
            )
            # Cubic Bezier formula
            velocity_fine[idx] = (
                (1 - t) ** 3 * P0[1]
                + 3 * (1 - t) ** 2 * t * P1[1]
                + 3 * (1 - t) * t**2 * P2[1]
                + t**3 * P3[1]
            )

    return s_fine, velocity_fine, s_orig, s_new, x_enhanced, y_enhanced, v_enhanced


def lin2ang(linear: float, radius: float = 0.36) -> float:
    """Convert linear velocity to angular velociry of rear wheels"""
    return linear / radius


def ang2lin(angular: float, radius: float = 0.36) -> float:
    """Convert angular velocity to linear velocity of rear wheels"""
    return angular / radius


dt = timestep / 1000.0
trajectory = init_kd_tree(xs, ys)

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

path = 0.0
prev_x, prev_y = None, None

path_values = []

# === ГЛАВНЫЙ ЦИКЛ ===
try:
    while robot.step(timestep) != -1:
        current_time = robot.getTime()

        # ----- GPS -----
        pos = gps.getValues()  # [x, y, z]
        current_x, current_y = pos[0], pos[1]  # Using x and z coordinates

        if prev_x is None or prev_y is None:
            prev_x = current_x
            prev_y = current_y

            s_fine, velocity_fine, s_orig, s_new, x_enhanced, y_enhanced, v_enhanced = (
                create_velocity_bezier_interpolator(
                    start_x=current_x,
                    start_y=current_y,
                    x_turn=x_turn,
                    y_turn=y_turn,
                    v_turn=v_turn,
                    num_points=1000,
                    max_segment_length=300.0,  # Add points if distance > 300m and < 600m
                    intermediate_velocity=ACCELERATION_VELOCITY,  # Velocity for added points
                )
            )

        path += np.sqrt((current_x - prev_x) ** 2 + (current_y - prev_y) ** 2)
        path_values.append(path)

        prev_x, prev_y = current_x, current_y
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

        closest_index_position = find_closest_point(trajectory, current_x, current_y)

        # Find closest velocity point
        closest_index_velocity = find_closest_velocity(s_fine, path)

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
            current_x, current_y, xs, ys, closest_index_position, 10
        )

        cross_track_errors.append(cross_track_error)

        cross_track_error = np.clip(cross_track_error, -1.0, 1.0)

        if np.abs(cross_track_error) == 1.0:
            red_pnt = closest_index_position

        velocity_ref = velocity_fine[closest_index_velocity]
        last_closest_index_velocity = closest_index_velocity

        velocities.append(velocity_ref)

        current_wheel_angle = left_rear_sensor.getValue()
        current_speed = (current_wheel_angle - last_wheel_angle) / dt
        current_velocity = current_speed * RADIUS
        last_wheel_angle = current_wheel_angle

        real_velocities.append(current_velocity)

        velocity_error = velocity_ref - current_velocity

        vel_errors.append(velocity_error)

        # Get corresponding steering angle and velocity

        kp_ct = 0.2
        kd_ct = 0.1
        ki_ct = 0.01

        kp_v = 0.1
        kd_v = 0.01
        ki_v = 2

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

except KeyboardInterrupt:
    pass


# Plot trajectories
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

# Error plots
plt.figure(figsize=(10, 6))
min_length = min(len(cross_track_errors), len(vel_errors))
if min_length > 0:
    time_axis = np.arange(0, min_length)
    plt.plot(time_axis, cross_track_errors[:min_length], label="cross-track")
    plt.plot(time_axis, vel_errors[:min_length], label="velocity")
    plt.xlabel("Time steps")
    plt.ylabel("Errors")
    plt.grid(True)
    plt.legend()
    plt.title("Control Errors Over Time")
else:
    print("No error data to plot")

# Velocity profile plot
plt.figure(figsize=(12, 8))

# Plot reference vs real velocities with length validation
if len(path_values) > 0 and len(velocities) > 0 and len(real_velocities) > 0:
    min_len = min(len(path_values), len(velocities), len(real_velocities))
    plt.plot(
        path_values[:min_len],
        velocities[:min_len],
        linestyle="--",
        color="red",
        linewidth=1,
        label="Reference",
    )
    plt.plot(
        path_values[:min_len], real_velocities[:min_len], color="blue", label="Real"
    )
else:
    print("No velocity data to plot")

# Plot interpolation curve
if len(s_fine) > 0 and len(velocity_fine) > 0:
    plt.plot(
        s_fine,
        velocity_fine,
        color="orange",
        alpha=0.7,
        linewidth=2,
        label="Interpolation",
    )

# Plot target points
if len(s_orig) > 0 and len(v_turn) > 0:
    min_len = min(len(s_orig), len(v_turn))
    plt.scatter(
        s_orig[:min_len],
        v_turn[:min_len],
        marker="x",
        color="magenta",
        s=100,
        label="Target points",
        zorder=5,
    )

# Plot acceleration points
if len(s_new) > 0:
    plt.scatter(
        s_new,
        np.ones(len(s_new)) * ACCELERATION_VELOCITY,
        marker="+",
        color="red",
        s=100,
        linewidth=2,
        label="Acceleration points",
        zorder=5,
    )

plt.xlabel("Path Coordinate (m)")
plt.ylabel("Velocity (m/s)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Velocity Profile and Tracking Performance")
plt.tight_layout()

plt.show()

plt.show()

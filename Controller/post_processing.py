import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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

def find_turn_peaks(yaw_rates, threshold=0.1, window_size=4):
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

    return turn_peaks, turn_starts, turn_ends

def normalize_coordinates(xs, ys):
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1

    xs_norm = [(x - min_x) / range_x for x in xs]
    ys_norm = [(y - min_y) / range_y for y in ys]

    return xs_norm, ys_norm, min_x, max_x, min_y, max_y

def plot_trajectory(xs_norm, ys_norm, turn_starts, turn_ends, turn_peaks):
    plt.figure(figsize=(8, 8))
    plt.plot(xs_norm, ys_norm, marker='o', color='blue', linewidth=1, label='Trajectory')

    for start, end in zip(turn_starts, turn_ends):
        plt.plot(xs_norm[start:end+1], ys_norm[start:end+1], marker='o', linestyle='', color='yellow', alpha=0.7)

    plt.scatter([xs_norm[i] for i in turn_starts], [ys_norm[i] for i in turn_ends], color='red', s=60, label='Turn Start')
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

def smooth_trajectory_with_subsampling(xs, ys, turn_peaks, step=20, num_interp=500):
    xs = np.array(xs)
    ys = np.array(ys)

    n_points = len(xs)
    sampled_indices = list(range(0, n_points, step))

    all_indices = sorted(set(turn_peaks) | set(sampled_indices))

    control_xs = xs[all_indices]
    control_ys = ys[all_indices]

    cs_x = CubicSpline(all_indices, control_xs)
    cs_y = CubicSpline(all_indices, control_ys)

    interp_indices = np.linspace(0, n_points - 1, num_interp)
    smooth_xs = cs_x(interp_indices)
    smooth_ys = cs_y(interp_indices)

    return smooth_xs, smooth_ys

def plot_trajectory_with_smoothing(xs, ys, turn_peaks, smooth_xs, smooth_ys):
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1

    xs_norm = (np.array(xs) - min_x) / range_x
    ys_norm = (np.array(ys) - min_y) / range_y
    smooth_xs_norm = (smooth_xs - min_x) / range_x
    smooth_ys_norm = (smooth_ys - min_y) / range_y

    plt.figure(figsize=(10, 8))
    plt.plot(xs_norm, ys_norm, marker='o', color='blue', linewidth=1, label='Original Trajectory')
    plt.scatter(xs_norm[turn_peaks], ys_norm[turn_peaks], color='green', s=100, label='Turn Peaks')
    plt.plot(smooth_xs_norm, smooth_ys_norm, color='red', linewidth=2, label='Smoothed Trajectory')

    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.title("Trajectory with Additional Sampling Smoothing")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def plot_smoothed_trajectory(xs, ys, turn_peaks, min_x, max_x, min_y, max_y):
    xs = np.array(xs)
    ys = np.array(ys)
    peak_indices = np.array(turn_peaks)

    peak_xs = xs[peak_indices]
    peak_ys = ys[peak_indices]

    cs_x = CubicSpline(peak_indices, peak_xs)
    cs_y = CubicSpline(peak_indices, peak_ys)

    interp_indices = np.linspace(peak_indices[0], peak_indices[-1], 500)
    smooth_xs = cs_x(interp_indices)
    smooth_ys = cs_y(interp_indices)

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

def compute_green_distances(green_indices):
    green_indices = sorted(green_indices)
    distances = []

    for a, b in zip(green_indices, green_indices[1:]):
        distances.append((a, b, b - a))

    return distances
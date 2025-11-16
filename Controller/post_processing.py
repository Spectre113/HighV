import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

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

def plot_trajectory(xs_norm, ys_norm):
    plt.figure(figsize=(8, 8))
    plt.plot(xs_norm, ys_norm, marker='o', color='blue', linewidth=1, label='Trajectory')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajectory")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.savefig("trajectory.png", dpi=200)
    print("Saved trajectory.png")
    plt.show()

def smooth_trajectory_with_window(xs, ys, window_size=5, num_interp_per_segment=100):
    xs = np.array(xs)
    ys = np.array(ys)

    n_points = len(xs)
    smooth_xs_all = []
    smooth_ys_all = []

    for start_idx in range(0, n_points, window_size):
        end_idx = start_idx + window_size
        if end_idx > n_points:
            end_idx = n_points

        segment_indices = np.arange(start_idx, end_idx)
        segment_xs = xs[segment_indices]
        segment_ys = ys[segment_indices]

        if len(segment_indices) < 2:
            break

        cs_x = CubicSpline(segment_indices, segment_xs)
        cs_y = CubicSpline(segment_indices, segment_ys)

        interp_indices = np.linspace(segment_indices[0], segment_indices[-1], num_interp_per_segment)

        smooth_xs_all.append(cs_x(interp_indices))
        smooth_ys_all.append(cs_y(interp_indices))

    smooth_xs = np.concatenate(smooth_xs_all)
    smooth_ys = np.concatenate(smooth_ys_all)

    return smooth_xs, smooth_ys

def plot_trajectory_with_smoothing(xs, ys, smooth_xs, smooth_ys):
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
    plt.plot(smooth_xs_norm, smooth_ys_norm, color='red', linewidth=2, label='Smoothed Trajectory')

    plt.xlabel("Normalized X")
    plt.ylabel("Normalized Y")
    plt.title("Trajectory with Additional Sampling Smoothing")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

def smooth_coordinates(xs, ys, window_length=11, polyorder=3):
    xs_smooth = savgol_filter(xs, window_length, polyorder)
    ys_smooth = savgol_filter(ys, window_length, polyorder)

    return xs_smooth, ys_smooth

def calculate_curvature_global_spline(smooth_xs, smooth_ys, threshold=None):
    n_points = len(smooth_xs)
    s = np.arange(n_points)

    cs_x = CubicSpline(s, smooth_xs)
    cs_y = CubicSpline(s, smooth_ys)

    dx = cs_x.derivative(1)(s)
    dy = cs_y.derivative(1)(s)
    ddx = cs_x.derivative(2)(s)
    ddy = cs_y.derivative(2)(s)

    curvature = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)

    turns_mask = None
    if threshold is not None:
        turns_mask = np.abs(curvature) > threshold

    return curvature, turns_mask


def plot_curvature(curvature):
    plt.figure(figsize=(10, 4))
    plt.plot(curvature, label='Curvature')
    plt.xlabel('Index along trajectory')
    plt.ylabel('Curvature (1/m)')
    plt.title('Curvature along smoothed trajectory')
    plt.grid(True)
    plt.legend()
    plt.show()
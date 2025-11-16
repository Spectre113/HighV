import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

def compute_arc_length(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    dx = np.diff(xs)
    dy = np.diff(ys)
    ds = np.sqrt(dx*dx + dy*dy)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return s

def normalize_coordinates(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    range_x = max_x - min_x if max_x != min_x else 1.0
    range_y = max_y - min_y if max_y != min_y else 1.0
    xs_norm = (xs - min_x) / range_x
    ys_norm = (ys - min_y) / range_y
    return xs_norm.tolist(), ys_norm.tolist(), min_x, max_x, min_y, max_y

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
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(xs) if len(xs)%2==1 else len(xs)-1)
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    if len(xs) < window_length:
        return xs.copy(), ys.copy()
    xs_smooth = savgol_filter(xs, window_length, polyorder)
    ys_smooth = savgol_filter(ys, window_length, polyorder)

    return xs_smooth, ys_smooth

def smooth_trajectory_with_window(xs, ys, window_size=5, num_interp_per_segment=100):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)
    if n < 3:
        return xs.copy(), ys.copy()

    out_x = []
    out_y = []
    i = 0
    while i < n:
        end = min(i + window_size, n - 1)
        seg_x = xs[i:end+1]
        seg_y = ys[i:end+1]
        if len(seg_x) < 2:
            break
        t = np.linspace(0.0, 1.0, len(seg_x))
        csx = CubicSpline(t, seg_x)
        csy = CubicSpline(t, seg_y)
        tt = np.linspace(0.0, 1.0, num_interp_per_segment)
        out_x.append(csx(tt))
        out_y.append(csy(tt))
        i += window_size
    if not out_x:
        return xs.copy(), ys.copy()
    smooth_xs = np.concatenate(out_x)
    smooth_ys = np.concatenate(out_y)
    return smooth_xs, smooth_ys

def calculate_curvature_global_spline(smooth_xs, smooth_ys, threshold=None, return_s=False, eps=1e-9):
    xs = np.asarray(smooth_xs, dtype=float)
    ys = np.asarray(smooth_ys, dtype=float)
    n = len(xs)
    if n == 0:
        if return_s:
            return np.array([]), np.array([])
        return np.array([])

    if n < 3:
        s = compute_arc_length(xs, ys)
        curvature = np.zeros(n, dtype=float)
        if threshold is not None:
            curvature = np.where(np.abs(curvature) > threshold, curvature, 0.0)
        if return_s:
            return curvature, s
        
        return curvature
    
    s = compute_arc_length(xs, ys)

    uniq_s, uniq_idx, inv = np.unique(s, return_index=True, return_inverse=True)

    if uniq_s.size < 2:
        curvature = np.zeros(n, dtype=float)
        if threshold is not None:
            curvature = np.where(np.abs(curvature) > threshold, curvature, 0.0)
        if return_s:
            return curvature, s
        return curvature

    xs_u = xs[uniq_idx]
    ys_u = ys[uniq_idx]

    try:
        cs_x = CubicSpline(uniq_s, xs_u)
        cs_y = CubicSpline(uniq_s, ys_u)

    except Exception as e:
        jitter = np.linspace(0.0, 1e-8, len(uniq_s))
        uniq_s_j = uniq_s + jitter
        cs_x = CubicSpline(uniq_s_j, xs_u)
        cs_y = CubicSpline(uniq_s_j, ys_u)

    dx_ds_u = cs_x.derivative(1)(uniq_s)
    dy_ds_u = cs_y.derivative(1)(uniq_s)
    ddx_ds2_u = cs_x.derivative(2)(uniq_s)
    ddy_ds2_u = cs_y.derivative(2)(uniq_s)

    denom_u = (dx_ds_u**2 + dy_ds_u**2)**1.5
    curvature_u = np.zeros_like(denom_u, dtype=float)
    mask = denom_u > eps
    curvature_u[mask] = (dx_ds_u[mask]*ddy_ds2_u[mask] - dy_ds_u[mask]*ddx_ds2_u[mask]) / denom_u[mask]

    curvature_full = curvature_u[inv]
    curvature_full = np.nan_to_num(curvature_full, nan=0.0, posinf=0.0, neginf=0.0)

    if threshold is not None:
        curvature_filtered = np.where(np.abs(curvature_full) > threshold, curvature_full, 0.0)
        if return_s:
            return curvature_filtered, s
        return curvature_filtered

    if return_s:
        return curvature_full, s
    return curvature_full

def detect_turn_segments(curvature, threshold=0.01, min_length=1):
    mask = np.abs(curvature) > threshold
    segments = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            end = i - 1
            if (end - start + 1) >= min_length:
                segments.append((start, end))
        else:
            i += 1
    return segments

def merge_close_segments(segments, max_gap=5):
    if not segments:
        return []
    
    segments_sorted = sorted(segments, key=lambda x: x[0])
    merged = []
    cur_s, cur_e = segments_sorted[0]
    for s, e in segments_sorted[1:]:
        if s - cur_e - 1 <= max_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    return merged

def filter_turns(turns_info, min_points=20, min_peak_curvature=0.02, min_radius=0.5, max_radius=1000.0):
    out = []
    for t in turns_info:
        pts = t['end'] - t['start'] + 1
        peak = abs(t['apex_curvature'])
        R = t['apex_radius']
        if not np.isfinite(peak) or not np.isfinite(R):
            continue
        if pts < min_points:
            continue
        if peak < min_peak_curvature:
            continue
        if R < min_radius or R > max_radius:
            continue
        out.append(t)

    return out

def analyze_turns_robust(smooth_xs, smooth_ys,
                         threshold=0.08,
                         min_length_pts=20,
                         merge_gap=20,
                         min_peak_curvature=0.02,
                         min_radius=0.5,
                         max_radius=1000.0):
    
    curvature, s = calculate_curvature_global_spline(smooth_xs, smooth_ys, threshold=None, return_s=True)
    raw_segments = detect_turn_segments(curvature, threshold=threshold, min_length=1)
    merged = merge_close_segments(raw_segments, max_gap=merge_gap)
    raw_turns_info = []
    for (st, ed) in merged:
        seg_curv = curvature[st:ed+1]
        if seg_curv.size == 0:
            continue

        local_arg = int(np.argmax(np.abs(seg_curv)))
        apex_idx = st + local_arg
        apex_curv = float(curvature[apex_idx])
        apex_radius = float(np.inf) if abs(apex_curv) < 1e-12 else float(1.0 / abs(apex_curv))
        seg_length = float(s[ed] - s[st]) if len(s) > 1 else 0.0

        info = {
            'start': int(st),
            'end': int(ed),
            'apex_idx': int(apex_idx),
            'apex_curvature': float(apex_curv),
            'apex_radius': float(apex_radius),
            'length': float(seg_length),
            's_start': float(s[st]),
            's_end': float(s[ed])
        }
        raw_turns_info.append(info)

    filtered = filter_turns(raw_turns_info,
                            min_points=min_length_pts,
                            min_peak_curvature=min_peak_curvature,
                            min_radius=min_radius,
                            max_radius=max_radius)

    return filtered, s, curvature, raw_turns_info

def analyze_turns(smooth_xs, smooth_ys, curvature=None, threshold=0.02, min_length=20):
    smooth_xs = np.asarray(smooth_xs)
    smooth_ys = np.asarray(smooth_ys)

    if curvature is None:
        filtered, s, curvature_full, raw = analyze_turns_robust(
            smooth_xs, smooth_ys,
            threshold=threshold,
            min_length_pts=min_length
        )
        return filtered, s

    if isinstance(curvature, (list, tuple)) and len(curvature) == 2:
        cur_arr, s_arr = curvature
        cur_arr = np.asarray(cur_arr)
        s_arr = np.asarray(s_arr)

    else:
        cur_arr = np.asarray(curvature)
        s_arr = compute_arc_length(smooth_xs, smooth_ys)

    raw_segments = detect_turn_segments(cur_arr, threshold=threshold, min_length=1)
    merged = merge_close_segments(raw_segments, max_gap=5)
    raw_turns_info = []
    for (st, ed) in merged:
        seg_curv = cur_arr[st:ed+1]
        if seg_curv.size == 0:
            continue
        local_arg = int(np.argmax(np.abs(seg_curv)))
        apex_idx = st + local_arg
        apex_curv = float(cur_arr[apex_idx])
        apex_radius = float(np.inf) if abs(apex_curv) < 1e-12 else float(1.0 / abs(apex_curv))
        seg_length = float(s_arr[ed] - s_arr[st]) if len(s_arr) > 1 else 0.0
        info = {
            'start': int(st),
            'end': int(ed),
            'apex_idx': int(apex_idx),
            'apex_curvature': float(apex_curv),
            'apex_radius': float(apex_radius),
            'length': float(seg_length),
            's_start': float(s_arr[st]),
            's_end': float(s_arr[ed])
        }
        raw_turns_info.append(info)

    filtered = filter_turns(raw_turns_info, min_points=min_length, min_peak_curvature=0.02, min_radius=0.5, max_radius=1000.0)

    return filtered, s_arr

def plot_curvature_visual(curvature, threshold=0.01):
    curvature = np.asarray(curvature)
    curvature_display = np.copy(curvature)
    curvature_display[np.abs(curvature) <= threshold] = np.nan

    plt.figure(figsize=(12, 5))
    plt.plot(curvature, color='lightgray', label='Curvature (full)', linewidth=1)
    plt.plot(curvature_display, 'r-', label='Turns (above threshold)', linewidth=2)
    plt.axhline(y=threshold, color='blue', linestyle='--', linewidth=1, label='Threshold (+)')
    plt.axhline(y=-threshold, color='blue', linestyle='--', linewidth=1, label='Threshold (-)')
    plt.xlabel('Index along trajectory')
    plt.ylabel('Curvature')
    plt.title('Curvature with highlighted turns')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_turns_on_trajectory(smooth_xs, smooth_ys, turns_info, show_apex=True, figsize=(8,8)):
    plt.figure(figsize=figsize)
    plt.plot(smooth_xs, smooth_ys, color='gray', linewidth=1, label='Smoothed trajectory')
    cmap = plt.get_cmap('tab10')
    for i, info in enumerate(turns_info):
        st, ed = info['start'], info['end']
        color = cmap(i % 10)
        plt.plot(smooth_xs[st:ed+1], smooth_ys[st:ed+1], color=color, linewidth=2.2, label=f'Turn {i+1}')
        if show_apex:
            ax = plt.gca()
            ax.plot(smooth_xs[info['apex_idx']], smooth_ys[info['apex_idx']],
                    marker='o', markersize=6, color='k', markeredgecolor='yellow')
            ax.text(smooth_xs[info['apex_idx']], smooth_ys[info['apex_idx']],
                    f" a{i+1}\nR={info['apex_radius']:.1f}m", fontsize=8, color='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory with detected turns (segments) and apexes')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    plt.show()

def plot_turns_on_curvature(curvature, turns_info, threshold=0.01, figsize=(12,4)):
    plt.figure(figsize=figsize)
    x = np.arange(len(curvature))
    plt.plot(x, curvature, color='lightgray', linewidth=1, label='Curvature (full)')
    cmap = plt.get_cmap('tab10')
    for i, info in enumerate(turns_info):
        st, ed = info['start'], info['end']
        color = cmap(i % 10)
        plt.plot(x[st:ed+1], curvature[st:ed+1], color=color, linewidth=2, label=f'Turn {i+1}')
        ai = info['apex_idx']
        plt.plot(ai, curvature[ai], marker='o', markersize=6, color='k', markeredgecolor=color)
        plt.text(ai, curvature[ai], f" R={info['apex_radius']:.1f}m", fontsize=8, va='bottom')
    plt.axhline(y=threshold, color='blue', linestyle='--', linewidth=1)
    plt.axhline(y=-threshold, color='blue', linestyle='--', linewidth=1)
    plt.xlabel('Index along trajectory')
    plt.ylabel('Curvature (1/m)')
    plt.title('Curvature with detected turns')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    plt.show()

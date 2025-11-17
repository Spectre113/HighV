import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from copy import deepcopy

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


def smooth_coordinates(xs, ys, window_length=15, polyorder=4):
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
                         threshold=0.1,
                         min_length_pts=18,
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

def draw_scaled_circle(ax, center, radius, color=None, linestyle='--', alpha=0.5, fill=False, linewidth=1.5):
    if not np.isfinite(radius) or radius <= 0:
        return None
    circle = plt.Circle((center[0], center[1]), radius, color=color, fill=fill,
                        linestyle=linestyle, alpha=alpha, linewidth=linewidth)
    ax.add_patch(circle)
    return circle

def add_scaled_radius_and_intersections(turns_info, smooth_xs, smooth_ys,
                                        C=10.0, scale_min=4.0, scale_max=7.0):
    xs = np.asarray(smooth_xs)
    ys = np.asarray(smooth_ys)

    for t in turns_info:
        apex_x = float(xs[t['apex_idx']])
        apex_y = float(ys[t['apex_idx']])
        R = float(t.get('apex_radius', np.inf))

        if not np.isfinite(R) or R <= 0:
            t['scaled_radius'] = None
            t['apex_coords'] = (apex_x, apex_y)
            t['entry_idx'] = None
            t['exit_idx'] = None
            continue

        scale = np.clip(C / R, scale_min, scale_max)
        scaled_radius = R * scale

        t['scaled_radius'] = float(scaled_radius)
        t['apex_coords'] = (apex_x, apex_y)

        idxs = find_circle_track_intersections(
            (apex_x, apex_y),
            scaled_radius,
            xs, ys
        )

        if len(idxs) >= 2:
            t['entry_idx'] = int(idxs[0])
            t['exit_idx'] = int(idxs[-1])
        elif len(idxs) == 1:
            t['entry_idx'] = t['exit_idx'] = int(idxs[0])
        else:
            t['entry_idx'] = None
            t['exit_idx'] = None

    return turns_info

def circles_intersect(c1, r1, c2, r2, eps=1e-9):
    if r1 is None or r2 is None:
        return False
    
    if not (np.isfinite(r1) and np.isfinite(r2)):
        return False
    
    d = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
    if d <= (r1 + r2 + eps) and d + eps >= abs(r1 - r2):
        return True
    
    return False

def find_circle_track_intersections(center, radius, xs, ys, tol_ratio=0.05):
    cx, cy = center
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if radius is None or not np.isfinite(radius) or radius <= 0:
        return []

    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    tol = max(1e-6, radius * tol_ratio)

    return np.where(np.abs(dist - radius) < tol)[0].tolist()



def merge_overlapping_scaled_circles(turns_info, smooth_xs, smooth_ys, s=None,
                                     C=10.0, scale_min=4.0, scale_max=7.0,
                                     eps=1e-9):

    xs = np.asarray(smooth_xs)
    ys = np.asarray(smooth_ys)

    if s is None:
        s = compute_arc_length(xs, ys)
    else:
        s = np.asarray(s)

    used = [False] * len(turns_info)
    indexed = sorted(enumerate(turns_info), key=lambda ie: ie[1]['apex_idx'])
    result = []

    i = 0
    while i < len(indexed):
        idx_i, ti = indexed[i]
        if used[idx_i]:
            i += 1
            continue

        merged_turn = dict(ti)

        apex_idx_i = int(merged_turn.get('apex_idx', 0))
        merged_turn['apex_coords'] = (
            float(xs[apex_idx_i]),
            float(ys[apex_idx_i])
        )

        R_i = float(merged_turn.get('apex_radius', np.inf))
        if np.isfinite(R_i) and R_i > 0:
            scale_i = np.clip(C / R_i, scale_min, scale_max)
            merged_turn['scaled_radius'] = float(R_i * scale_i)
        else:
            merged_turn['scaled_radius'] = None

        merged_turn['entry_idx'] = merged_turn.get('entry_idx', None)
        merged_turn['exit_idx'] = merged_turn.get('exit_idx', None)

        used[idx_i] = True

        j = i + 1
        while j < len(indexed):
            idx_j, tj = indexed[j]
            if used[idx_j]:
                j += 1
                continue

            c1 = merged_turn['apex_coords']
            r1_inner = merged_turn.get('apex_radius', None)
            r1_outer = merged_turn.get('scaled_radius', None)

            apex_idx_j = int(tj.get('apex_idx', 0))
            c2 = (
                float(xs[apex_idx_j]),
                float(ys[apex_idx_j])
            )
            r2_inner = tj.get('apex_radius', None)
            if r2_inner is not None:
                r2_inner = float(r2_inner)

            r2_outer = tj.get('scaled_radius', None)
            if r2_outer is None and (r2_inner is not None) and np.isfinite(r2_inner) and r2_inner > 0:
                scale_j = np.clip(C / float(r2_inner), scale_min, scale_max)
                r2_outer = float(r2_inner * scale_j)

            if c1[0] is None or c2[0] is None:
                break

            intersect_inner = False
            intersect_outer = False
            intersect_outer_inner_1 = False
            intersect_outer_inner_2 = False

            if r1_inner is not None and r2_inner is not None:
                try:
                    intersect_inner = circles_intersect(
                        c1, float(r1_inner),
                        c2, float(r2_inner),
                        eps=eps
                    )
                except Exception:
                    intersect_inner = False

            if r1_outer is not None and r2_outer is not None:
                try:
                    intersect_outer = circles_intersect(
                        c1, float(r1_outer),
                        c2, float(r2_outer),
                        eps=eps
                    )
                except Exception:
                    intersect_outer = False

            if r1_outer is not None and r2_inner is not None:
                try:
                    intersect_outer_inner_1 = circles_intersect(
                        c1, float(r1_outer),
                        c2, float(r2_inner),
                        eps=eps
                    )
                except Exception:
                    intersect_outer_inner_1 = False

            if r1_inner is not None and r2_outer is not None:
                try:
                    intersect_outer_inner_2 = circles_intersect(
                        c1, float(r1_inner),
                        c2, float(r2_outer),
                        eps=eps
                    )
                except Exception:
                    intersect_outer_inner_2 = False

            if (intersect_inner or
                intersect_outer or
                intersect_outer_inner_1 or
                intersect_outer_inner_2):

                apex_idx_1 = int(merged_turn['apex_idx'])
                apex_idx_2 = int(tj['apex_idx'])
                s1 = float(s[apex_idx_1])
                s2 = float(s[apex_idx_2])
                s_mid = 0.5 * (s1 + s2)

                new_apex_idx = int(np.argmin(np.abs(s - s_mid)))
                x_mid = float(xs[new_apex_idx])
                y_mid = float(ys[new_apex_idx])

                new_start = min(merged_turn.get('start', apex_idx_1),
                                tj.get('start', apex_idx_2))
                new_end   = max(merged_turn.get('end', apex_idx_1),
                                tj.get('end', apex_idx_2))

                new_s_start = min(merged_turn.get('s_start', s1),
                                  tj.get('s_start', s2))
                new_s_end   = max(merged_turn.get('s_end', s1),
                                  tj.get('s_end', s2))

                curv1 = float(merged_turn.get('apex_curvature', 0.0))
                curv2 = float(tj.get('apex_curvature', 0.0))

                new_apex_curv = curv1 if abs(curv1) >= abs(curv2) else curv2
                new_apex_radius = float(np.inf) if abs(new_apex_curv) < 1e-12 else 1.0 / abs(new_apex_curv)

                if np.isfinite(new_apex_radius) and new_apex_radius > 0:
                    new_scale = np.clip(C / new_apex_radius, scale_min, scale_max)
                    new_scaled_radius = float(new_apex_radius * new_scale)
                else:
                    new_scaled_radius = None

                merged_turn = {
                    'start': int(new_start),
                    'end': int(new_end),
                    'apex_idx': int(new_apex_idx),
                    'apex_curvature': float(new_apex_curv),
                    'apex_radius': float(new_apex_radius),
                    'length': float(new_s_end - new_s_start),
                    's_start': float(new_s_start),
                    's_end': float(new_s_end),
                    'apex_coords': (x_mid, y_mid),
                    'scaled_radius': new_scaled_radius,
                    'entry_idx': None,
                    'exit_idx': None
                }

                used[idx_j] = True
                j += 1

            else:
                break

        result.append(merged_turn)
        i += 1

    result = sorted(result, key=lambda t: t['apex_idx'])
    result = add_scaled_radius_and_intersections(
        result, smooth_xs, smooth_ys,
        C=C, scale_min=scale_min, scale_max=scale_max
    )

    return result

def plot_turns_on_trajectory(smooth_xs, smooth_ys, turns_info, show_apex=True, figsize=(8,8)):
    plt.figure(figsize=figsize)
    plt.plot(smooth_xs, smooth_ys, color='gray', linewidth=1, label='Smoothed trajectory')
    cmap = plt.get_cmap('tab10')
    ax = plt.gca()

    for i, info in enumerate(turns_info):
        st, ed = info['start'], info['end']
        color = cmap(i % 10)
        plt.plot(smooth_xs[st:ed+1], smooth_ys[st:ed+1], color=color, linewidth=2.2, label=f'Turn {i+1}')
        if show_apex:
            apex_x = float(info['apex_coords'][0]) if 'apex_coords' in info else float(smooth_xs[info['apex_idx']])
            apex_y = float(info['apex_coords'][1]) if 'apex_coords' in info else float(smooth_ys[info['apex_idx']])
            apex_radius = info.get('apex_radius', np.inf)

            ax.plot(apex_x, apex_y,
                    marker='o', markersize=6, color='k', markeredgecolor='yellow')
            ax.text(apex_x, apex_y,
                    f" a{i+1}\nR={apex_radius:.1f}m", fontsize=8, color='k')

            if np.isfinite(apex_radius) and apex_radius < 1e6:
                draw_scaled_circle(ax, (apex_x, apex_y), apex_radius, color=color, linestyle='--', alpha=0.25, fill=False, linewidth=1.0)

            scaled_r = info.get('scaled_radius', None)
            if scaled_r is not None and np.isfinite(scaled_r):
                draw_scaled_circle(ax, (apex_x, apex_y), scaled_r, color=color, linestyle=':', alpha=0.35, fill=False, linewidth=1.5)

                eidx = info.get('entry_idx', None)
                xidx = info.get('exit_idx', None)
                if eidx is not None and 0 <= eidx < len(smooth_xs):
                    ax.plot(smooth_xs[eidx], smooth_ys[eidx], marker='D', markersize=8, color=color, label=f'Entry Turn {i+1}')
                if xidx is not None and 0 <= xidx < len(smooth_xs):
                    ax.plot(smooth_xs[xidx], smooth_ys[xidx], marker='X', markersize=8, color=color, label=f'Exit Turn {i+1}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory with detected turns, apexes and adaptive scaled circle intersections')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    plt.show()

def max_accelerations(mass, mu, g=9.81, downforce=0.0):
    normal_force = mass * g + downforce
    max_friction_force = mu * normal_force
    max_acceleration = max_friction_force / mass
    max_lateral_acc = max_acceleration
    max_longitudinal_acc = max_acceleration

    return max_lateral_acc, max_longitudinal_acc

def max_speed(apex_radii, max_lateral_acc):
    apex_radii = np.array(apex_radii)
    speeds = np.sqrt(max_lateral_acc * apex_radii)

    return speeds


def _compute_allowed_lateral_acceleration(mass, mu, g=9.81, downforce=0.0):
    if mass is None or mass <= 0:
        raise ValueError("mass must be positive")
    
    return mu * (g + (downforce / mass))


def _vmax_from_radius(a_allowed, radius):
    if radius is None:
        return np.inf
    try:
        R = float(radius)
    except Exception:
        return np.inf
    if not np.isfinite(R) or R <= 0:
        return np.inf
    return math.sqrt(max(0.0, a_allowed * R))


def compute_turns_vmax_amax(turns_info, mass=1500.0, mu=1.0, g=9.81, downforce=0.0):
    if turns_info is None:
        return []

    a_allowed = _compute_allowed_lateral_acceleration(mass, mu, g, downforce)

    out = deepcopy(turns_info)
    for info in out:
        apex_r = info.get('apex_radius', None)
        apex_v = _vmax_from_radius(a_allowed, apex_r)

        scaled_r = info.get('scaled_radius', None)
        entry_v = _vmax_from_radius(a_allowed, scaled_r if scaled_r is not None else apex_r)
        exit_v = _vmax_from_radius(a_allowed, scaled_r if scaled_r is not None else apex_r)

        info['apex_a_max'] = float(a_allowed)
        info['apex_v_max'] = float(apex_v) if np.isfinite(apex_v) else float('inf')

        info['entry_a_max'] = float(a_allowed)
        info['entry_v_max'] = float(entry_v) if np.isfinite(entry_v) else float('inf')

        info['exit_a_max'] = float(a_allowed)
        info['exit_v_max'] = float(exit_v) if np.isfinite(exit_v) else float('inf')

    return out


def print_turns_vmax_summary(turns_info, to_kmh=True, show_indices=True):
    if not turns_info:
        print("No turns to summarize")
        return

    for i, t in enumerate(turns_info):
        print(f"Turn {i+1}:")
        a = t.get('apex_a_max', None)
        v = t.get('apex_v_max', None)
        if a is None or v is None:
            print("  apex: not available")
        else:
            if to_kmh and np.isfinite(v):
                print(f"  apex: a_max = {a:.3f} m/s^2, V_max = {v:.3f} m/s ({v*3.6:.1f} km/h)")
            elif to_kmh and not np.isfinite(v):
                print(f"  apex: a_max = {a:.3f} m/s^2, V_max = inf")
            else:
                print(f"  apex: a_max = {a:.3f} m/s^2, V_max = {v:.3f} m/s")

        ev = t.get('entry_v_max', None)
        exv = t.get('exit_v_max', None)
        if ev is None or exv is None:
            print("  entry/exit: not available")
        else:
            if to_kmh:
                ev_kmh = ev*3.6 if np.isfinite(ev) else float('inf')
                exv_kmh = exv*3.6 if np.isfinite(exv) else float('inf')
                print(f"  entry: V_max = {ev:.3f} m/s ({ev_kmh:.1f} km/h), exit: V_max = {exv:.3f} m/s ({exv_kmh:.1f} km/h)")
            else:
                print(f"  entry: V_max = {ev:.3f} m/s, exit: V_max = {exv:.3f} m/s")

        if show_indices:
            eidx = t.get('entry_idx', None)
            xidx = t.get('exit_idx', None)
            aidx = t.get('apex_idx', None)
            print(f"    indices -> apex: {aidx}, entry: {eidx}, exit: {xidx}")

        print("")
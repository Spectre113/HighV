import numpy as np


def stanley_control(trajectory, x_current, v, k=0.05):
    pos = x_current[:2]  # [x, y]
    phi = x_current[2]  # Current heading

    # 1. Find closest point on trajectory
    min_dist = np.inf
    closest_point = None

    for point in trajectory:
        pos_t = point[:2]
        # phi_t = point[2]  # Path heading at this point
        dist = np.linalg.norm(pos - pos_t)
        if dist < min_dist:
            min_dist = dist
            closest_point = point

    if closest_point is None:
        return 0.0

    pos_r = closest_point[:2]
    phi_r = closest_point[2]

    # 2. Calculate cross-track error (lateral distance to path)
    # path_tangent = np.array([np.cos(phi_r), np.sin(phi_r)])
    path_normal = np.array([-np.sin(phi_r), np.cos(phi_r)])  # 90° rotated

    vector_to_path = pos - pos_r
    cross_track_error = np.dot(vector_to_path, path_normal)

    # 3. Calculate heading error (difference from path direction)
    heading_error = phi_r - phi
    heading_error = np.arctan2(
        np.sin(heading_error), np.cos(heading_error)
    )  # Wrap to [-π, π]

    # 4. Stanley control law
    if abs(v) < 0.1:  # Avoid division by zero
        steer_correction = 0.0
    else:
        steer_correction = np.arctan2(k * cross_track_error, v)

    # Final steering angle
    delta = heading_error + steer_correction

    return delta


class AdaptiveStanleyController:
    def __init__(self, k_max=2.0, k_min=0.1, transition_point=0.3):
        self.k_max = k_max
        self.k_min = k_min
        self.transition_point = transition_point
        self.cross_track_history = []

    def compute_cross_track_error(self, trajectory, x_current):
        pos = x_current[:2]  # [x, y]

        # 1. Find closest point on trajectory
        min_dist = np.inf
        closest_point = None

        for point in trajectory:
            pos_t = point[:2]
            dist = np.linalg.norm(pos - pos_t)
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        if closest_point is None:
            return 0.0

        pos_r = closest_point[:2]
        phi_r = closest_point[2]

        # 2. Calculate cross-track error (lateral distance to path)
        # path_tangent = np.array([np.cos(phi_r), np.sin(phi_r)])
        path_normal = np.array([-np.sin(phi_r), np.cos(phi_r)])  # 90° rotated

        vector_to_path = pos - pos_r
        cross_track_error = np.dot(vector_to_path, path_normal)

        return cross_track_error

    def compute_heading_error(self, trajectory, x_current):
        pos = x_current[:2]  # [x, y]
        phi = x_current[2]  # Current heading

        # 1. Find closest point on trajectory
        min_dist = np.inf
        closest_point = None

        for point in trajectory:
            pos_t = point[:2]
            # phi_t = point[2]  # Path heading at this point
            dist = np.linalg.norm(pos - pos_t)
            if dist < min_dist:
                min_dist = dist
                closest_point = point

        if closest_point is None:
            return 0.0

        phi_r = closest_point[2]

        # 3. Calculate heading error (difference from path direction)
        heading_error = phi_r - phi
        heading_error = np.arctan2(
            np.sin(heading_error), np.cos(heading_error)
        )  # Wrap to [-π, π]

        return heading_error

    def compute_control(self, trajectory, x_current, v):
        cross_track_error = self.compute_cross_track_error(trajectory, x_current)

        # Store recent errors for smoothing
        self.cross_track_history.append(abs(cross_track_error))
        if len(self.cross_track_history) > 5:
            self.cross_track_history.pop(0)

        # Use filtered error magnitude
        error_magnitude = np.mean(self.cross_track_history)

        # Adaptive gain (exponential decay)
        k = self.k_min + (self.k_max - self.k_min) * np.exp(-3.0 * error_magnitude)

        # Stanley control law
        heading_error = self.compute_heading_error(trajectory, x_current)
        steer_correction = np.arctan2(k * cross_track_error, max(v, 0.1))

        delta = heading_error + steer_correction
        return np.clip(delta, -0.5, 0.5)  # Steering limits


class AdaptiveForwardStanleyController:
    def __init__(self, k_max=2.0, k_min=0.1, lookahead_dist=3.0, max_angle=60.0):
        self.k_max = k_max
        self.k_min = k_min
        self.lookahead_dist = lookahead_dist
        self.max_angle = np.radians(max_angle)
        self.cross_track_history = []

    def find_forward_target_point(self, trajectory, x_current):
        """Find the best target point using forward-aware filtering"""
        pos = x_current[:2]
        phi = x_current[2]
        vehicle_direction = np.array([np.cos(phi), np.sin(phi)])

        candidate_points = []

        for point in trajectory:
            pos_t = point[:2]
            vector_to_point = pos_t - pos
            dist = np.linalg.norm(vector_to_point)

            # Skip points too far ahead
            if dist > self.lookahead_dist:
                continue

            # Check if point is within forward cone
            if dist > 0.01:  # Avoid division by zero
                point_direction = vector_to_point / dist
                dot_product = np.dot(vehicle_direction, point_direction)
                angle = np.arccos(np.clip(dot_product, -1, 1))

                if angle <= self.max_angle:  # Within forward cone
                    candidate_points.append((point, dist, angle))
            else:
                # Current position is on trajectory point
                candidate_points.append((point, 0.0, 0.0))

        if not candidate_points:
            # Fallback: find closest point in any direction
            closest_point = min(trajectory, key=lambda p: np.linalg.norm(p[:2] - pos))
            return closest_point, 0.0, np.pi  # Return with max angle as penalty

        # Prefer points with small angle and reasonable distance
        # Weighted score: angle + distance penalty
        candidate_points.sort(key=lambda x: x[2] + 0.2 * x[1])
        best_point, best_dist, best_angle = candidate_points[0]

        return best_point, best_dist, best_angle

    def compute_cross_track_error(self, trajectory, x_current):
        """Compute cross-track error using forward-aware target point"""
        pos = x_current[:2]

        # Find forward target point
        target_point, dist, angle = self.find_forward_target_point(
            trajectory, x_current
        )

        if target_point is None:
            return 0.0

        pos_r = target_point[:2]
        phi_r = target_point[2]

        # Calculate cross-track error (lateral distance to path)
        path_normal = np.array([-np.sin(phi_r), np.cos(phi_r)])
        vector_to_path = pos - pos_r
        cross_track_error = np.dot(vector_to_path, path_normal)

        return cross_track_error

    def compute_heading_error(self, trajectory, x_current):
        """Compute heading error using forward-aware target point"""
        pos = x_current[:2]
        phi = x_current[2]

        # Find forward target point
        target_point, dist, angle = self.find_forward_target_point(
            trajectory, x_current
        )

        if target_point is None:
            return 0.0

        phi_r = target_point[2]

        # Calculate heading error (difference from path direction)
        heading_error = phi_r - phi
        heading_error = np.arctan2(
            np.sin(heading_error), np.cos(heading_error)
        )  # Wrap to [-π, π]

        return heading_error

    def compute_control(self, trajectory, x_current, v):
        """Compute Stanley steering control with forward awareness and adaptive gain"""
        # Get errors using forward-aware target selection
        cross_track_error = self.compute_cross_track_error(trajectory, x_current)
        heading_error = self.compute_heading_error(trajectory, x_current)

        # Get target point info for debugging/adaptation
        target_point, target_dist, target_angle = self.find_forward_target_point(
            trajectory, x_current
        )

        # Store recent errors for smoothing
        self.cross_track_history.append(abs(cross_track_error))
        if len(self.cross_track_history) > 5:
            self.cross_track_history.pop(0)

        # Use filtered error magnitude
        error_magnitude = np.mean(self.cross_track_history)

        # Adaptive gain: exponential decay + distance-based adjustment
        base_gain = self.k_min + (self.k_max - self.k_min) * np.exp(
            -3.0 * error_magnitude
        )

        # Adjust gain based on target distance and angle
        distance_factor = min(target_dist / self.lookahead_dist, 1.0)
        angle_factor = 1.0 - (target_angle / self.max_angle)

        k = base_gain * (0.7 + 0.3 * distance_factor) * (0.8 + 0.2 * angle_factor)
        k = np.clip(k, self.k_min, self.k_max)

        # Stanley control law with velocity-dependent smoothing
        if abs(v) < 0.1:
            steer_correction = 0.0
        else:
            # Add velocity-dependent smoothing
            velocity_factor = min(abs(v) / 2.0, 1.0)
            steer_correction = (
                np.arctan2(k * cross_track_error, max(v, 0.1)) * velocity_factor
            )

        delta = heading_error + steer_correction

        # Additional smoothing for large angles
        if abs(delta) > 0.3:
            delta = np.sign(delta) * (0.3 + 0.7 * (abs(delta) - 0.3))

        return np.clip(delta, -0.5, 0.5)  # Steering limits

    def get_debug_info(self, trajectory, x_current):
        """Get debugging information about target selection"""
        target_point, target_dist, target_angle = self.find_forward_target_point(
            trajectory, x_current
        )
        cross_track_error = self.compute_cross_track_error(trajectory, x_current)
        heading_error = self.compute_heading_error(trajectory, x_current)

        return {
            "target_point": target_point,
            "target_distance": target_dist,
            "target_angle_deg": np.degrees(target_angle),
            "cross_track_error": cross_track_error,
            "heading_error_deg": np.degrees(heading_error),
            "current_gain": self.k_min
            + (self.k_max - self.k_min)
            * np.exp(-3.0 * np.mean(self.cross_track_history)),
        }


T = 100
reference_y = 44.638

x_t = np.linspace(15.0, -20.0, T)
y_t = np.ones(T) * reference_y
phi_t = np.ones(T) * np.pi


trajectory = np.column_stack([x_t, y_t, phi_t])
# stanley_controller = AdaptiveStanleyController()
# stanley_controller = AdaptiveForwardStanleyController(k_max=2.0, k_min=0.1, lookahead_dist=3.0, max_angle=60.0)

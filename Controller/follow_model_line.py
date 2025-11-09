

class LineFollowerController:
    def __init__(self, k=100.0):
        self.k = k

    def compute_steering(self, angle_err_rad: float) -> float:
        steering_velocity = -self.k * angle_err_rad
        steering_velocity = max(min(steering_velocity, 0.5), -0.5)

        return steering_velocity


class LineFollowerController2:
    def __init__(self, zeta = 0.01, omega_n = 100):
        self.zeta = zeta
        self.omega_n = omega_n

    def compute_steering(self, steering_velocity: float, angle_err_rad: float) -> float:
        steering_acceleration = -2 * self.zeta * self.omega_n * steering_velocity - self.omega_n ** 2 * angle_err_rad

        return steering_acceleration

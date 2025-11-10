class PController:
    def __init__(self, kp=1.0):
        self.kp = kp

    def compute_steering(self, err: float) -> float:
        steering_angle = self.kp * err
        steering_angle = max(min(steering_angle, 0.5), -0.5)

        return steering_angle


class PDController:
    def __init__(self, kp=1.0, kd=1.0):
        self.kp = kp
        self.kd = kd

    def compute_steering(self, err: float, de_dt: float) -> float:
        steering_angle = self.kp * err + self.kd * de_dt
        steering_angle = max(min(steering_angle, 1.0), -1.0)

        return steering_angle


class PIDController:
    def __init__(self, kp=1.0, kd=1.0, ki=1.0):
        self.kp = kp
        self.kd = kd
        self.ki = ki

    def compute_steering(self, err: float, de_dt: float, int_e: float) -> float:
        steering_angle = self.kp * err + self.kd * de_dt + self.ki * int_e
        steering_angle = max(min(steering_angle, 0.5), -0.5)

        return steering_angle

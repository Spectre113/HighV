import math

class LineFollowerController:
    def __init__(self, k=0.02):
        self.k = k

    def compute_steering(self, line_angle_deg: float) -> float:
        line_angle_rad = math.radians(line_angle_deg)
        steering_angle = -self.k * line_angle_rad
        steering_angle = max(min(steering_angle, 0.5), -0.5)

        return steering_angle

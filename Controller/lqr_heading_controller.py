from controller import Robot
import numpy as np
from scipy.linalg import solve_discrete_are

robot = Robot()
timestep = int(robot.getBasicTimeStep())
delta_t = timestep / 1000.0

# === ИНИЦИАЛИЗАЦИЯ УСТРОЙСТВ ===
# [Your existing device initialization remains the same]
left_motor = robot.getDevice("left_rear_wheel")
right_motor = robot.getDevice("right_rear_wheel")
left_steer = robot.getDevice("left_steer")
right_steer = robot.getDevice("right_steer")

# Сенсоры
gps = robot.getDevice("gps")
imu = robot.getDevice("inertial unit")
gyro = robot.getDevice("gyro")
lidar = robot.getDevice("lidar_on")
camera = robot.getDevice("central")

# Датчики вращения колёс
left_rear_sensor = robot.getDevice("left_rear_sensor")
right_rear_sensor = robot.getDevice("right_rear_sensor")
left_steer_sensor = robot.getDevice("left_steer_sensor")
right_steer_sensor = robot.getDevice("right_steer_sensor")

# === ВКЛЮЧЕНИЕ СЕНСОРОВ ===
# [Your existing sensor enabling remains the same]
gps.enable(timestep)
imu.enable(timestep)
gyro.enable(timestep)
lidar.enable(timestep)
camera.enable(timestep)
left_rear_sensor.enable(timestep)
right_rear_sensor.enable(timestep)
left_steer_sensor.enable(timestep)
right_steer_sensor.enable(timestep)

# === НАСТРОЙКА МОТОРОВ ===
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# === DISCRETE BICYCLE MODEL LQR CONTROLLER ===
class DiscreteBicycleLQRController:
    def __init__(
        self,
        wheelbase=2.5,
        max_steer_angle=0.5,
        max_speed=10.0,
        max_omega=2.0,
        dt=0.032,
    ):
        self.L = wheelbase
        self.max_steer = max_steer_angle
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.dt = dt

        # Desired state (goal position and orientation)
        self.x_des = np.array([15, 45, np.pi])  # [x, y, phi]
        self.u_des = np.array([0.1, 0.1])  # [v, omega] at equilibrium

        self.K = self.compute_lqr_gains()

    def discrete_bicycle_dynamics(self, x, u):
        """Discrete bicycle model with both steering and angular velocity"""
        x_pos, y_pos, phi = x
        v, omega = u

        # Discrete update (Euler integration)
        x_new = x_pos + v * np.cos(phi) * self.dt
        y_new = y_pos + v * np.sin(phi) * self.dt

        # Combined steering and direct angular velocity
        phi_new = phi + (omega) * self.dt

        # Normalize angle to [-pi, pi]
        phi_new = np.arctan2(np.sin(phi_new), np.cos(phi_new))

        return np.array([x_new, y_new, phi_new])

    def numerical_linearization(self, x_nom, u_nom, epsilon=1e-6):
        """Numerical linearization with 3 control inputs"""
        n_states = len(x_nom)
        n_inputs = len(u_nom)

        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_inputs))

        # Nominal next state
        x_next_nom = self.discrete_bicycle_dynamics(x_nom, u_nom)

        # Compute A matrix: df/dx
        for i in range(n_states):
            dx = np.zeros(n_states)
            dx[i] = epsilon
            x_next_perturbed = self.discrete_bicycle_dynamics(x_nom + dx, u_nom)
            A[:, i] = (x_next_perturbed - x_next_nom) / epsilon

        # Compute B matrix: df/du (now with 3 inputs)
        for i in range(n_inputs):
            du = np.zeros(n_inputs)
            du[i] = epsilon
            x_next_perturbed = self.discrete_bicycle_dynamics(x_nom, u_nom + du)
            B[:, i] = (x_next_perturbed - x_next_nom) / epsilon

        return A, B

    def robust_solve_dare(self, A, B, Q, R, max_attempts=5):
        """Robust Discrete Algebraic Riccati Equation solver"""
        for attempt in range(max_attempts):
            try:
                # Add regularization for numerical stability
                epsilon = 10 ** (-4 - attempt)
                Q_reg = Q + epsilon * np.eye(Q.shape[0])
                R_reg = R + epsilon * np.eye(R.shape[0])

                P = solve_discrete_are(A, B, Q_reg, R_reg)
                K = np.linalg.inv(R_reg + B.T @ P @ B) @ B.T @ P @ A
                return K
            except np.linalg.LinAlgError:
                if attempt == max_attempts - 1:
                    print(
                        f"DARE failed after {max_attempts} attempts, using fallback gains"
                    )
                    return np.array([[0.3, 0, 0], [0, 0, 1.0]])
                continue

    def compute_lqr_gains(self):
        """Compute LQR gains with 2 control inputs"""
        try:
            A, B = self.numerical_linearization(self.x_des, self.u_des)

            # Cost matrices - now with 2 control inputs
            Q = np.diag([50.0, 50.0, 1.0])  # State penalty
            R = np.diag([1000.0, 1000.0])  # Control penalty [v, omega]

            return self.robust_solve_dare(A, B, Q, R)

        except Exception as e:
            print(f"LQR computation failed: {e}, using fallback gains")
            return np.array([[0.3, 0], [0, 0], [0, 0]])

    def compute_control(self, x_current):
        """Compute LQR control with 3 control inputs"""
        error = x_current - self.x_des

        # LQR control law: u = u_nom - K*(x - x_nom)
        u_control = self.u_des - self.K @ error

        # Apply saturation
        v_cmd = np.clip(u_control[0], -self.max_speed, self.max_speed)
        omega_cmd = np.clip(u_control[1], -self.max_omega, self.max_omega)

        return v_cmd, omega_cmd


def compute_motor_commands(v_cmd, omega_cmd, wheel_radius=0.3, track_width=2.851):
    """Convert to motor commands for combined steering"""
    if abs(v_cmd) > 0.1:  # Avoid division by zero
        delta = np.arctan(omega_cmd * track_width / v_cmd)
    else:
        delta = 0.0

    # Wheel velocities (equal for rear-wheel drive)
    w_left = v_cmd / wheel_radius
    w_right = v_cmd / wheel_radius

    # Steering positions (equal for Ackermann)
    steer_left = np.clip(delta, -1.0, 1.0)
    steer_right = np.clip(delta, -1.0, 1.0)

    return w_left, w_right, steer_left, steer_right


# Initialize discrete LQR controller
controller = DiscreteBicycleLQRController(
    wheelbase=2.851, max_steer_angle=0.5, max_speed=10.0, dt=delta_t
)

# === ГЛАВНЫЙ ЦИКЛ ===
while robot.step(timestep) != -1:
    # ----- СБОР ДАННЫХ С ДАТЧИКОВ -----
    pos = gps.getValues()  # [x, y, z]
    roll, pitch, yaw = imu.getRollPitchYaw()

    # Текущее состояние для модели велосипеда
    x_current = np.array([pos[0], pos[1], yaw])  # [x, y, yaw]

    # ----- ВЫЧИСЛЕНИЕ УПРАВЛЕНИЯ LQR -----
    v_cmd, omega_cmd = controller.compute_control(x_current)

    (
        left_motor_velocity,
        right_motor_velocity,
        left_steer_position,
        right_steer_position,
    ) = compute_motor_commands(v_cmd, omega_cmd)

    # ----- ПРИМЕНЕНИЕ УПРАВЛЕНИЯ -----
    left_motor.setVelocity(left_motor_velocity)
    right_motor.setVelocity(right_motor_velocity)
    left_steer.setPosition(left_steer_position)
    right_steer.setPosition(right_steer_position)

    # ----- ВЫВОД ОТЛАДОЧНОЙ ИНФОРМАЦИИ -----
    error_norm = np.linalg.norm(x_current - controller.x_des)
    print(f"Position: x={pos[0]:.2f}, y={pos[1]:.2f}, yaw={yaw:.3f}")
    print(f"Error norm: {error_norm:.3f}")
    print(f"Control: v={v_cmd:.2f}, omega={omega_cmd:.3f}")
    print(f"Motor commands: L={left_motor_velocity:.2f}, R={right_motor_velocity:.2f}")

    # ----- ПРОВЕРКА ДОСТИЖЕНИЯ ЦЕЛИ -----
    if error_norm < 0.3:  # Tolerance for goal reaching
        print("Goal reached! Stopping...")
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        left_steer.setPosition(0)
        right_steer.setPosition(0)
        # Uncomment break if you want to stop the controller
        # break

    print("-" * 60)

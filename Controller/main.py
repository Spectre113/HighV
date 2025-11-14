from controller import Robot
import numpy as np
# from follow_model_line import LineFollowerController, LineFollowerController2
from pid_controller import PDController
from baseline_detection import MultiCameralineDetector


robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Движение
left_motor = robot.getDevice('left_rear_wheel')
right_motor = robot.getDevice('right_rear_wheel')
left_steer = robot.getDevice('left_steer')
right_steer = robot.getDevice('right_steer')

# Сенсоры
gps = robot.getDevice('gps')
imu = robot.getDevice('inertial unit')
gyro = robot.getDevice('gyro')
lidar = robot.getDevice('lidar_on')

left_camera = robot.getDevice('left')
central_camera = robot.getDevice('central')
right_camera = robot.getDevice('right')

# Датчики вращения колёс
left_rear_sensor = robot.getDevice('left_rear_sensor')
right_rear_sensor = robot.getDevice('right_rear_sensor')
left_steer_sensor = robot.getDevice('left_steer_sensor')
right_steer_sensor = robot.getDevice('right_steer_sensor')

# ВКЛЮЧЕНИЕ СЕНСОРОВ
gps.enable(timestep)
imu.enable(timestep)
gyro.enable(timestep)
lidar.enable(timestep)

left_camera.enable(timestep)
central_camera.enable(timestep)
right_camera.enable(timestep)

left_rear_sensor.enable(timestep)
right_rear_sensor.enable(timestep)
left_steer_sensor.enable(timestep)
right_steer_sensor.enable(timestep)

# === НАСТРОЙКА МОТОРОВ ===
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)


# Setup visualization

width = central_camera.getWidth()
height = central_camera.getHeight()

cameras = [left_camera, central_camera, right_camera]

speed = 20.0
angle = 0.0
err = 0.0

kp = 0.1
kd = 0.01
controller = PDController(MultiCameralineDetector(), kp=kp, kd=kd)
controller.start_processing()

# === ГЛАВНЫЙ ЦИКЛ ===
while robot.step(timestep) != -1:
    # ----- GPS -----
    pos = gps.getValues()  # [x, y, z]
    print(f"GPS: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

    # ----- IMU -----
    roll, pitch, yaw = imu.getRollPitchYaw()
    print(f"IMU: roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")

    # ----- Гироскоп -----
    gyro_values = gyro.getValues()
    print(f"Gyro: x={gyro_values[0]:.3f}, y={gyro_values[1]:.3f}, z={gyro_values[2]:.3f}")

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

    controller.update_sensor(cameras, angle)

    output, err, state = controller.get_latest_result()

    if state and err:
        # img_display.set_data(output

        # Pull towards the line if too close to the edge of camera frame
        counter = 3
        if np.abs(err) > 0.7 and counter > 0:
            counter -= 1
            controller.kp = 0.008
            controller.kd = 0.01
        else:
            counter = 3

    angle = controller.get_current_command()

    # 2nd order control
    # line_angle_rad = -0.05
    # error_rad = left_steer_sensor.getValue() - line_angle_rad
    # acceleration = LineFollowerController2().compute_steering(velocity, error_rad)

    # dt = timestep / 1000.0
    # velocity = velocity + acceleration * dt
    # angle = left_steer_sensor.getValue() + velocity * dt

    # 1st order control
    # velocity = LineFollowerController().compute_steering(error_rad)

    # dt = timestep / 1000.0
    # angle = left_steer_sensor.getValue() + velocity * dt

    # angle = np.clip(angle, -1, 1)

    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)
    left_steer.setPosition(angle)
    right_steer.setPosition(angle)

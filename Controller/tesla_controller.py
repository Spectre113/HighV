from controller import Robot, Camera
import cv2
import numpy as np
from follow_model_line import LineFollowerController


def read_img(camera: Camera):
    width = camera.getWidth()
    height = camera.getHeight()
    img_bytes = camera.getImage()
    img = np.frombuffer(img_bytes, np.uint8).reshape((height, width, 4)).copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


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
camera = robot.getDevice('camera')

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
camera.enable(timestep)
left_rear_sensor.enable(timestep)
right_rear_sensor.enable(timestep)
left_steer_sensor.enable(timestep)
right_steer_sensor.enable(timestep)

# === НАСТРОЙКА МОТОРОВ ===
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

speed = 20.0
angle = 0

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

    # ----- Камера -----
    img = read_img(camera)
    height, width, _ = img.shape
    print(f"Camera: {width}x{height}")

    # ----- Датчики вращения -----
    print(f"Left rear wheel: {left_rear_sensor.getValue():.3f}")
    print(f"Right rear wheel: {right_rear_sensor.getValue():.3f}")
    print(f"Left steer angle: {left_steer_sensor.getValue():.3f}")
    print(f"Right steer angle: {right_steer_sensor.getValue():.3f}")

    print("-" * 60)

    # ----- Управление движением -----
    line_angle_deg = 120
    angle = LineFollowerController().compute_steering(line_angle_deg)

    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)
    left_steer.setPosition(angle)
    right_steer.setPosition(angle)

# HighV: Adaptive Control for Driverless Cars

An automated control system for driverless vehicles with the ability to adapt to unseen tracks. This project implements motion planning that preserves velocity capabilities by modeling track curvatures, featuring a dual-component controller with camera-aided map extraction and adaptive control with velocity profiling.

## Project Overview

The controller consists of two major components:

### Stage 1: Camera-Aided Map Information Extraction
- Triad camera system (central, left, right) on the front of the car
- Central camera observes forward perspective, side cameras capture angled views
- Classical CV techniques (Canny Edge Detection, Hough Line Transform) for lane detection
- Lateral error computation as signed distance from lane center to image center
- PD-controller for steering angle regulation
- Trajectory capture of x and y coordinates

### Stage 2: Adaptive Control with Velocity Profiling
- Post-processing of captured trajectory
- Turn point detection via geometric analysis of road curvature
- Apex point computation for acceleration/velocity limitation
- Velocity interpolation for linear track sections
- Dual PID-controllers: cross-track error for steering, velocity error for acceleration

## Installation & Setup

1. **Install Webots Simulator** from [https://cyberbotics.com/](https://cyberbotics.com/)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Spectre113/HighV.git
   ```

3. **Set up virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   # or use Conda
   pip install -r requirements.txt
   ```

4. **Configure Webots** for external controllers ([Webots documentation](https://cyberbotics.com/doc/guide/running-extern-robot-controllers))

5. **Load the world** in Webots: `File->Open World` from `World/yas_marina.wbt`

## Usage

Run the controllers in sequence:

```bash
$WEBOTS-HOME/webots-controller Controller/camera_controller.py  # lane detection controller (run first)
$WEBOTS-HOME/webots-controller Controller/trajectory_follower.py  # adaptive controller
```

**Important**: Reset simulation (first button to the right of timer) before reaching start point to avoid smoothing malfunctions.

## Results & Artifacts

Dynamic plots display key performance parameters:
- **First controller**: Trajectory and detected turn points
- **Second controller**: Trajectory, velocities, and errors

Testing conducted with Tesla Model 3 on Yas-Marina track using Webots Simulator. Performance details available in the `report` folder.

## Team Contributions

- **Ilya Grigorev**: Camera hierarchy control logic, PID-controller design and tests
- **Vladimir Toporkov**: Post-processing pipeline and vehicle cameras setup  
- **Salavat Faizullin**: CV technology application on lane detection and documentation

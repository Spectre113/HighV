import threading
import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

# =============================================================================
# Abstract State Definitions
# =============================================================================


@dataclass
class ControlState(ABC):
    """Base class for all control states"""

    timestamp: float
    sensor_data: Any
    current_command: float = 0.0

    @abstractmethod
    def get_error_components(self) -> Dict[str, float]:
        """Return error components needed for control"""
        pass


@dataclass
class PState(ControlState):
    """State for Proportional control"""

    error: float = 0.0

    def get_error_components(self) -> Dict[str, float]:
        return {"error": self.error}


@dataclass
class PDState(ControlState):
    """State for Proportional-Derivative control"""

    error: float = 0.0
    error_derivative: float = 0.0
    previous_error: float = 0.0

    def get_error_components(self) -> Dict[str, float]:
        return {"error": self.error, "error_derivative": self.error_derivative}


@dataclass
class PIDState(ControlState):
    """State for Proportional-Integral-Derivative control"""

    error: float = 0.0
    error_derivative: float = 0.0
    error_integral: float = 0.0
    previous_error: float = 0.0
    max_integral: float = 2.0

    def get_error_components(self) -> Dict[str, float]:
        return {
            "error": self.error,
            "error_derivative": self.error_derivative,
            "error_integral": self.error_integral,
        }

    def update_integral(self, error: float, dt: float):
        """Update integral with anti-windup"""
        self.error_integral += error * dt
        self.error_integral = max(
            min(self.error_integral, self.max_integral), -self.max_integral
        )

    def reset_integral(self):
        """Reset integral windup"""
        self.error_integral = 0.0


# =============================================================================
# Abstract Controller Base
# =============================================================================


class NonBlockingSensorController(ABC):
    def __init__(self, sensor_callback, state_type: type):
        self.state_queue = queue.Queue(maxsize=2)  # Store recent states
        self.result_queue = queue.Queue(maxsize=2)
        self.current_command = 0.0
        self.processing = False
        self.thread = None
        self.stop_flag = False
        self.sensor_callback = sensor_callback
        self.state_type = state_type
        self.current_state: Optional[ControlState] = None

    def start_processing(self):
        """Start the background processing thread"""
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    @abstractmethod
    def compute_command(self, state: ControlState) -> float:
        """Compute control command from state"""
        pass

    @abstractmethod
    def update_state(
        self, state: ControlState, processed_data: Any, error: float
    ) -> ControlState:
        """Update state with new sensor data and error"""
        pass

    def _processing_loop(self):
        """Background processing loop with state management"""
        while not self.stop_flag:
            try:
                # Wait for new state
                state = self.state_queue.get(timeout=0.1)

                # Process sensor data
                self.processing = True
                processed_data, error = self.sensor_callback(state.sensor_data)

                # Update state with new information
                updated_state = self.update_state(state, processed_data, error)

                # Compute control command
                command = self.compute_command(updated_state)
                updated_state.current_command = command

                self.processing = False

                # Store updated state and result
                self.current_state = updated_state
                if self.result_queue.full():
                    self.result_queue.get_nowait()
                self.result_queue.put((processed_data, error, updated_state))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                self.processing = False

    def update_sensor(
        self, sensor_data: Any, current_command: float = 0.0
    ) -> ControlState:
        """Create and queue new state with sensor data"""
        new_state = self.state_type(
            timestamp=time.time(),
            sensor_data=sensor_data,
            current_command=current_command,
        )

        if self.state_queue.full():
            self.state_queue.get_nowait()  # Discard oldest state
        self.state_queue.put(new_state)

        return new_state

    def get_latest_result(self):
        """Get latest processing result if available"""
        try:
            processed_data, error, state = self.result_queue.get_nowait()
            self.current_state = state
            return processed_data, error, state
        except queue.Empty:
            if self.current_state:
                return None, self.current_state.get_error_components()['error'], self.current_state
            return None, None, self.current_state

    def get_current_command(self):
        """Get current active command"""
        if self.current_state:
            return self.current_state.current_command
        return self.current_command

    def get_current_state(self) -> Optional[ControlState]:
        """Get current state for debugging/monitoring"""
        return self.current_state

    def stop(self):
        """Stop the processing thread"""
        self.stop_flag = True
        if self.thread:
            self.thread.join()


# =============================================================================
# Concrete Controller Implementations
# =============================================================================


class PController(NonBlockingSensorController):
    def __init__(self, sensor_callback, kp=1.0):
        super().__init__(sensor_callback, PState)
        self.kp = kp

    def compute_command(self, state: PState) -> float:
        error_components = state.get_error_components()
        steering_angle = self.kp * error_components["error"]
        return max(min(steering_angle, 0.5), -0.5)

    def update_state(self, state: PState, processed_data: Any, error: float) -> PState:
        """Update P state with new error"""
        state.error = error
        return state


class PDController(NonBlockingSensorController):
    def __init__(self, sensor_callback, kp=1.0, kd=0.1):
        super().__init__(sensor_callback, PDState)
        self.kp = kp
        self.kd = kd
        self.last_timestamp = time.time()

    def compute_command(self, state: PDState) -> float:
        error_components = state.get_error_components()
        steering_angle = (
            self.kp * error_components["error"]
            + self.kd * error_components["error_derivative"]
        )
        return max(min(steering_angle, 1.0), -1.0)

    def update_state(
        self, state: PDState, processed_data: Any, error: float
    ) -> PDState:
        """Update PD state with error and derivative"""
        current_time = time.time()
        dt = current_time - self.last_timestamp if self.last_timestamp else 0.01

        if dt > 0:
            state.error_derivative = (error - state.previous_error) / dt
        else:
            state.error_derivative = 0.0

        state.error = error
        state.previous_error = error
        self.last_timestamp = current_time

        return state


class PIDController(NonBlockingSensorController):
    def __init__(self, sensor_callback, kp=1.0, kd=0.1, ki=0.01, max_integral=2.0):
        super().__init__(sensor_callback, PIDState)
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.last_timestamp = time.time()

    def compute_command(self, state: PIDState) -> float:
        error_components = state.get_error_components()
        steering_angle = (
            self.kp * error_components["error"]
            + self.kd * error_components["error_derivative"]
            + self.ki * error_components["error_integral"]
        )
        return max(min(steering_angle, 0.5), -0.5)

    def update_state(
        self, state: PIDState, processed_data: Any, error: float
    ) -> PIDState:
        """Update PID state with error, derivative, and integral"""
        current_time = time.time()
        dt = current_time - self.last_timestamp if self.last_timestamp else 0.01

        if dt > 0:
            state.error_derivative = (error - state.previous_error) / dt
            state.update_integral(error, dt)
        else:
            state.error_derivative = 0.0

        state.error = error
        state.previous_error = error
        self.last_timestamp = current_time

        return state

    def reset_integral(self):
        """Reset integral windup"""
        if self.current_state and isinstance(self.current_state, PIDState):
            self.current_state.reset_integral()


# =============================================================================
# Usage Example
# =============================================================================

# Create PID controller
# pid_controller = PIDController(
#     sensor_callback=example_sensor_processing, kp=0.8, kd=0.2, ki=0.05
# )
# pid_controller.start_processing()

# # Simulate main loop
# for i in range(10):
#     # Update with new sensor data
#     sensor_data = f"image_frame_{i}"
#     pid_controller.update_sensor(sensor_data)

#     # Get latest result
#     processed_data, error, state = pid_controller.get_latest_result()
#     if state:
#         print(f"Command: {state.current_command:.3f}, Error: {error:.3f}")
#         if isinstance(state, PIDState):
#             print(
#                 f"  Integral: {state.error_integral:.3f}, Derivative: {state.error_derivative:.3f}"
#             )
#     command = pid_controller.get_current_command()
#     time.sleep(0.1)

# pid_controller.stop()

import threading
import time

# Simulate YuMiRobotEnv with just the relevant parts
class DummyRobot:
    def __init__(self):
        self._reset_event = threading.Event()
        self.reset_triggered = False

    def _on_interface_fail(self):
        print("[Robot] Failure detected.")
        self.reset_triggered = True
        self._reset_event.clear()
        print("[Robot] Waiting for external reset...")
        self._reset_event.wait()
        print("[Robot] Resumed after reset.")

    def _on_interface_reset_request(self):
        print("[Robot] Resetting interface...")
        time.sleep(0.5)
        self.reset_triggered = False
        self._reset_event.set()
        print("[Robot] Interface reset complete.")


robot = DummyRobot()

# Start the fail handler in a separate thread
fail_thread = threading.Thread(target=robot._on_interface_fail)
fail_thread.start()

# Let it block for a bit
time.sleep(2)

# Now simulate a reset request (should unblock the fail thread)
robot._on_interface_reset_request()

# Wait for thread to finish
fail_thread.join()
print("Test complete.")

import rospy
import cv2
import numpy as np
import json
import os
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.configs.envconfig import ExperimentConfig


class ClipPlacementGUI:
    def __init__(self):
        rospy.init_node("clip_placement_tool")
        self.zed_cam = ZedCameraSubscriber()
        cfg = ExperimentConfig

        self.wait_for_camera()
        self.image, (self.crop_x, self.crop_y) = self.load_zed_image()
        self.clip_positions = self.load_board_config()
        self.current_orientation = 0
        self.current_clip_type = 1
        self.preview_position = None
        self.clip_types = {1: "6Pin", 2: "2Pin", 3: "C-Clip", 4: "Plug"}
        self.config_path = cfg.board_cfg_path
        self.annotated_img_path = cfg.bg_img_path
        cv2.namedWindow("Board Setup")
        cv2.setMouseCallback("Board Setup", self.preview_clip)

    def wait_for_camera(self):
        print("Waiting for ZED camera stream...")
        while self.zed_cam.rgb_image is None or self.zed_cam.depth_image is None:
            rospy.sleep(0.1)

    def load_zed_image(self):
        rgb_frame = self.zed_cam.rgb_image
        if rgb_frame is None:
            print("No frame received!")
            exit()
        return rgb_frame.copy(), (0, 0)  # Assuming no cropping for now

    def preview_clip(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.preview_position = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.clip_positions.append(
                {
                    "x": x,
                    "y": y,
                    "type": self.current_clip_type,
                    "orientation": self.current_orientation,
                }
            )

    def draw_clips(self):
        img_display = self.image.copy()

        # Draw instructions on top-left
        instructions = [
            "Instructions:",
            "1-4: Select Clip Type",
            "Left/Right Arrows: Rotate Clip",
            "Click: Place Clip",
            "S: Save Config & Image",
            "Q: Quit",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(
                img_display,
                text,
                (10, 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        for clip in self.clip_positions:
            self.draw_single_clip(
                img_display, clip["x"], clip["y"], clip["type"], clip["orientation"]
            )

        if self.preview_position:
            self.draw_single_clip(
                img_display,
                self.preview_position[0],
                self.preview_position[1],
                self.current_clip_type,
                self.current_orientation,
                preview=True,
            )

        return img_display

    def draw_single_clip(self, img, x, y, clip_type, orientation, preview=False):
        center = (x, y)
        color = (0, 255, 255) if preview else (0, 0, 255)
        cv2.circle(img, center, 10, color, -1)

        arrow_length = 30
        angle_rad = np.deg2rad(orientation)

        if clip_type == 3:  # Type C - arrow from the middle
            arrow_start = center
            arrow_end = (
                int(center[0] + arrow_length * np.cos(angle_rad)),
                int(center[1] + arrow_length * np.sin(angle_rad)),
            )
        else:
            arrow_start = (
                int(center[0] - (arrow_length / 2) * np.cos(angle_rad)),
                int(center[1] - (arrow_length / 2) * np.sin(angle_rad)),
            )
            arrow_end = (
                int(center[0] + (arrow_length / 2) * np.cos(angle_rad)),
                int(center[1] + (arrow_length / 2) * np.sin(angle_rad)),
            )

        cv2.arrowedLine(img, arrow_start, arrow_end, (255, 0, 0), 2)
        cv2.putText(
            img,
            self.clip_types[clip_type] + str([x, y]),
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    def save_board_config(self):
        with open(self.config_path, "w") as f:
            json.dump(self.clip_positions, f, indent=4)
        print(f"Board configuration saved to {self.config_path}")

        img_display = self.draw_clips()
        cv2.imwrite(self.annotated_img_path, img_display)
        print(f"Annotated image saved to {self.annotated_img_path}")

    def load_board_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    print("Loaded existing board configuration.")
                    return json.load(f)
        except:
            print("No existing board configuration found.")
        return []

    def run(self):
        while True:
            img_display = self.draw_clips()
            cv2.imshow("Board Setup", img_display)
            key = cv2.waitKey(50) & 0xFF

            if key in [ord("1"), ord("2"), ord("3"), ord("4")]:
                self.current_clip_type = int(chr(key))
                print(f"Selected Clip Type: {self.clip_types[self.current_clip_type]}")

            elif key == ord("s"):  # Save
                self.save_board_config()

            elif key == ord("q"):  # Quit
                self.save_board_config()
                print("Final Board Configuration:", self.clip_positions)
                break

            elif key == 81:  # Left Arrow
                self.current_orientation = (self.current_orientation - 10) % 360
            elif key == 83:  # Right Arrow
                self.current_orientation = (self.current_orientation + 10) % 360

        cv2.destroyAllWindows()


if __name__ == "__main__":
    ClipPlacementGUI().run()

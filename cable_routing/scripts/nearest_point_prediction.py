import rospy
import cv2
import numpy as np
import json
import os
from string import ascii_uppercase
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber
from cable_routing.configs.envconfig import ExperimentConfig
import heapq


class NearestPointGUI:
    def __init__(self):
        rospy.init_node("nearest_point_tool")
        self.zed_cam = ZedCameraSubscriber()

        self.wait_for_camera()
        self.image, (self.crop_x, self.crop_y) = self.load_zed_image()
        self.preview_position = None
        self.current_orientation = 0

        self.annotations = {}
        cv2.namedWindow("Board Setup")
        cv2.setMouseCallback("Board Setup", self.preview_point)

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

    def preview_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.preview_position = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            existing_keys = set(self.annotations.keys())

            # Find the next available letter
            for letter in letters:
                if letter not in existing_keys:
                    # Make nearest point prediction
                    predicted_x, predicted_y, predicted_orientation = (
                        self.find_nearest_point((x, y))
                    )
                    # self.annotations = {}

                    self.annotations[letter] = {
                        "x": x,
                        "y": y,
                        "label": "",  # "Point Before Regrasp",
                        "color": (0, 0, 255),
                        "orientation": self.current_orientation,
                    }
                    # print(f"Placed point '{letter}' at ({x}, {y})")

                    # Add the nearest cable point
                    self.annotations["Prediction " + letter] = {
                        "x": predicted_x,
                        "y": predicted_y,
                        "label": "",  # "Predicted Point",
                        "color": (255, 0, 0),
                        "orientation": (
                            predicted_orientation
                            if (predicted_x != x or predicted_y != y)
                            else self.current_orientation
                        ),
                    }
                    print(f"Placed point '{letter}' at ({predicted_x}, {predicted_y})")
                    break
            else:
                print("Maximum points reached, no available letters!")

    def render_annotations(self):
        img_display = self.image.copy()

        # Draw instructions on top-left
        instructions = ["Visualizing Nearest Cable Point"]
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

        for idx, (letter, point) in enumerate(self.annotations.items()):
            # print(letter)
            # print(point)
            self.draw_point(
                img_display,
                point["x"],
                point["y"],
                point["label"],
                point["color"],
                point["orientation"],
            )

        if self.preview_position:
            self.draw_point(
                img_display,
                self.preview_position[0],
                self.preview_position[1],
                "Tentative Point",
                (0, 255, 255),
                self.current_orientation,
            )

        return img_display

    def draw_point(
        self, img, x, y, label="Default", color=(0, 255, 255), orientation=0
    ):
        center = (x, y)
        cv2.circle(img, center, 10, color, -1)

        arrow_length = 30
        angle_rad = np.deg2rad(orientation)

        arrow_start = (
            int(center[0] - (arrow_length / 2) * np.cos(angle_rad)),
            int(center[1] - (arrow_length / 2) * np.sin(angle_rad)),
        )
        arrow_end = (
            int(center[0] + (arrow_length / 2) * np.cos(angle_rad)),
            int(center[1] + (arrow_length / 2) * np.sin(angle_rad)),
        )

        cv2.arrowedLine(img, arrow_start, arrow_end, (255, 255, 0), 2)

        cv2.putText(
            img,
            label,
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    def is_cable_pixel(self, gray, pos, threshold=100, use_erode=True):
        x, y = pos

        if use_erode:
            kernel = np.ones((3, 3), np.uint8)
            gray = cv2.erode(gray, kernel, iterations=1)

        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            return gray[y, x] > threshold
        else:
            return False
        
    
    def run(self):
        while True:
            img_display = self.render_annotations()
            cv2.imshow("Board Setup", img_display)
            key = cv2.waitKey(50) & 0xFF

            if key == ord("q"):  # Quit
                print("Final Board Configuration:", self.annotations)
                break

            elif key == 81:  # Left Arrow
                self.current_orientation = (self.current_orientation - 10) % 360
            elif key == 83:  # Right Arrow
                self.current_orientation = (self.current_orientation + 10) % 360

        cv2.destroyAllWindows()

    # NEW FUNCTION: Find the nearest point with sufficient contrast with the background
    def find_nearest_point(self, start_point):
        squared_magnitude = lambda my_tuple: sum([i**2 for i in my_tuple])

        x, y = start_point[0], start_point[1]
        image = self.image
        if len(image.shape) == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image.copy()
            
        directions = sum([[(a, b) for a in [-1, 0, 1]] for b in [-1, 0, 1]], start=[])

        heap = [(0, (x, y))]
        visited = set()
        visited.add((x, y))

        direction = (1, 0)
        steps_in_interval = 0
        interval_length = 1

        while heap:
            dist, pos = heapq.heappop(heap)
            # if squared_magnitude(image[pos[1]][pos[0]]) > 170000:
            #     x = pos[0]
            #     y = pos[1]
            #     break
            if self.is_cable_pixel(gray_img, pos):
                x = pos[0]
                y = pos[1]
                break
            for dx, dy in directions:
                nx, ny = pos[0] + dx, pos[1] + dy
                neighbor = (nx, ny)
                if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_dist = squared_magnitude(
                            (neighbor[0] - start_point[0], neighbor[1] - start_point[1])
                        )
                        heapq.heappush(heap, (new_dist, neighbor))

        delta_x = start_point[0] - x
        delta_y = start_point[1] - y

        predicted_x = x
        predicted_y = y
        predicted_orientation = 180 / np.pi * np.arctan2(delta_y, delta_x)
        return predicted_x, predicted_y, predicted_orientation

    # Find the nearest point with sufficient contrast with the background, Manhattan method
    def find_nearest_point_manhattan(self, start_point):
        squared_magnitude = lambda my_tuple: sum([i**2 for i in my_tuple])

        x, y = start_point[0], start_point[1]
        image = self.image

        direction = (1, 0)
        steps_in_interval = 0
        interval_length = 1

        while interval_length < 1000:
            x += direction[0]
            y += direction[1]
            steps_in_interval += 1
            # self.annotations[f"{x},{y}"] = {
            #             "x": x,
            #             "y": y,
            #             "label": "",
            #             "color": (255, 0, 0)
            #         }

            if steps_in_interval >= interval_length:
                print(direction)
                if direction[0] == 1:
                    direction = (0, -1)
                    print("Color", x, y, image[y][x])
                elif direction[1] == -1:
                    direction = (-1, 0)
                    interval_length += 1
                    print("Color", x, y, image[y][x])
                elif direction[0] == -1:
                    direction = (0, 1)
                    print("Color", x, y, image[y][x])
                elif direction[1] == 1:
                    direction = (1, 0)
                    interval_length += 1
                    print("Color", x, y, image[y][x])
                steps_in_interval = 0

            if squared_magnitude(image[y][x]) > 150000:
                print(x, y)
                break

        predicted_x = x
        predicted_y = y
        return predicted_x, predicted_y


if __name__ == "__main__":
    NearestPointGUI().run()

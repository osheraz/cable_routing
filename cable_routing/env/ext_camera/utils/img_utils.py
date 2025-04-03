import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SCALE_FACTOR = 1.0
SAFE_HEIGHT = 0.002

def normalize(vec):
    return vec / np.linalg.norm(vec)

def mask_clip_region(image, clip, mask_size=30):
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    x, y = clip["x"], clip["y"]

    for i in range(-mask_size, mask_size):
        for j in range(-mask_size, mask_size):
            px, py = x + i, y + j
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                mask[py, px] = 0

    return mask


def center_pixels_on_cable(image, pixels, num_options=10, display=False):
    image_mask = image[:, :, 0] > 100
    kernel = np.ones((2, 2), np.uint8)
    image_mask = cv2.erode(image_mask.astype(np.uint8), kernel, iterations=1)
    white_pixels = np.argwhere(image_mask)

    processed_pixels = []
    for pixel in pixels:
        distances = np.linalg.norm(white_pixels - pixel, axis=1)
        valid_indices = np.where(distances >= 100)[0]
        if len(valid_indices) > 0:
            sorted_indices = np.argsort(distances[valid_indices])
            selected_pixels = white_pixels[valid_indices[sorted_indices[:num_options]]]
            processed_pixels.append(selected_pixels)

    if display:
        pixels = np.atleast_2d(pixels)
        plt.imshow(image_mask, cmap="gray")
        for pixel in pixels:
            plt.scatter(*pixel[::-1], c="r")
        for pixel_set in processed_pixels:
            for p in pixel_set:
                plt.scatter(*p[::-1], c="g")
        plt.show()

    return np.array(processed_pixels)


def find_nearest_white_pixel(image, clip, num_options=10, display=False):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image

    clip_pixel = np.array([[clip["y"], clip["x"]]])

    masked_image = cv2.bitwise_and(image_gray, image_gray)

    centered_pixels = center_pixels_on_cable(
        masked_image[..., None], clip_pixel, num_options=num_options, display=display
    )

    nearest_pixels = centered_pixels[0]

    if display:
        vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        cv2.circle(vis, (clip["x"], clip["y"]), 5, (0, 0, 255), -1)
        for pixel in nearest_pixels:
            cv2.circle(vis, (pixel[1], pixel[0]), 5, (255, 255, 0), -1)

        cv2.imshow("Image with Nearest Cable Pixels", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return [(pixel[1], pixel[0]) for pixel in nearest_pixels]


def get_path_angle(path, N=5):
    """Estimate the cable's direction from the last N points."""
    if len(path) < N + 1:
        return None

    delta = np.array(path[-1]) - np.array(path[-N - 1])
    angle = np.arctan2(delta[1], delta[0])
    return angle


def get_perpendicular_orientation(b, a):

    b, a = np.array(b)[:2], np.array(a)[:2]
    tangent = a - b
    tangent /= np.linalg.norm(tangent)

    perp_vec = np.array([-tangent[1], tangent[0]])
    return np.arctan2(perp_vec[1], perp_vec[0])


def find_nearest_point(path, coordinate):
    path_array = np.array(path)
    idx = np.argmin(np.sum((path_array - coordinate) ** 2, axis=1))
    return path[idx], idx


def pick_target_on_path(img, path):
    img_display = img.copy()

    for x, y in path:
        cv2.circle(img_display, (x, y), 2, (255, 0, 0), -1)

    selected_point = select_target_point(img_display)
    if selected_point is not None:
        cv2.destroyAllWindows()
        return selected_point


# def get_world_coord_from_pixel_coord(
#     pixel_coord, cam_intrinsics, cam_extrinsics, image_shape=None, table_depth=0.8
# ):
#     pixel_coord = np.array(pixel_coord, dtype=np.float32)

#     if image_shape and (
#         cam_intrinsics.width != image_shape[1]
#         or cam_intrinsics.height != image_shape[0]
#     ):
#         scale_x = cam_intrinsics.width / image_shape[1]
#         scale_y = cam_intrinsics.height / image_shape[0]
#         pixel_coord[0] *= scale_x
#         pixel_coord[1] *= scale_y

#     pixel_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1.0])
#     point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(pixel_homogeneous) * table_depth

#     point_3d_world = (
#         cam_extrinsics.rotation.dot(point_3d_cam) + cam_extrinsics.translation
#     )

#     return point_3d_world


def pixel_to_3d_world(pixel_coord, depth, cam_intrinsics, cam_extrinsics):
    pixel_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1.0])
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(pixel_homogeneous) * depth
    point_3d_world = (
        cam_extrinsics.rotation.dot(point_3d_cam) + cam_extrinsics.translation
    )
    return point_3d_world


def plot_neighborhood_in_3d(
    pixel_coord, depth_map, cam_intrinsics, cam_extrinsics, neighborhood_radius=3
):
    x, y = int(pixel_coord[0]), int(pixel_coord[1])

    y_min, y_max = max(0, y - neighborhood_radius), min(
        depth_map.shape[0], y + neighborhood_radius + 1
    )
    x_min, x_max = max(0, x - neighborhood_radius), min(
        depth_map.shape[1], x + neighborhood_radius + 1
    )

    focus_region = depth_map[y_min:y_max, x_min:x_max].copy()

    mask = np.zeros_like(focus_region, dtype=np.uint8)
    cv2.circle(
        mask, (neighborhood_radius, neighborhood_radius), neighborhood_radius, 1, -1
    )

    focus_region *= mask

    ys, xs = np.where(mask > 0)
    xs = xs + x_min
    ys = ys + y_min

    points_3d = []
    for px, py in zip(xs, ys):
        d = depth_map[py, px]
        if d > 0:
            point_3d = pixel_to_3d_world((px, py), d, cam_intrinsics, cam_extrinsics)
            points_3d.append(point_3d)

    points_3d = np.array(points_3d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap="jet"
    )
    plt.colorbar(sc, label="Z (meters)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_zlabel("Z (meters)")
    ax.set_title(f"3D Points around ({x}, {y})")
    plt.show()


def get_world_coord_from_pixel_coord(
    pixel_coord,
    cam_intrinsics,
    cam_extrinsics,
    image_shape=None,
    table_depth=0.83815,  # 9
    depth_map=None,
    neighborhood_radius=10,
    display=False,
    is_clip=False,
):
    pixel_coord = np.array(pixel_coord, dtype=np.float32)

    if image_shape and (
        cam_intrinsics.width != image_shape[1]
        or cam_intrinsics.height != image_shape[0]
    ):
        scale_x = cam_intrinsics.width / image_shape[1]
        scale_y = cam_intrinsics.height / image_shape[0]
        pixel_coord[0] *= scale_x
        pixel_coord[1] *= scale_y

    if is_clip:
        depth = 0.8
    else:
        if depth_map is not None:
            x, y = int(pixel_coord[0]), int(pixel_coord[1])

            y_min, y_max = max(0, y - neighborhood_radius), min(
                depth_map.shape[0], y + neighborhood_radius + 1
            )
            x_min, x_max = max(0, x - neighborhood_radius), min(
                depth_map.shape[1], x + neighborhood_radius + 1
            )

            focus_region = depth_map[y_min:y_max, x_min:x_max].copy()

            mask = np.zeros_like(focus_region, dtype=np.uint8)
            cv2.circle(
                mask,
                (neighborhood_radius, neighborhood_radius),
                neighborhood_radius,
                1,
                -1,
            )

            focus_region *= mask
            valid_depths = focus_region[focus_region > 0]
            depth = np.min(valid_depths) if valid_depths.size > 0 else table_depth

            if display:
                plot_neighborhood_in_3d(
                    pixel_coord,
                    depth_map,
                    cam_intrinsics,
                    cam_extrinsics,
                    neighborhood_radius,
                )

        else:
            depth = table_depth

        depth = min(depth, table_depth)

    return pixel_to_3d_world(pixel_coord, depth, cam_intrinsics, cam_extrinsics)


def crop_board(img):

    point1, point2 = define_board_region(img)[0]

    x1, x2, y1, y2 = (
        min(point1[0], point2[0]),
        max(point1[0], point2[0]),
        min(point1[1], point2[1]),
        max(point1[1], point2[1]),
    )
    return img[y1:y2, x1:x2], (x1, y1), (x2, y2)


def crop_img(img, point1, point2):

    (x1, y1), (x2, y2) = point1, point2

    return img[y1:y2, x1:x2]


def green_color_segment(point_cloud, display=True):

    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    red = colors[:, 0]
    green = colors[:, 1]
    blue = colors[:, 2]

    green_dominance = green - (red + blue) / 2
    mask = (green_dominance > 0.07) & (green > 0.3)

    fixture_points = points[mask]

    if display:
        import open3d as o3d

        if fixture_points.shape[0] == 0:
            print("No green points detected.")
        else:
            print(f"Found {len(fixture_points)} green points.")

        segmented_pcd = o3d.geometry.PointCloud()
        segmented_pcd.points = o3d.utility.Vector3dVector(fixture_points)
        segmented_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        o3d.visualization.draw_geometries([segmented_pcd])

    return fixture_points


def sort_cable_points(cable_points, start_point, end_point):
    """
    Sorts the given cable points into the correct order to follow the trajectory of the cable from one end to the other
    """


def rescale_intrinsics(K, scale_factor):
    K_rescaled = K.copy()
    K_rescaled[0, 0] *= scale_factor
    K_rescaled[1, 1] *= scale_factor
    K_rescaled[0, 2] *= scale_factor
    K_rescaled[1, 2] *= scale_factor
    return K_rescaled


def mask_image_outside_roi(image, roi):
    x, y, w, h = roi
    masked_image = image.copy()
    masked_image[:y, :] = 255
    masked_image[y + h :, :] = 255
    masked_image[:, :x] = 255
    masked_image[:, x + w :] = 255
    return masked_image


def draw_rectangle(event, x, y, flags, param):
    """
    Allows user to draw a rectangle to define the board region.
    """
    img, board_rect = param["img"], param["board_rect"]

    if event == cv2.EVENT_LBUTTONDOWN:
        param["rect_start"] = (x, y)
        param["drawing"] = True

    elif event == cv2.EVENT_MOUSEMOVE and param["drawing"]:
        temp_img = img.copy()
        cv2.rectangle(temp_img, param["rect_start"], (x, y), (0, 255, 0), 2)
        cv2.imshow("Define Board Area", cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR))

    elif event == cv2.EVENT_LBUTTONUP:
        param["rect_end"] = (x, y)
        param["drawing"] = False
        board_rect.append((param["rect_start"], param["rect_end"]))
        cv2.rectangle(img, param["rect_start"], param["rect_end"], (0, 255, 0), 2)
        cv2.imshow("Define Board Area", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def click_event(event, x, y, flags, param):
    """
    Maps the clicked point on the resized image back to the original resolution.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        param["u"], param["v"] = int(x / SCALE_FACTOR), int(y / SCALE_FACTOR)


def define_board_region(image):
    """
    Displays the image and allows user to draw a rectangle for the board region.
    """
    board_rect = []
    param = {
        "img": image.copy(),
        "drawing": False,
        "rect_start": None,
        "rect_end": None,
        "board_rect": board_rect,
    }
    cv2.imshow("Define Board Area", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Define Board Area", draw_rectangle, param)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Press 's' to save
            break
        elif key == 27:  # Press 'ESC' to cancel
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return board_rect


def select_target_point(image, rule="start"):
    """
    Displays the image and allows user to select a target point.
    """

    params = {"u": None, "v": None}
    cv2.imshow(f"Select {rule} Point", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback(f"Select {rule} Point", click_event, param=params)

    while params["u"] is None or params["v"] is None:
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return [params["u"], params["v"]]


def distance(p1: list, p2: list):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

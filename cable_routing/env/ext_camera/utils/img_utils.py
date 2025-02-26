import re
import cv2
import numpy as np

SCALE_FACTOR = 1.0


def get_perpendicular_ori(b, a):
    b, a = np.array(b)[:2], np.array(a)[:2]
    tangent = a - b
    yaw = np.arctan2(tangent[1], tangent[0])
    return (yaw + np.pi / 2) % (2 * np.pi)


def find_nearest_point(path, coordinate):
    path_array = np.array(path)
    idx = np.argmin(np.sum((path_array - coordinate) ** 2, axis=1))
    return path[idx], idx


def pick_target_on_path(img, path):
    img_display = img.copy()

    for x, y in path:
        cv2.circle(img_display, (x, y), 2, (0, 255, 0), -1)

    selected_point = select_target_point(img_display)
    if selected_point is not None:
        cv2.destroyAllWindows()
        return selected_point


def get_world_coord_from_pixel_coord(
    pixel_coord, cam_intrinsics, cam_extrinsics, image_shape=None, table_depth=0.8
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

    pixel_homogeneous = np.array([pixel_coord[0], pixel_coord[1], 1.0])
    point_3d_cam = np.linalg.inv(cam_intrinsics._K).dot(pixel_homogeneous) * table_depth

    point_3d_world = (
        cam_extrinsics.rotation.dot(point_3d_cam) + cam_extrinsics.translation
    )

    return point_3d_world


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


def select_target_point(image):
    """
    Displays the image and allows user to select a target point.
    """
    print("Select a target point.")

    params = {"u": None, "v": None}
    cv2.imshow("Select Target Point", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Select Target Point", click_event, param=params)

    while params["u"] is None or params["v"] is None:
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return [params["u"], params["v"]]

import cv2
import numpy as np

SCALE_FACTOR = 1.0


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
    params = {"u": None, "v": None}
    cv2.imshow("Select Target Point", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Select Target Point", click_event, param=params)

    while params["u"] is None or params["v"] is None:
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    return [params["u"], params["v"]]

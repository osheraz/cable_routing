import cv2
import numpy as np

SCALE_FACTOR = 0.25


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

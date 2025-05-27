import cv2
import os
import rospy
from cable_routing.env.ext_camera.ros.zed_camera import ZedCameraSubscriber

def main():
    cam_name = "zed"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(script_dir, "test_calibration")
    os.makedirs(save_dir, exist_ok=True)

    rospy.init_node("zed_calibration_checker", anonymous=True)

    # Initialize ZED Camera Subscriber
    zed_cam = ZedCameraSubscriber()

    rospy.loginfo("Waiting for images from ZED camera...")
    while zed_cam.rgb_image is None or zed_cam.depth_image is None:
        rospy.sleep(0.1)

    # Pull reference image with OpenCV
    reference_img = cv2.imread("/home/osheraz/cable_routing/cable_routing/configs/board/board_setup.png")
    

    cv2.namedWindow("Board Orientation Setup")
    # Stream and overlap images continuously
    while True:
        # Get current frame from ZED
        zed_img = zed_cam.get_rgb()
        
        if zed_img is not None:
            # Resize ZED image to match reference dimensions
            zed_img = cv2.resize(zed_img, (reference_img.shape[1], reference_img.shape[0]))
            
            # Create overlapped image with reference image more transparent
            alpha = 0.3
            beta = 0.7
            overlapped = cv2.addWeighted(reference_img, alpha, zed_img, beta, 0)
            
            # Display overlapped result
            cv2.imshow("Overlapped Images", overlapped)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
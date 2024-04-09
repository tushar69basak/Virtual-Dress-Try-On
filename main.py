import cv2
import numpy as np
from background_removal import remove_background
from dress_blending import blend_dress_with_user

def main(user_image_path, dress_image_path, roi_start_point):
    # Load the user image
    user_img = cv2.imread(user_image_path)
    if user_img is None:
        print(f"Error: Failed to load user image from {user_image_path}")
        return
    
    # Remove background from the dress image
    dress_img_without_bg = remove_background(dress_image_path)
    
    # Blend the dress onto the user image within the specified ROI
    final_img = blend_dress_with_user(user_img, dress_img_without_bg, roi_start_point)
    
    # Display the final image
    cv2.imshow("Final Image", final_img)
    cv2.waitKey(0)  # Wait for a key press before closing the image window
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    # Set the paths to your user image and dress image
    user_image_path = 'images/user1.jpg'
    dress_image_path = 'images/dress1.jpg'

    # Define the top-left corner (x, y) of the ROI where the dress should be applied
    roi_start_point = (80, 121)  # Adjust this based on your specific requirements

    # Call the main function to blend the dress with the user image
    main(user_image_path, dress_image_path, roi_start_point)


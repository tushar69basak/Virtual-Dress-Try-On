import cv2

def blend_dress_with_user(user_img, dress_img, roi_start_point):
    """
    Blend the dress onto the user image within the defined ROI.
    roi_start_point: A tuple (x, y) indicating the top left point of the ROI on the user image.
    """
    dress_height, dress_width, _ = dress_img.shape
    x, y = roi_start_point
    
    # Define the region of interest (ROI) on the user image
    roi = user_img[y:y+dress_height, x:x+dress_width]
    
    # Create a mask for the dress image (using the dress image without background)
    dress_gray = cv2.cvtColor(dress_img, cv2.COLOR_BGR2GRAY)
    _, dress_mask = cv2.threshold(dress_gray, 10, 255, cv2.THRESH_BINARY)
    dress_mask_inv = cv2.bitwise_not(dress_mask)
    
    # Mask out the dress region from the user image
    user_bg = cv2.bitwise_and(roi, roi, mask=dress_mask_inv)
    
    # Extract the dress region from the dress image
    dress_fg = cv2.bitwise_and(dress_img, dress_img, mask=dress_mask)
    
    # Combine the dress region with the background region
    blended_roi = cv2.add(user_bg, dress_fg)
    
    # Replace the ROI in the user image with the blended region
    user_img[y:y+dress_height, x:x+dress_width] = blended_roi
    
    return user_img

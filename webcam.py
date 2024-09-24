import cv2
import numpy as np

def preprocess_and_detect_plate(image):
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Perform edge detection (Canny)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Step 4: Find contours to detect the rectangular plate card
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_contour = None
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for rectangles (4-sided polygons)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if 1000 < area < 10000:  # Adjust area threshold for size of the plate card
                plate_contour = approx
                break

    # If a plate card is detected
    if plate_contour is not None:
        # Step 5: Draw the contour on the original image (for visualization)
        cv2.drawContours(image, [plate_contour], -1, (0, 255, 0), 2)
        
        # Step 6: Crop the Region of Interest (ROI) using the bounding rectangle
        x, y, w, h = cv2.boundingRect(plate_contour)
        roi = image[y:y+h, x:x+w]
        
        # Step 7: Convert the ROI to grayscale for circle detection
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Step 8: Apply a blur before detecting circles
        roi_blur = cv2.medianBlur(roi_gray, 5)
        
        # Step 9: Detect circles using HoughCircles (adjust radius to catch larger circles)
        circles = cv2.HoughCircles(
            roi_blur, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=20, 
            param1=100, 
            param2=30, 
            minRadius=10,  # Adjust this to fit the smallest circle
            maxRadius=60   # Increase this to capture the larger red circle
        )
        
        # Step 10: If circles are detected, draw them
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (cx, cy, r) in circles:
                # Draw the outer circle
                cv2.circle(roi, (cx, cy), r, (0, 255, 0), 3)
                # Draw the center of the circle
                cv2.rectangle(roi, (cx - 2, cy - 2), (cx + 2, cy + 2), (0, 128, 255), -1)
        
        # Show the ROI with circles detected
        cv2.imshow("Detected Circles on Plate", roi)
    
    return image

# Webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Pre-process the frame and detect plate card and circles
    result_frame = preprocess_and_detect_plate(frame)
    
    # Display the result
    cv2.imshow("Plate Card Detection", result_frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

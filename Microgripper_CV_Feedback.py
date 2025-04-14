#!/usr/bin/env python

#general imports
import math
import numpy as np
import time
import sys

#ROS imports
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion  # For robot position
from mag_msgs.msg import DipoleGradientStamped
from std_msgs.msg import Float64MultiArray
import tf.transformations as tf_transfrormations
from sensor_msgs.msg import Image 

#imaging imports
from cv_bridge import CvBridge
import cv2 # pip install opencv-python

# Import saved parameters
try:
    from CVParameters3 import MICROGRIPPER_PARAMS
    USE_SAVED_PARAMS = True
    print("Using saved parameters from CVParameters1.py")
except ImportError:
    USE_SAVED_PARAMS = False
    print("No saved parameters found, using default parameters")

# New global variables for timestamp-based velocity estimation
last_timestamps = []
last_positions = []
last_angles = []
skipped_counter = 0
q,w,e,r,t,y = 0, 0, 0, 0, 0, 0  # Initialize counters for debugging 
PixelToMM = 0.003187 # Conversion factor from pixels to mm (assuming 1600 pixels in width)

# Apply scale correction factor to match simulation scale
# This value can be adjusted if the ROS image scale doesn't match the simulation scale
scale_correction_factor = 1.0  # Adjust this value as needed to match simulation scale
PixelToMM = PixelToMM * scale_correction_factor

# Global variable to track the time of the last received image
last_image_time = time.time()
image_timeout = 30000.0  # 3 seconds timeout
# Function to check for image timeout
def check_image_timeout():
    if time.time() - last_image_time > image_timeout:
        rospy.logerr("No images received for {} seconds. Shutting down for safety.".format(image_timeout))
        rospy.signal_shutdown("Image timeout - no vision feedback")
        sys.exit(0)

# Timer callback to check for timeout
def timeout_callback(event):
    check_image_timeout()

# New function to predict next position and angle based on weighted average of changes
def predict_next_values(centroids, angles, timestamp=None):
    """
    Predicts the next centroid and angle values using more robust methods
    """
    # Need at least 2 measurements to calculate changes
    if len(centroids) < 2 or len(angles) < 2:
        return centroids[-1] if centroids else None, angles[-1] if angles else None
    
    # For angles, convert to sin/cos representation to avoid wraparound issues
    angles_rad = np.radians(angles)
    sin_vals = np.sin(angles_rad)
    cos_vals = np.cos(angles_rad)
    
    # Calculate weighted average of recent values (not differences)
    # This creates a more stable prediction that's less affected by noise
    weights = np.exp(np.linspace(0, 2, len(angles)))  # Exponential weights
    weights = weights / np.sum(weights)  # Normalize
    
    # Use slightly more weight on older values to stabilize prediction
    weighted_sin = np.sum(weights * sin_vals)
    weighted_cos = np.sum(weights * cos_vals)
    
    # Convert back to angle
    predicted_angle = np.degrees(np.arctan2(weighted_sin, weighted_cos)) % 360
    
    # For position, keep the existing approach but use weighted average
    pos_weights = np.exp(np.linspace(0, 1, len(centroids)))
    pos_weights = pos_weights / np.sum(pos_weights)
    
    # Weighted centroid (more stable than velocity approach)
    weighted_centroid = np.zeros(2)
    for i in range(len(centroids)):
        weighted_centroid += pos_weights[i] * np.array(centroids[i])
    
    # Apply a small velocity component to the prediction
    if len(centroids) >= 3:
        recent_velocity = np.array(centroids[-1]) - np.array(centroids[-2])
        predicted_centroid = weighted_centroid + recent_velocity * 0.2  # Reduced influence
    else:
        predicted_centroid = weighted_centroid
        
    return predicted_centroid, predicted_angle

# Add this helper function at the top
def normalize_angle_degrees(angle):
    return (angle % 360 + 360) % 360  # Ensures angle is always 0-360°

def angle_difference(a1, a2):
    a1 = normalize_angle_degrees(a1)
    a2 = normalize_angle_degrees(a2)
    # Returns smallest angle between two angles (0-180°)
    diff = abs((a1 - a2) % 360)
    return min(diff, 360 - diff)

# --- Define HSV Color Ranges ---
# You MUST tune these ranges for your specific fiducials and lighting conditions
# Use a tool like HSV Color Picker (many online) or OpenCV code to find these values

# Example: Red (Note: Red wraps around 0/180 in HSV)
lower_red1 = np.array([0, 200, 200])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 200, 200])
upper_red2 = np.array([180, 255, 255])

# Minimum contour area to filter noise
MIN_FIDUCIAL_AREA = 100 # Adjust as needed

def find_colored_fiducials(image):
    """
    Finds the two largest red fiducials in the image.
    Returns a list containing tuples: [(center1, 'red', contour1), (center2, 'red', contour2)]
    or fewer items if not enough are found.
    """
    found_fiducials = []
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Process Red Fiducials ---
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2) # Combine both red ranges

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find all red contours above the minimum area
    valid_red_contours = []
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > MIN_FIDUCIAL_AREA:
             # Optional: Add shape filtering here (e.g., circularity)
            valid_red_contours.append({'contour': contour, 'area': area})

    # Sort the valid contours by area in descending order
    valid_red_contours.sort(key=lambda x: x['area'], reverse=True)

    # Take the top two largest contours if they exist
    num_found = min(len(valid_red_contours), 2)
    for i in range(num_found):
        contour_info = valid_red_contours[i]
        contour = contour_info['contour']
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            found_fiducials.append(((cx, cy), 'red', contour)) # Store center, color, and contour

    # --- Return results ---
    # Sorting is less critical now but doesn't hurt
    found_fiducials.sort(key=lambda x: x[0][0]) # Sort by x-coordinate for consistency
    return found_fiducials

def find_contour(contours,predicted_centroid=None):
    global skipped_counter
    NUM_LARGEST = 6 # number of the largest contours to keep/check if robot
    max_position_deviation = 200  # pixels
    if USE_SAVED_PARAMS:   
        # Update contour filtering parameters
        MAX_AREA = MICROGRIPPER_PARAMS['max_area']
        MIN_AREA = MICROGRIPPER_PARAMS['min_area']
        hull_epsilon = MICROGRIPPER_PARAMS['hull_epsilon']
        min_hull_points = MICROGRIPPER_PARAMS['min_hull_points']
        max_hull_points = MICROGRIPPER_PARAMS['max_hull_points']
        aspect_ratio = MICROGRIPPER_PARAMS['aspect_ratio']
    else:
        # Default contour filtering parameters
        MAX_AREA = 100000
        MIN_AREA = 40000
        hull_epsilon = 0.013
        min_hull_points = 3
        max_hull_points = 15
        aspect_ratio = 1.75

    for i in range(0, min(len(contours),NUM_LARGEST)):
        max_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        contour = contours[max_idx]
        contours.pop(max_idx)
        hull = cv2.convexHull(contour)
        # Calculate the convex hull of the contour and simplify it to a hexagon
        # Use the hull_epsilon parameter to control the simplification
        simple_hull = cv2.approxPolyDP(hull, hull_epsilon * cv2.arcLength(hull, True), True)
        if len(simple_hull) >= min_hull_points and len(simple_hull) <= max_hull_points:
            rect = cv2.minAreaRect(contour)
            (cx, cy), (width, height), angle = rect  
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            # Ensure width is the longer dimension
            if width < height:
                width, height = height, width
            else:
                angle = angle-90 
            
            area = cv2.contourArea(contour)
            #print(area)
            # Apply aspect ratio and area constraints
            if width < aspect_ratio*height and area < MAX_AREA and area > MIN_AREA:
                # Get prediction for this frame based on past measurements
                if predicted_centroid is not None:# and predicted_angle is not None:
                    # Calculate distance to predicted position
                    pred_cx, pred_cy = predicted_centroid
                    distance_to_prediction = math.sqrt((cx - pred_cx)**2 + (cy - pred_cy)**2)
                    # Check if current measurement is too far from prediction
                    if distance_to_prediction > max_position_deviation:
                        print(f"Position outlier rejected: ({cx:.1f}, {cy:.1f}) - too far from prediction: {distance_to_prediction:.2f}px")
                        skipped_counter +=1
                        return (None,None), None  
                return (cx, cy), simple_hull    
    return (None,None), None  


def microgripperDetection(cvImage, timestamp, openColor, centroids, angle_vectors, openlengths, timestamps):
    global skipped_counter, PixelToMM
    # Define maximum allowed deviation from prediction
    max_angle_deviation = 5  # degrees
   
    SEARCH_AREA = 175
    cropping = False
    area_threshold = 10.0  # 20% threshold for contour area change
    
    # Position and angle outlier rejection thresholds
    position_threshold = 100.0 # Standard deviations
    angle_threshold = 100.0     # Standard deviations
    
    start_time = time.time()
    
    # Convert image to grayscale
    frame = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
                    
    # draw cordinate system
    x = cvImage.shape[1] / 2  # Adjust x to have 0,0 at the center of the image
    y = cvImage.shape[0] / 2  # Adjust y to have 0,0 at the center of the image
    cv2.arrowedLine(cvImage, (int(x), int(y)), (int(x+(1/PixelToMM)/3), int(y)), (225, 0, 0), 2)  # X-axis
    cv2.arrowedLine(cvImage, (int(x), int(y)), (int(x), int(y-1/PixelToMM/3)), (225, 0, 0), 2)  # Y-axis
    cv2.putText(cvImage, "X", (int(x+15), int(y-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 1)
    cv2.putText(cvImage, "Y", (int(x+50), int(y-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 1)
    cv2.circle(cvImage, (int(x), int(y)), radius=3, color=(225, 0, 0), thickness=-1)  # Origin point

   # Use saved parameters if available, otherwise use defaults
    if USE_SAVED_PARAMS:
        # Apply bilateral filter if enabled
        if MICROGRIPPER_PARAMS['use_bilateral_filter']:
            frame = cv2.bilateralFilter(
                frame,
                MICROGRIPPER_PARAMS['bilateral_d'],
                MICROGRIPPER_PARAMS['bilateral_sigma_color'],
                MICROGRIPPER_PARAMS['bilateral_sigma_space']
            )
        
        # Edge detection
        if MICROGRIPPER_PARAMS['use_canny']:
            edges = cv2.Canny(
                frame, 
                MICROGRIPPER_PARAMS['canny_threshold1'],
                MICROGRIPPER_PARAMS['canny_threshold2']
            )
        else:
            # Adaptive thresholding
            block_size = MICROGRIPPER_PARAMS['adaptive_block_size']
            if block_size % 2 == 0:  # Must be odd
                block_size += 1
                
            edges = cv2.adaptiveThreshold(
                frame,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                MICROGRIPPER_PARAMS['adaptive_constant']
            )
        
        # Morphological operations
        if MICROGRIPPER_PARAMS['erode_iterations'] > 0:
            edges = cv2.erode(edges, MICROGRIPPER_PARAMS['kernel'], iterations=MICROGRIPPER_PARAMS['erode_iterations'])
        
        if MICROGRIPPER_PARAMS['dilate1_iterations'] > 0:
            edges = cv2.dilate(edges, MICROGRIPPER_PARAMS['kernel'], iterations=MICROGRIPPER_PARAMS['dilate1_iterations'])
        
        if MICROGRIPPER_PARAMS['erode2_iterations'] > 0:
            edges = cv2.erode(edges, MICROGRIPPER_PARAMS['kernel'], iterations=MICROGRIPPER_PARAMS['erode2_iterations'])
        
        if MICROGRIPPER_PARAMS['dilate2_iterations'] > 0:
            edges = cv2.dilate(edges, MICROGRIPPER_PARAMS['kernel'], iterations=MICROGRIPPER_PARAMS['dilate2_iterations'])
    else:
        # Use default image processing parameters
        kernel = np.ones((3,3))
        blurred = cv2.bilateralFilter(frame, 5, 10, 10)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -6)
        edges = cv2.erode(edges, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=5)
        edges = cv2.erode(edges, kernel, iterations=3)
        edges = cv2.dilate(edges, kernel, iterations=4) 
     
    # --- Find Colored Fiducials ---
    fiducials = find_colored_fiducials(cvImage) # Use the original color image

    # --- Visualization and Main Axis Calculation (if 2 fiducials found) ---
    centroid_point = None # Will be calculated later if gripper is found

    if len(fiducials) == 2:
        (center1, color1, contour1) = fiducials[0]
        (center2, color2, contour2) = fiducials[1]

        # Calculate main axis based on fiducial centers
        f_center1 = np.array(center1)
        f_center2 = np.array(center2)

        # Midpoint between fiducials
        fiducials_mid_point = (f_center1 + f_center2) / 2
        openlength = np.linalg.norm(np.array(center1) - np.array(center2)) 
        openlengths.append(openlength)
        if len(openlengths) >= 10:
            openlengths.pop(0)

         

        # Add text showing the opening distance
        cv2.putText(cvImage, f"Opening: {openlength*PixelToMM*1000:.1f}um", (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)           
        # Draw fiducials and main axis
        cv2.circle(cvImage, center1, 7, (0, 255, 255), -1) # Yellow dot
        cv2.drawContours(cvImage, [contour1], -1, (0, 255, 255), 2) # Draw contour

        cv2.circle(cvImage, center2, 7, (0, 255, 255), -1) # Yellow dot
        cv2.drawContours(cvImage, [contour2], -1, (0, 255, 255), 2) # Draw contour

        # cv2.line(cvImage, center1, center2, (0, 255, 255), 2) # Yellow line for main axis
        # cv2.putText(cvImage, "Fiducial Axis", tuple(map(int, fiducials_mid_point + np.array([-50, -15]))),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    
    
    # Display the edges for debugging
    #cv2.imshow("Edges", edges)
    
    # Create crop mask for region of interest
    crop_mask = np.zeros_like(edges)
    
    predicted_centroid, predicted_angle = predict_next_values(centroids, angle_vectors) 
    # If we have predictions, use them for cropping to improve processing speed and accuracy
    if len(centroids) >= 5 and cropping:
        if predicted_centroid is not None:
            cx, cy = predicted_centroid
            cv2.rectangle(crop_mask, (int(cx-SEARCH_AREA), int(cy-SEARCH_AREA)), 
                         (int(cx+2*SEARCH_AREA), int(cy+2*SEARCH_AREA)), 255, thickness=-1)
            edges = cv2.bitwise_and(edges, crop_mask)
    elif cropping and centroids:
        # Use the last known position if no prediction is available
        cx, cy = centroids[-1]
        cv2.rectangle(crop_mask, (int(cx-SEARCH_AREA), int(cy-SEARCH_AREA)), 
                     (int(cx+2*SEARCH_AREA), int(cy+2*SEARCH_AREA)), 255, thickness=-1)
        edges = cv2.bitwise_and(edges, crop_mask)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if contours is not None:
        (cx,cy), simple_hull = find_contour(contours)              
        if (skipped_counter > 10):
            if len(centroids) > 3: centroids.pop(0)
            if len(angle_vectors) > 3 :angle_vectors.pop(0)
        if len(centroids) > 10:
            centroids.pop(0)
        if len(angle_vectors)>10:
            angle_vectors.pop(0)
        if simple_hull is None:
            openColor = (0, 0, 255)  # red
        else:
            # Accept the detection
            openColor = (0, 255, 0)  # green
            centroids.append((cx,cy))
            timestamps.append(timestamp)        
            # Draw the contour outline
            cv2.drawContours(cvImage, [simple_hull.astype(np.int32)], 0, openColor, 2)

                
            # Draw the centroid of the contour
            centroidtuple = tuple(map(int, [cx,cy]))
            cv2.circle(cvImage, centroidtuple, radius=7, color=(0, 255, 0), thickness=-1)  # Green dot                    
            if len(fiducials) == 2:
                angle_color = [0, 255, 0]  # green
                # Use the centroid of the contour (cx, cy) as the center point for our baseline
                centroid_point = np.array([cx, cy])
                direction = fiducials_mid_point - centroid_point
                
                # Calculate angle from the direction vector
                angle_vector = np.arctan2(direction[1], direction[0])
                angle_vector = normalize_angle_degrees(np.degrees(angle_vector))
                if len(angle_vectors) > 5:              
                    angle_to_prediction = angle_difference(angle_vector, predicted_angle) if predicted_angle is not None else 0
                    # Calculate smallest angle difference accounting for wraparound
                    angle_diff = angle_difference(angle_vector, angle_vectors[-1]) if angle_vectors[-1] is not None else 0
                    # Check if angle is within acceptable range
                    if angle_diff > max_angle_deviation and angle_to_prediction > max_angle_deviation:
                        # Reject this angle as an outlier
                        print(f"Angle rejected: {angle_vector:.1f}°, Diff: {angle_diff:.2f}°, skipped:{skipped_counter}")
                        if not angle_vectors:  # Check if list is empty
                            angle_vectors.append(angle_vector)  # Use current value as fallback
                        angle_color = [0,0,255] # red
                        skipped_counter +=1
                        # return cvImage, openColor, centroids, angles, openlengths, timestamps
                        if angle_diff > max_angle_deviation:
                            print(f"Angle deviation: {angle_vector:.1f}°, previous: {angle_vectors[-1]:.2f}°")
                        if angle_to_prediction > max_angle_deviation:
                            print(f"Angle prediction deviation: {angle_vector:.1f}°, predicted: {predicted_angle:.2f}°")
                    else:
                        angle_vectors.append(angle_vector)
                        skipped_counter = 0
                else:
                    angle_vectors.append(angle_vector)
                    # Draw the angle vector
                cv2.arrowedLine(cvImage, centroidtuple, (int(centroidtuple[0] + 50 * np.cos(np.radians(angle_vector))),
                                int(centroidtuple[1] + 50 * np.sin(np.radians(angle_vectors[-1])))), angle_color, 2)
            else:
                print(f"Fiducials not found")
                skipped_counter +=1
                if angle_vectors:  # Check if list is not empty
                    angle_vectors.append(angle_vectors[-1])
                else:
                    angle_vectors.append(angle_vector)            
        if simple_hull is None and len(fiducials) > 0:
            # If only one fiducial is found, we can still use it to estimate the position
            # but we won't have a valid angle vector
            cx, cy = fiducials[0][0]
            centroids.append((cx, cy))
            timestamps.append(timestamp)
            cv2.circle(cvImage, (int(cx), int(cy)), radius=7, color=(0, 255, 0), thickness=-1)
    else:
        cv2.imshow("No contours found", cvImage)
        print("No robot contours found.")
        
    # Visualize prediction if available
    if predicted_centroid is not None:
        pred_cx, pred_cy = predicted_centroid
        # Draw the prediction point
        cv2.circle(cvImage, (int(pred_cx), int(pred_cy)), radius=8, color=(255, 0, 255), thickness=2)
        # Draw line from prediction to actual position
        if len(centroids) > 0:
            actual_cx, actual_cy = centroids[-1]
            cv2.line(cvImage, (int(pred_cx), int(pred_cy)), (int(actual_cx), int(actual_cy)), 
                    color=(255, 0, 255), thickness=1)
    
    end_time = time.time()
    processing_time = (end_time-start_time)*1000
    
    # Display processing time on the image
    cv2.putText(cvImage, f"Processing: {processing_time:.1f} ms", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
    # If using saved parameters, show which parameters are being used
    if USE_SAVED_PARAMS:
        cv2.putText(cvImage, "Using tuned parameters", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return cvImage, openColor, centroids, angle_vectors, openlengths, timestamps

def publish_pose(publisher, x, y, theta, opening, timestamp=None):
        try:
            # print(f"x,y:({x},{y})")

            # Convert to mm 
            x = x * PixelToMM
            y = y * PixelToMM
            opening = opening * PixelToMM*1000 # opening in um

            # convert to radians
            # print("theta", theta)
            theta = math.radians(-theta-90)
            
            # Convert to quaternion  
            z = 0
            qx = 0
            qy = 0
            
            # Create the quaternion
            qz = math.sin(theta/2.0)
            qw = math.cos(theta/2.0)
            
            # Create message
            pose_msg = Float64MultiArray()
            pose_msg.data = [x, y, z, qx, qy, qz, qw, opening, timestamp.to_sec()]
            # print("pose_msg", pose_msg.data)
            # Publish the message
            publisher.publish(pose_msg)
        except Exception as e:
            print(f"Error publishing pose: {e}")

def image_callback(msg):
    bridge = CvBridge()
    global centroids, angle_vectors, publisher, openlengths, last_image_time, timestamps
    
    # Update the last_image_time whenever we receive an image
    last_image_time = time.time()
    
    openColor = (0,255,0)  # green
    timestamp = msg.header.stamp
    timestamp_sec = timestamp.to_sec()
    
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except:
        rospy.logerr("Image could not be read")
        return
    
    processed_img, openColor, centroids, angle_vectors, openlengths, timestamps = microgripperDetection(cv_image, timestamp_sec, openColor, centroids, angle_vectors, openlengths, timestamps)
    
    if (processed_img is not None and centroids):
        x = centroids[-1][0] - cv_image.shape[1] / 2  # Adjust x to have 0,0 at the center of the image
        y = -centroids[-1][1] + cv_image.shape[0] / 2  # Adjust y to have 0,0 at the center of the image
        publish_pose(publisher, x, y, angle_vectors[-1], openlengths[-1], timestamp)
        cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
        if cv2.waitKey(3) & 0xFF == ord(' '):
            cv2.destroyAllWindows()
    # elif processed_img is not None:
    #     cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))


def main():
    rospy.init_node('image_processor_node', anonymous=True)
    
    # Initialize global variables
    global centroids, angle_vectors, publisher, openlengths, last_image_time, timestamps
    openlengths = [0]
    centroids = []
    angle_vectors = []
    timestamps = []
    last_image_time = time.time()
    # Set up the timeout timer to check every 0.5 seconds
    rospy.Timer(rospy.Duration(0.5), timeout_callback)
    
    rospy.Subscriber("/camera/basler_camera_1/image_raw", Image, image_callback)
    publisher = rospy.Publisher('/vision_feedback/pose_estimation', Float64MultiArray, queue_size=10)
    
    rospy.loginfo("MicroGripper Vision Feedback started. Will quit if no images received for {} seconds.".format(image_timeout))
    
    rospy.spin()   

if __name__ == "__main__":
    main()


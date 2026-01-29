#!/usr/bin/env python

#general imports
import math
import numpy as np
import time
import sys
from collections import deque
import threading

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
    print("Using saved parameters from CVParameters3.py")
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

# --- GPU / OpenCL Acceleration Setup ---
GPU_ENABLED = False  # will be set in setup_gpu()
GPU_RUNTIME_OK = True  # new flag

try:
    import pyopencl as cl
    _PYOPENCL_AVAILABLE = True
except ImportError:
    _PYOPENCL_AVAILABLE = False

def _diagnose_opencv_opencl():
    """Log minimal OpenCV build info related to OpenCL."""
    try:
        info = cv2.getBuildInformation()
        lines = [l for l in info.split('\n') if 'OpenCL' in l or 'ocl' in l]
        rospy.loginfo("OpenCV OpenCL build lines:\n" + "\n".join(lines[:15]))
    except Exception as e:
        rospy.logwarn(f"Could not extract OpenCV build info: {e}")

def _diagnose_platforms():
    if not _PYOPENCL_AVAILABLE:
        rospy.logwarn("pyopencl not installed (pip install pyopencl) -> deeper diagnostics skipped.")
        return
    try:
        plats = cl.get_platforms()
        if not plats:
            rospy.logwarn("No OpenCL platforms found by pyopencl.")
            return
        for p in plats:
            rospy.loginfo(f"OpenCL Platform: {p.name} | Vendor: {p.vendor} | Version: {p.version}")
            for d in p.get_devices():
                rospy.loginfo(f"  Device: {d.name} | Type: {cl.device_type.to_string(d.type)} | Version: {d.version}")
    except Exception as e:
        rospy.logwarn(f"Error enumerating OpenCL platforms: {e}")


def setup_gpu():
    global GPU_ENABLED, GPU_RUNTIME_OK
    use_gpu = rospy.get_param('~use_gpu', True)
    have = cv2.ocl.haveOpenCL()
    if use_gpu and have:
        cv2.ocl.setUseOpenCL(True)
        GPU_ENABLED = cv2.ocl.useOpenCL()
        if GPU_ENABLED:
            rospy.loginfo("OpenCL GPU acceleration enabled (OpenCV T-API).")
        else:
            rospy.logwarn("OpenCL reported present but could not activate cv2.ocl.useOpenCL().")
    else:
        GPU_ENABLED = False
        rospy.loginfo("GPU acceleration disabled or OpenCL unavailable (using CPU).")
    # Conditional diagnostics
    if (not GPU_ENABLED and use_gpu):
        rospy.loginfo("Running OpenCL diagnostics...")
        _diagnose_opencv_opencl()
        _diagnose_platforms()

    # Runtime probe
    if GPU_ENABLED:
        GPU_RUNTIME_OK = _probe_opencl_runtime()
        if not GPU_RUNTIME_OK:
            rospy.logwarn("Disabling GPU due to OpenCL runtime failure (falling back to CPU).")
            GPU_ENABLED = False

def _probe_opencl_runtime():
    """Try a tiny OpenCL operation; return True if success else False."""
    try:
        test = np.zeros((8,8), np.uint8)
        u = cv2.UMat(test)
        _ = cv2.GaussianBlur(u, (3,3), 0)  # simple kernel compile
        return True
    except Exception as e:
        rospy.logwarn(f"OpenCL runtime probe failed: {e}")
        return False

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
lower_red1 = np.array([0, 10, 10])
upper_red1 = np.array([0, 255, 255])
lower_red2 = np.array([145, 100, 100])
upper_red2 = np.array([170, 255, 255])
# lower_red1 = np.array([12, 150, 150]) # yellow fiducials for imaging in DMEM
# upper_red1 = np.array([35, 255, 255]) # yellow fiducials for imaging in DMEM
# lower_red2 = np.array([180, 255, 255])# yellow fiducials for imaging in DMEM
# upper_red2 = np.array([180, 255, 255]) # yellow fiducials for imaging in DMEM


# Minimum contour area to filter noise
MIN_FIDUCIAL_AREA = 50 # Adjust as needed 
def find_colored_fiducials(image):
    """
    Finds the two largest red fiducials in the image.
    Returns a list containing tuples: [(center1, 'red', contour1), (center2, 'red', contour2)]
    or fewer items if not enough are found.
    """
    found_fiducials = []
    # Use GPU path for HSV if available
    if GPU_ENABLED:
        hsv_image = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2HSV).get()
    else:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Process Red Fiducials ---
    # mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red = cv2.inRange(hsv_image, lower_red2, upper_red2)
    # mask_red = cv2.bitwise_or(mask_red1, mask_red2) # Combine both red ranges

    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("Red Mask", mask_red)
    # cv2.waitKey(1)
    # Find contours in the red mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find all red contours above the minimum area
    valid_red_contours = []
    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > MIN_FIDUCIAL_AREA:
             # Optional: Add shape filtering here (e.g., circularity)
            valid_red_contours.append({'contour': contour, 'area': area})

    # Take the top two largest contours if they exist
    largest_red_contours = [None, None]  # To store the two largest contours
    largest_area = 0
    secnd_largest_area = 0
    for contour in valid_red_contours:
        area = contour['area']
        if area > largest_area:
            largest_red_contours[1] = largest_red_contours[0]  # Shift largest to second
            largest_red_contours[0] = contour
            secnd_largest_area = largest_area
            largest_area = area
        elif area > secnd_largest_area:
            largest_red_contours[1] = contour
            secnd_largest_area = area
    if largest_red_contours[0] is not None:
        num_found = len([c for c in largest_red_contours if c is not None])  # 1 or 2 depending on what was found
    else:
        num_found = 0

    for i in range(num_found):
        contour_info = largest_red_contours[i]
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
            if (i==0):
                print(area)

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
    global skipped_counter, PixelToMM, crop_mask, GPU_ENABLED

    # Define maximum allowed deviation from prediction
    max_angle_deviation = 5  # degrees
   
    SEARCH_AREA = CROP_HALF_SIZE
    cropping = ENABLE_CROPPING
    area_threshold = 10.0  # 20% threshold for contour area change
    
    # Position and angle outlier rejection thresholds
    position_threshold = 100.0 # Standard deviations
    angle_threshold = 100.0     # Standard deviations
    
    start_time = time.time()
    
    # --- Grayscale (GPU UMat if enabled) ---
    if GPU_ENABLED:
        try:
            u_bgr = cv2.UMat(cvImage)
            frame = cv2.cvtColor(u_bgr, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            rospy.logwarn(f"GPU path failed during grayscale ({e}); switching to CPU.")
            GPU_ENABLED = False
            frame = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    else:
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
            frame_proc = cv2.bilateralFilter(
                frame,
                MICROGRIPPER_PARAMS['bilateral_d'],
                MICROGRIPPER_PARAMS['bilateral_sigma_color'],
                MICROGRIPPER_PARAMS['bilateral_sigma_space']
            )
        else:
            frame_proc = frame
        
        # Edge detection
        if MICROGRIPPER_PARAMS['use_canny']:
            edges = cv2.Canny(
                frame_proc, 
                MICROGRIPPER_PARAMS['canny_threshold1'],
                MICROGRIPPER_PARAMS['canny_threshold2']
            )
        else:
            # Adaptive thresholding
            block_size = MICROGRIPPER_PARAMS['adaptive_block_size']
            if block_size % 2 == 0:  # Must be odd
                block_size += 1
                
            edges = cv2.adaptiveThreshold(
                frame_proc,
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
        # Removed manual trimming (deque auto-manages size). If it is still a list, slice it:
        if not isinstance(openlengths, deque) and len(openlengths) > 10:
            del openlengths[:-10]

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
    # cv2.imshow("Edges", edges)
    # cv2.waitKey(1)

    # Reuse crop_mask
    if cropping:
        if crop_mask is None or crop_mask.shape != cvImage.shape[:2]:
            crop_mask = np.zeros(cvImage.shape[:2], dtype=np.uint8)
        else:
            crop_mask.fill(0)

    # Create crop mask for region of interest
    predicted_centroid, predicted_angle = predict_next_values(centroids, angle_vectors) 
    # If we have predictions, use them for cropping to improve processing speed and accuracy
    if cropping:
        if predicted_centroid is not None:
            cxp, cyp = predicted_centroid
        elif centroids:
            cxp, cyp = centroids[-1]
        else:
            cxp, cyp = cvImage.shape[1]//2, cvImage.shape[0]//2
        x1 = int(max(0, cxp - SEARCH_AREA))
        y1 = int(max(0, cyp - SEARCH_AREA))
        x2 = int(min(cvImage.shape[1], cxp + SEARCH_AREA))
        y2 = int(min(cvImage.shape[0], cyp + SEARCH_AREA))
        cv2.rectangle(crop_mask, (x1, y1), (x2, y2), 255, -1)
        # If edges is UMat bring to host first for mask op
        if GPU_ENABLED and isinstance(edges, cv2.UMat):
            edges_host = edges.get()
        else:
            edges_host = edges
        edges_host = cv2.bitwise_and(edges_host, edges_host, mask=crop_mask)
    else:
        predicted_centroid, predicted_angle = predict_next_values(centroids, angle_vectors)
        if GPU_ENABLED and isinstance(edges, cv2.UMat):
            edges_host = edges.get()
        else:
            edges_host = edges

    # change from RETR_EXTERNAL to RETR_LIST if background noise is extreme
    contours, _ = cv2.findContours(edges_host, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if contours is not None:
        (cx,cy), simple_hull = find_contour(contours)              
        if (skipped_counter > 10):
            # discard oldest entries (was pop() removing newest)
            if len(centroids) > 3:
                if hasattr(centroids, 'popleft'):
                    centroids.popleft()
                else:
                    centroids.pop(0)
            if len(angle_vectors) > 3:
                if hasattr(angle_vectors, 'popleft'):
                    angle_vectors.popleft()
                else:
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
                    angle_vectors.append(0.0)  # Default to 0 if no previous angle   
        if simple_hull is None and len(fiducials) > 0:
            # If only one fiducial is found, we can still use it to estimate the position
            # but we won't have a valid angle vector
            cx, cy = fiducials[0][0]
            if angle_vectors is not None and len(angle_vectors) > 0:
                angle_vectors.append(angle_vectors[-1])
            else:
                angle_vectors.append(0.0)
            centroids.append((cx, cy))
            timestamps.append(timestamp)
            cv2.circle(cvImage, (int(cx), int(cy)), radius=7, color=(0, 255, 0), thickness=-1)
    else:
        # cv2.imshow("No contours found", cvImage)
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
    
    # At the overlay section (before return), add GPU status:
    if GPU_ENABLED:
        cv2.putText(cvImage, "GPU(OpenCL)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    if not GPU_ENABLED:
        cv2.putText(cvImage, "CPU Mode", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)

    return cvImage, openColor, centroids, angle_vectors, openlengths, timestamps

def publish_pose(publisher, x, y, theta, opening, timestamp=None):
        try:
            # print(f"x,y:({x},{y})")

            # Convert to mm using effective scale (handles downscaled processing)
            x = x * PixelToMM_EFFECTIVE
            y = y * PixelToMM_EFFECTIVE
            opening = opening * PixelToMM_EFFECTIVE * 1000 # opening in um

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

def publish_feedback_image(publisher,image,timestamp=None): 
    try:
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
        if timestamp is not None:
            img_msg.header.stamp = timestamp
        publisher.publish(img_msg)
    except Exception as e:
        print(f"Error publishing feedback image: {e}")
        

# --- Performance control & stats (added) ---
FRAME_SKIP = 0          # process every frame by default
frame_counter = 0
MAX_HISTORY = 15
ENABLE_CROPPING = True
CROP_HALF_SIZE = 175
crop_mask = None
# New scaling & threading params
PROCESS_SCALE = 1.0
PixelToMM_EFFECTIVE = PixelToMM  # updated if PROCESS_SCALE < 1
LATEST_ONLY = False
SHOW_GUI = False
latest_frame = None
latest_stamp = None
processing_thread = None
processing_thread_run = False
processing_lock = threading.Lock()
# FPS / latency stats
fps_frame_count = 0
fps_last_time = time.time()
latency_sum = 0.0
latency_count = 0

def image_callback(msg):
    # Consolidated globals (bridge must appear before any use)
    global frame_counter, fps_frame_count, fps_last_time, latency_sum, latency_count
    global latest_frame, latest_stamp
    global bridge, centroids, angle_vectors, pose_publisher, openlengths, last_image_time, timestamps

    frame_counter += 1
    if FRAME_SKIP > 0 and (frame_counter % (FRAME_SKIP + 1)) != 1:
        return

    if LATEST_ONLY:
        try:
            img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return
        with processing_lock:
            latest_frame = img
            latest_stamp = msg.header.stamp
        return

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
        publish_pose(pose_publisher, x, y, angle_vectors[-1], openlengths[-1], timestamp)
        publish_feedback_image(image_publisher, processed_img)
        if SHOW_GUI:
            cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
            if cv2.waitKey(3) & 0xFF == ord(' '):
                cv2.destroyAllWindows()
    # elif processed_img is not None:
    #     cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))

    # --- Latency & FPS accounting (added) ---
    now_time = time.time()
    frame_latency = now_time - timestamp_sec
    latency_sum += frame_latency
    latency_count += 1
    fps_frame_count += 1
    if now_time - fps_last_time >= 1.0:
        interval = now_time - fps_last_time
        fps = fps_frame_count / interval
        avg_latency_ms = (latency_sum / max(1, latency_count)) * 1000.0
        print(f"Vision FPS: {fps:.2f} | Avg Latency: {avg_latency_ms:.1f} ms | Frames: {fps_frame_count}")
        fps_last_time = now_time
        fps_frame_count = 0
        latency_sum = 0.0
        latency_count = 0

# Processing loop for latest-only mode
def processing_loop():
    global latest_frame, latest_stamp
    global fps_frame_count, fps_last_time, latency_sum, latency_count
    while processing_thread_run and not rospy.is_shutdown():
        frame = None
        stamp = None
        with processing_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                stamp = latest_stamp
                latest_frame = None  # mark consumed
        if frame is None:
            time.sleep(0.001)
            continue
        # Apply processing scale
        working = frame
        scale_used = PROCESS_SCALE
        if PROCESS_SCALE != 1.0:
            working = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE, interpolation=cv2.INTER_AREA)
        # Prepare timestamp
        if stamp is None:
            stamp = rospy.Time.now()
        stamp_sec = stamp.to_sec()
        # Run detection
        processed_img, _, _, _, _, _ = microgripperDetection(working, stamp_sec, (0,255,0), centroids, angle_vectors, openlengths, timestamps)
        # Publish if we have data
        if processed_img is not None and centroids:
            # Last centroid (scaled); convert to original pixel coordinates for pose math
            cx_scaled, cy_scaled = centroids[-1]
            cx_full = cx_scaled
            cy_full = cy_scaled
            # (We adapt PixelToMM_EFFECTIVE globally, so no need to upscale coords for publishing)
            x = cx_full - working.shape[1] / 2
            y = -cy_full + working.shape[0] / 2
            publish_pose(pose_publisher, x, y, angle_vectors[-1], openlengths[-1], stamp)
            if SHOW_GUI:
                cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5) if PROCESS_SCALE != 1.0 else processed_img)
                cv2.waitKey(1)
            publish_feedback_image(image_publisher, processed_img, stamp)
        # Latency / FPS
        now_time = time.time()
        frame_latency = now_time - stamp_sec
        latency_sum += frame_latency
        latency_count += 1
        fps_frame_count += 1
        if now_time - fps_last_time >= 1.0:
            interval = now_time - fps_last_time
            fps = fps_frame_count / interval
            avg_latency_ms = (latency_sum / max(1, latency_count)) * 1000.0
            print(f"Vision FPS: {fps:.2f} | Avg Latency: {avg_latency_ms:.1f} ms | Frames: {fps_frame_count}")
            fps_last_time = now_time
            fps_frame_count = 0
            latency_sum = 0.0
            latency_count = 0

def main():
    rospy.init_node('image_processor_node', anonymous=True)
    setup_gpu()
    # Load params
    global FRAME_SKIP, MAX_HISTORY, ENABLE_CROPPING, CROP_HALF_SIZE
    global PROCESS_SCALE, PixelToMM_EFFECTIVE, LATEST_ONLY, SHOW_GUI
    FRAME_SKIP = rospy.get_param('~frame_skip', 0)
    MAX_HISTORY = rospy.get_param('~max_history', 15)
    ENABLE_CROPPING = rospy.get_param('~enable_cropping', True)
    CROP_HALF_SIZE = rospy.get_param('~crop_half_size', 175)
    PROCESS_SCALE = float(rospy.get_param('~process_scale', 1.0))
    LATEST_ONLY = rospy.get_param('~latest_only', True)
    SHOW_GUI = rospy.get_param('~show_gui', True)
    if PROCESS_SCALE <= 0 or PROCESS_SCALE > 1.0:
        PROCESS_SCALE = 1.0
    # Adjust effective pixel size
    global PixelToMM_EFFECTIVE
    PixelToMM_EFFECTIVE = PixelToMM / PROCESS_SCALE
    rospy.loginfo(f"process_scale={PROCESS_SCALE} | effective PixelToMM={PixelToMM_EFFECTIVE:.6f} mm/px | latest_only={LATEST_ONLY}")

    # Initialize histories
    global centroids, angle_vectors, pose_publisher, image_publisher, openlengths, last_image_time, timestamps, bridge
    centroids = deque(maxlen=MAX_HISTORY)
    angle_vectors = deque(maxlen=MAX_HISTORY)
    openlengths = deque([0], maxlen=MAX_HISTORY)
    timestamps = deque(maxlen=MAX_HISTORY)
    bridge = CvBridge()
    last_image_time = time.time()

    rospy.Timer(rospy.Duration(0.5), timeout_callback)

    rospy.Subscriber("/camera/basler_camera_1/image_raw", Image, image_callback, queue_size=1, buff_size=2**22)
    pose_publisher = rospy.Publisher('/vision_feedback/pose_estimation', Float64MultiArray, queue_size=5)
    image_publisher = rospy.Publisher('/vision_feedback/processed_image', Image, queue_size=2)
    rospy.loginfo("MicroGripper Vision Feedback started. Will quit if no images received for {} seconds.".format(image_timeout))

    # Start processing thread if latest-only
    global processing_thread, processing_thread_run
    if LATEST_ONLY:
        processing_thread_run = True
        processing_thread = threading.Thread(target=processing_loop, daemon=True)
        processing_thread.start()
        rospy.loginfo("Latest-only processing thread started.")

    rospy.spin()
    # Cleanup
    if LATEST_ONLY:
        processing_thread_run = False
        if processing_thread is not None:
            processing_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()


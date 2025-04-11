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
image_timeout = 30.0  # 3 seconds timeout
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

def find_bright_squares(image):
    """
    Finds two bright squares with thin dark borders using edge detection.
    Returns the center points of the squares and their bounding rectangles.
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Get parameters from current settings
    blur_size = MICROGRIPPER_PARAMS["square_blur_size"]
    if blur_size % 2 == 0:  # Must be odd
        blur_size += 1
        
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Edge detection to find borders
    edges = cv2.Canny(blurred, 
                        MICROGRIPPER_PARAMS["square_canny_threshold1"], 
                        MICROGRIPPER_PARAMS["square_canny_threshold2"])
    
    # Dilate slightly to connect any broken edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=MICROGRIPPER_PARAMS["square_dilate_iterations"])
    
    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    # Filter contours to find squares of similar size
    potential_squares = []
    for contour in contours:
        # Calculate area
        hull = cv2.convexHull(contour)

        area = cv2.contourArea(hull)

        # Skip contours outside area range
        if area < MICROGRIPPER_PARAMS["square_min_area"] or area > MICROGRIPPER_PARAMS["square_max_area"]:
            continue
        
        # Approximate the contour to a polygon
        epsilon = MICROGRIPPER_PARAMS["square_epsilon"]/1000.0 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 1)  # Original in green
        cv2.drawContours(debug_image, [approx], -1, (255, 0, 255), 1)  # Approximation in magenta
        # Check if it's approximately a square (~4 vertices)
        if abs(len(approx)-4) <=2:
            # Get the bounding rectangle
            rect = cv2.minAreaRect(approx)
            (cx, cy), (width, height), angle = rect
            
            # Check if it's square-like (aspect ratio near 1)
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < MICROGRIPPER_PARAMS["square_aspect_ratio"]:
                # Calculate average brightness inside the contour
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [hull], 0, 255, -1)
                mean_brightness = cv2.mean(gray, mask=mask)[0]
                
                # Calculate mean brightness of whole image for comparison
                mean_image_brightness = np.mean(gray)
                
                # Add to potential squares if bright enough compared to image average
                brightness_ratio = MICROGRIPPER_PARAMS.get("square_brightness_ratio", 1.5)
                if mean_brightness > brightness_ratio * mean_image_brightness:
                    potential_squares.append({
                        'contour': hull,
                        'rect': rect,
                        'area': area,
                        'center': (cx, cy),
                        'brightness': mean_brightness
                    })
    # Sort by area
    potential_squares.sort(key=lambda x: x['area'], reverse=True)
    cv2.imshow("Debug Image", debug_image)
    cv2.waitKey(1)
    # Find the two most similar squares by area
    best_pair = None
    min_area_diff = float('inf')
    
    if len(potential_squares) >= 2:
        for i in range(len(potential_squares)):
            for j in range(i+1, len(potential_squares)):
                area_diff = abs(potential_squares[i]['area'] - potential_squares[j]['area'])
                # Calculate ratio to find squares of similar size
                area_ratio = max(potential_squares[i]['area'], potential_squares[j]['area']) / \
                            max(1, min(potential_squares[i]['area'], potential_squares[j]['area']))
                
                if area_ratio < 1.5 and area_diff < min_area_diff:
                    min_area_diff = area_diff
                    best_pair = (potential_squares[i], potential_squares[j])
    
    # Return the centers and rectangles of the two squares if found
    if best_pair:
        sq1, sq2 = best_pair
        return [(sq1['center'], sq1['rect']), (sq2['center'], sq2['rect'])]
    else:
        return []

def microgripperDetection(cvImage, timestamp, openColor, centroids, angle_vectors, areas, openlengths, timestamps):
    global skipped_counter, PixelToMM, previous_angle
    # Define maximum allowed deviation from prediction
    max_position_deviation = 200  # pixels
    max_angle_deviation = 10  # degrees
    global q,w,e,r,t,y
    
    NUM_LARGEST = 6 # number of the largest contours to keep/check if robot
    kernel = np.ones((3,3))
    SEARCH_AREA = 175
    cropping = False
    bot_rect = None
    area_threshold = 10.0  # 20% threshold for contour area change
    
    # Position and angle outlier rejection thresholds
    position_threshold = 100.0 # Standard deviations
    angle_threshold = 100.0     # Standard deviations
    
    start_time = time.time()
    
    # Convert image to grayscale
    frame = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    # find two squares 
    squares = find_bright_squares(frame)
        #return cvImage, openColor, centroids, angle_vectors, areas, openlengths, timestamps
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
            edges = cv2.erode(edges, kernel, iterations=MICROGRIPPER_PARAMS['erode_iterations'])
        
        if MICROGRIPPER_PARAMS['dilate1_iterations'] > 0:
            edges = cv2.dilate(edges, kernel, iterations=MICROGRIPPER_PARAMS['dilate1_iterations'])
        
        if MICROGRIPPER_PARAMS['erode2_iterations'] > 0:
            edges = cv2.erode(edges, kernel, iterations=MICROGRIPPER_PARAMS['erode2_iterations'])
        
        if MICROGRIPPER_PARAMS['dilate2_iterations'] > 0:
            edges = cv2.dilate(edges, kernel, iterations=MICROGRIPPER_PARAMS['dilate2_iterations'])
            
        # Update contour filtering parameters
        MAX_AREA = MICROGRIPPER_PARAMS['max_area']
        MIN_AREA = MICROGRIPPER_PARAMS['min_area']
        hull_epsilon = MICROGRIPPER_PARAMS['hull_epsilon']
        min_hull_points = MICROGRIPPER_PARAMS['min_hull_points']
        max_hull_points = MICROGRIPPER_PARAMS['max_hull_points']
        aspect_ratio = MICROGRIPPER_PARAMS['aspect_ratio']
        
    else:
        # Use default image processing parameters
        blurred = cv2.bilateralFilter(frame, 5, 10, 10)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -6)
        edges = cv2.erode(edges, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=5)
        edges = cv2.erode(edges, kernel, iterations=3)
        edges = cv2.dilate(edges, kernel, iterations=4)
        
        # Default contour filtering parameters
        MAX_AREA = 100000
        MIN_AREA = 40000
        hull_epsilon = 0.013
        min_hull_points = 3
        max_hull_points = 15
        aspect_ratio = 1.75
    
    # Display the edges for debugging
    #cv2.imshow("Edges", edges)
    
    # Create crop mask for region of interest
    crop_mask = np.zeros_like(edges)
    
    # If we have predictions, use them for cropping to improve processing speed and accuracy
    predicted_centroid = None
    predicted_angle = None
    if len(centroids) >= 5 and cropping:
        predicted_centroid, predicted_angle = predict_next_values(centroids, angle_vectors, timestamps[-1])
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
    current_contour_area = None
    if contours is not None:
        for i in range(0, min(len(contours),NUM_LARGEST)):
            max_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
            contour = contours[max_idx]
            contours.pop(max_idx)
            hull = cv2.convexHull(contour)
            # Calculate the convex hull of the contour and simplify it to a hexagon
            # Use the hull_epsilon parameter to control the simplification
            poly = 0
            while poly != 6:
                simple_hull = cv2.approxPolyDP(hull, hull_epsilon * cv2.arcLength(hull, True), True)
                poly = len(simple_hull)
                if poly < 6:
                    hull_epsilon = hull_epsilon / 1.01
                elif poly > 6:
                    hull_epsilon = hull_epsilon * 1.1

            if len(simple_hull) >= min_hull_points and len(simple_hull) <= max_hull_points:
                w+=1
                rect = cv2.minAreaRect(contour)
                (cx, cy), (width, height), angle = rect  
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    # Fallback if contour area is zero
                    cx, cy = rect[0]                # Ensure width is the longer dimension
                if width < height:
                    width, height = height, width
                else:
                    angle = angle-90 
                
                area = cv2.contourArea(contour)
                #print(area)
                # Apply aspect ratio and area constraints
                if width < aspect_ratio*height and area < MAX_AREA and area > MIN_AREA:
                    # rest of the detection code remains the same
                    for j in [1]:
                        # Perform outlier rejection if we have enough history
                        is_outlier = False
                        if len(centroids) >= 5:
                                                        # Get prediction for this frame based on past measurements
                            predicted_centroid, predicted_angle = predict_next_values(centroids, angle_vectors) 
                            
                            if predicted_centroid is not None and predicted_angle is not None:
                                # Calculate distance to predicted position
                                pred_cx, pred_cy = predicted_centroid
                                distance_to_prediction = math.sqrt((cx - pred_cx)**2 + (cy - pred_cy)**2)
                                
                                # # Calculate angle difference to prediction (handling wraparound)
                                if previous_angle is not None:
                                    angle_diff = angle_difference(angle, previous_angle)
                                else:
                                    angle_diff = 0

                                # print(angle)
                                if angle_diff > 90:
                                    print(f"Angle outlier rejected: {angle:.1f}° - Diff: {angle_diff:.2f}°")
                                    is_outlier = True
                                    skipped_counter +=1
                                    continue

                                # Check if current measurement is too far from prediction
                                if distance_to_prediction > max_position_deviation:
                                    print(f"Position outlier rejected: ({cx:.1f}, {cy:.1f}) - too far from prediction: {distance_to_prediction:.2f}px")
                                    is_outlier = True
                                    skipped_counter +=1
                                    continue
                            
                            # Also perform traditional statistical outlier rejection
                            # Check if area is within threshold of previous area
                            mean_area = np.mean(areas)  
                            std_area = np.std(areas)
                            if std_area > 0:  # Avoid division by zero
                                # Calculate z-score for area
                                z_area = abs(area - mean_area) / std_area
                                if False and z_area > area_threshold:  # 3 standard deviations
                                    print(f"Area outlier rejected: {area:.0f} - z-score: {z_area:.2f}")
                                    is_outlier = True
                                    skipped_counter +=1
                                    continue 
                            # Calculate statistics for centroids
                            centroids_array = np.array(centroids)
                            mean_x = np.mean(centroids_array[:, 0])
                            mean_y = np.mean(centroids_array[:, 1])
                            std_x = np.std(centroids_array[:, 0])
                            std_y = np.std(centroids_array[:, 1])
                            
                            # Check if new position is an outlier
                            if std_x > 0 and std_y > 0:  # Avoid division by zero
                                z_x = abs(cx - mean_x) / std_x
                                z_y = abs(cy - mean_y) / std_y
                                
                                if z_x > position_threshold or z_y > position_threshold:
                                    print(f"Position outlier rejected: ({cx:.1f}, {cy:.1f}) - z-scores: x={z_x:.2f}, y={z_y:.2f}")
                                    is_outlier = True
                                    skipped_counter +=1
                                    continue 
                                                                 
                    # Skip this contour if it's an outlier
                    if is_outlier:
                        openColor = (0, 0, 255)  # red
                        if skipped_counter > 10 and len(centroids) > 3:
                            centroids.pop(0)
                            angle_vectors.pop(0)
                            areas.pop(0)
                        continue
                                            
                    if len(centroids) < 5:
                        bot_rect = rect
                        openColor = (0, 255, 0)  # green
                        break
                    else:
                        (avg_cx, avg_cy) = np.mean(centroids, axis=0)  
                        avg_angle = np.mean(angle_vectors)
                        
                        # Accept the detection
                        bot_rect = rect
                        openColor = (0, 255, 0)  # green
                        
                        if len(centroids) > 10:
                            centroids.pop(0)
                            angle_vectors.pop(0)
                            areas.pop(0)   
                        break                     

        if bot_rect:
            y+=1   
            thetas = []    
            lengths = []   
            for i in range(len(simple_hull)):
                v1 = (simple_hull[i-1] - simple_hull[i]).flatten()
                v2 = (simple_hull[(i+1) % len(simple_hull)] - simple_hull[i]).flatten()
                theta = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                theta = np.degrees(np.arccos(theta))
                thetas.append((theta,i))
                lengths.append((np.linalg.norm(v1),i))
            
            # Reshape the hull points to a more usable format
            simple_hull = np.array(simple_hull).reshape(-1, 2)
            if squares != []:
                s1x,s1y=squares[0][0]
                s2x,s2y=squares[1][0]
                main_axis = np.array([s1x-s2x,s1y-s2y])
                squares_mid = np.array([s1x+s2x,s1y+s2y])/2

                main_axis_norm = np.linalg.norm(main_axis)
                if main_axis_norm > 0:
                    main_axis_unit = main_axis / main_axis_norm
                else:
                    main_axis_unit = np.array([1.0, 0.0])  # Fallback if zero length (shouldn't happen)
                
                # NEW: Use the centroid (cx, cy) as the center point for our baseline
                # instead of the midpoint of the longest side
                centroid_point = np.array([cx, cy])
                forwards = centroid_point-squares_mid
                
                # Calculate vector perpendicular to main axis
                perp_vector = np.array([-main_axis_unit[1], main_axis_unit[0]])
                #  lets use the perpendicular vector to the base of the rect
                # perp_vector = np.array([-math.cos(angle),math.sin(angle)])
                


                # Step 2: Find all edges (consecutive point pairs) around the hull
                edges = []
                for i in range(len(simple_hull)):
                    next_i = (i + 1) % len(simple_hull)
                    
                    # Calculate edge vector, midpoint and length
                    pt1 = simple_hull[i]
                    pt2 = simple_hull[next_i]
                    edge_vector = pt2 - pt1
                    edge_midpoint = (pt1 + pt2) / 2
                    edge_length = np.linalg.norm(edge_vector)
                    
                    # Calculate vector from centroid to edge midpoint
                    to_edge_vector = edge_midpoint - centroid_point
                    
                    # Calculate alignment with main axis (how parallel they are)
                    if edge_length > 0:
                        edge_unit = edge_vector / edge_length
                        # Higher value = more parallel (1.0 = perfectly parallel)
                        alignment = abs(np.dot(edge_unit, main_axis_unit))
                        
                        # Check if edge is in forwards direction from centroid
                        in_forwards_direction = np.dot(to_edge_vector, forwards) > 0
                        
                        edges.append({
                            'pt1': pt1,
                            'pt2': pt2,
                            'midpoint': edge_midpoint,
                            'vector': edge_vector,
                            'length': edge_length,
                            'alignment': alignment,
                            'in_forwards_direction': in_forwards_direction
                        })
                    
                # Find edge most parallel to main axis in forwards direction
                best_edge = None
                best_alignment = -1

                for edge in edges:
                    if edge['in_forwards_direction'] and edge['alignment'] > best_alignment:
                        best_alignment = edge['alignment']
                        best_edge = edge

                # Use the best edge if found
                if best_edge:
                    # Draw the best edge with distinct appearance
                    tipl = tuple(map(int, best_edge['pt1']))
                    tipr = tuple(map(int, best_edge['pt2']))
                    cv2.line(cvImage, tipl, tipr, (0, 255, 255), 3)  # Yellow line

                    s1_center = tuple(map(int, squares[0][0]))
                    s2_center = tuple(map(int, squares[1][0]))
                    cv2.line(cvImage, s1_center, s2_center, (0, 0, 255), 2)  # Red line
                    
                    # Label it
                    midpoint = tuple(map(int, best_edge['midpoint']))
                    cv2.putText(cvImage, f"Parallel Edge ({best_alignment:.2f})", midpoint,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Calculate distance from centroid to edge (perpendicular distance)
                    edge_unit_vector = best_edge['vector'] / best_edge['length']
                    perp_vector = np.array([-edge_unit_vector[1], edge_unit_vector[0]])  # Perpendicular to edge
                    
                    # Calculate angle between main axis and this edge
                    angle_deg = np.degrees(np.arccos(best_alignment))
                    cv2.putText(cvImage, f"Angle: {angle_deg:.1f}°", 
                            (midpoint[0], midpoint[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                                
                    # Draw the perpendicular bisector line for visualization
                    # Calculate a point 200 pixels along the perpendicular in each direction from centroid
                    perp_pt1 = centroid_point + perp_vector * 200
                    perp_pt2 = centroid_point - perp_vector * 200
                    cv2.line(cvImage, tuple(map(int, perp_pt1)), tuple(map(int, perp_pt2)), (255, 255, 255), 1)
                    
                    # Draw the centroid point
                    cv2.circle(cvImage, (int(cx), int(cy)), radius=6, color=(0, 0, 255), thickness=-1)
                    cv2.putText(cvImage, "Centroid", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # For angle calculation, use centroid and tip midpoint
                    mid2 = tuple(centroid_point.astype(int))  # Centroid point
                    mid1 = midpoint  # Tips midpoint
                
                    # Calculate angle from the direction vector
                    v = mid1 - centroid_point  # Vector pointing from base to tips
                    angle_vector = np.arctan2(v[1], v[0])
                    angle_vector = normalize_angle_degrees(np.degrees(angle_vector))
                    
                    # Calculate statistics for angles
                    mean_angle = np.mean(angle_vectors) if len(angle_vectors) > 0 else angle_vector
                    std_angle = np.std(angle_vectors) if len(angle_vectors) > 0 else 1
                    
                    angle_to_prediction = angle_difference(angle_vector, predicted_angle) if predicted_angle is not None else 0
                    # Calculate smallest angle difference accounting for wraparound
                    angle_diff = angle_difference(angle_vector, mean_angle)
                    
                    acceptable_deviation = max(10.0, std_angle * 2)  # Adapt to observed variability but cap it
                    if angle_diff > acceptable_deviation and angle_to_prediction > max_angle_deviation:
                        # Reject this angle as an outlier
                        print(f"Angle rejected: {angle_vector:.1f}°, Diff: {angle_diff:.2f}°, skipped:{skipped_counter}")
                        if angle_vectors:  # Check if list is not empty
                            angle_vectors.append(angle_vectors[-1])
                        else:
                            angle_vectors.append(angle_vector)  # Use current value as fallback
                        angle_color = [0,0,255] # red
                        skipped_counter +=1
                        # return cvImage, openColor, centroids, angles, areas, openlengths, timestamps
                        if angle_diff > acceptable_deviation:
                            print(f"Angle deviation: {angle_vector:.1f}°, Diff: {angle_diff:.2f}°")
                        if angle_to_prediction > max_angle_deviation:
                            print(f"Angle prediction deviation: {angle_vector:.1f}°, Diff: {angle_to_prediction:.2f}°")
                    else:
                        angle_vectors.append(angle_vector)

                    previous_angle = angle
                    skipped_counter = 0
                    angle_color = [0, 255, 0]  # green

                    if len(tipl) != 0 and len(tipr) != 0:
                        openlength = np.linalg.norm(np.array(tipl) - np.array(tipr)) 
                        openlengths.append(openlength)
                        if len(openlengths) >= 5:
                            openlengths.pop(0)

                        centroidtuple = tuple(map(int, centroid_point))
                        
                        # Draw tip points
                        cv2.circle(cvImage, tipl, radius=6, color=openColor, thickness=-1)
                        cv2.circle(cvImage, tipr, radius=6, color=openColor, thickness=-1)
                        
                        # Draw centroid
                        cv2.circle(cvImage, centroidtuple, radius=6, color=(0, 225, 225), thickness=-1)
                        # Dreas midline
                        cv2.arrowedLine(cvImage, mid2, mid1, angle_color, 4)

                        # draw cordinate system
                        x = cvImage.shape[1] / 2  # Adjust x to have 0,0 at the center of the image
                        y = cvImage.shape[0] / 2  # Adjust y to have 0,0 at the center of the image

                        # cv2.arrowedLine(cvImage, (int(x), int(y)), (int(x+1/PixelToMM), int(y)), (225, 0, 0), 2)  # X-axis
                        cv2.arrowedLine(cvImage, (int(x), int(y)), (int(x+(1/PixelToMM)/2), int(y)), (225, 0, 0), 2)  # X-axis
                        cv2.arrowedLine(cvImage, (int(x), int(y)), (int(x), int(y-1/PixelToMM/2)), (225, 0, 0), 2)  # Y-axis
                        cv2.putText(cvImage, "X", (int(x+15), int(y-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 1)
                        cv2.putText(cvImage, "Y", (int(x+50), int(y-15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 0), 1)
                        cv2.circle(cvImage, (int(x), int(y)), radius=3, color=(225, 0, 0), thickness=-1)  # Origin point

                        # Add text showing the opening distance
                        cv2.putText(cvImage, f"Opening: {openlength*PixelToMM*1000:.1f}um", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                print(f"squares not found")
                skipped_counter +=1
                if angle_vectors:  # Check if list is not empty
                    angle_vectors.append(angle_vectors[-1])
                else:
                    angle_vectors.append(angle_vector)            
            centroids.append((cx,cy))
            areas.append(area)
            timestamps.append(timestamp)        
            # Draw the contour outline
            cv2.drawContours(cvImage, [simple_hull.astype(np.int32)], 0, openColor, 2)
    else:
        cv2.imshow("No contours found", cvImage)
        print("No robot contours found.")
        
    # Print performance metrics
    # print(f"{q:0.2f} {w/q:0.2f} {e/q:0.2f} {r/q:0.2f} {t/q:0.2f} {y/q:0.2f} - Skipped Counter: {skipped_counter:.12f}")
    
    # Visualize prediction if available
    if predicted_centroid is not None and bot_rect is not None:
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
    return cvImage, openColor, centroids, angle_vectors, areas, openlengths, timestamps

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
    global centroids, angle_vectors, areas, publisher, openlengths, last_image_time, timestamps
    
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
    
    processed_img, openColor, centroids, angle_vectors, areas, openlengths, timestamps = microgripperDetection(cv_image, timestamp_sec, openColor, centroids, angle_vectors, areas, openlengths, timestamps)
    
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
    global centroids, angle_vectors, publisher, areas, openlengths, last_image_time, timestamps, previous_angle
    openlengths = [0]
    areas = []
    centroids = []
    angle_vectors = []
    timestamps = []
    last_image_time = time.time()
    previous_angle = None
    # Set up the timeout timer to check every 0.5 seconds
    rospy.Timer(rospy.Duration(0.5), timeout_callback)
    
    rospy.Subscriber("/camera/basler_camera_1/image_raw", Image, image_callback)
    publisher = rospy.Publisher('/vision_feedback/pose_estimation', Float64MultiArray, queue_size=10)
    
    rospy.loginfo("MicroGripper Vision Feedback started. Will quit if no images received for {} seconds.".format(image_timeout))
    
    rospy.spin()   

if __name__ == "__main__":
    main()


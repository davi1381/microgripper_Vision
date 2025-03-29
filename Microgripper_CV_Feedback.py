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
    from CVParameters1 import MICROGRIPPER_PARAMS
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
PixelToMM = 0.00315 # Conversion factor from pixels to mm (assuming 1600 pixels in width)


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
    Predicts the next centroid and angle values based on the weighted average 
    of changes in the last measurements.
    
    Args:
        centroids: List of (x, y) tuples for past centroids
        angles: List of past angle values (in degrees)
        timestamp: Current timestamp for time-based prediction
        
    Returns:
        predicted_centroid: (x, y) tuple with predicted position
        predicted_angle: Predicted angle in degrees
    """
    global last_timestamps, last_positions, last_angles
    
    # Need at least 2 measurements to calculate changes
    if len(centroids) < 2:
        return centroids[-1] if centroids else None, angles[-1] if angles else None
    
    # Create numpy arrays for easier computation
    centroids_array = np.array(centroids)
    
    # Calculate position changes between consecutive measurements
    position_diffs = np.diff(centroids_array, axis=0)
    
    # Calculate angle changes (handling wraparound at 360 degrees)
    angle_diffs = []
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i-1]
        # Handle wraparound (e.g., 350° to 10° should be +20° not -340°)
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        angle_diffs.append(diff)
    
    # Assign weights - more recent changes get higher weights
    # We'll use an exponential weighting scheme
    num_changes = len(position_diffs)
    weights = np.exp(np.linspace(0, 1, num_changes))
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    # Calculate weighted average position change
    weighted_pos_change = np.zeros(2)
    for i in range(num_changes):
        weighted_pos_change += weights[i] * position_diffs[i]
    
    # Calculate weighted average angle change
    weighted_angle_change = 0
    for i in range(len(angle_diffs)):
        weighted_angle_change += weights[i] * angle_diffs[i]
    
    # Generate predictions
    predicted_centroid = centroids[-1] + weighted_pos_change
    predicted_angle = angles[-1] + weighted_angle_change
    
    # Normalize angle to [0, 360] range
    predicted_angle = predicted_angle % 360
    
    # If timestamp is provided, we can do time-based prediction
    if timestamp is not None and last_timestamps and timestamp > last_timestamps[-1]:
        # Store timestamp and position for future use
        if len(last_timestamps) >= 5:
            last_timestamps.pop(0)
            last_positions.pop(0)
            last_angles.pop(0)
            
        last_timestamps.append(timestamp)
        last_positions.append(centroids[-1])
        last_angles.append(angles[-1])
        
        if len(last_timestamps) >= 2:
            # Calculate time differences
            time_diffs = np.diff(last_timestamps)
            
            # Calculate velocity over time (pixels per second)
            velocities = []
            for i in range(len(time_diffs)):
                if time_diffs[i] > 0:  # Avoid division by zero
                    velocity = (np.array(last_positions[i+1]) - np.array(last_positions[i])) / time_diffs[i]
                    velocities.append(velocity)
                    
            # Calculate angle velocity over time (degrees per second)
            angle_velocities = []
            for i in range(len(time_diffs)):
                if time_diffs[i] > 0:
                    angle_diff = last_angles[i+1] - last_angles[i]
                    # Handle wraparound
                    if angle_diff > 180:
                        angle_diff -= 360
                    elif angle_diff < -180:
                        angle_diff += 360
                        
                    angle_velocity = angle_diff / time_diffs[i]
                    angle_velocities.append(angle_velocity)
            
            # If we have velocities, use them for prediction
            if velocities:
                # Use weighted average of velocities
                weights = np.exp(np.linspace(0, 1, len(velocities)))
                weights = weights / np.sum(weights)
                
                weighted_velocity = np.zeros(2)
                for i in range(len(velocities)):
                    weighted_velocity += weights[i] * velocities[i]
                    
                # Time since last measurement
                time_since_last = timestamp - last_timestamps[-1]
                
                # Predict position using velocity
                predicted_centroid = tuple(np.array(centroids[-1]) + weighted_velocity * time_since_last)
                
                # Predict angle using angle velocity if available
                if angle_velocities:
                    weighted_angle_velocity = 0
                    for i in range(len(angle_velocities)):
                        weighted_angle_velocity += weights[i] * angle_velocities[i]
                        
                    predicted_angle = angles[-1] + weighted_angle_velocity * time_since_last
                    predicted_angle = predicted_angle % 360
    
    return predicted_centroid, predicted_angle


def microgripperDetection(cvImage, timestamp, openColor, centroids, angles, areas, openlengths, timestamps):
    global skipped_counter
    # Define maximum allowed deviation from prediction
    max_position_deviation = 200  # pixels
    max_angle_deviation = 90  # degrees
    global q,w,e,r,t,y
    
    kernel = np.ones((3,3))
    SEARCH_AREA = 175
    cropping = False
    bot_rect = None
    area_threshold = 10.0e6  # 20% threshold for contour area change
    
    # Position and angle outlier rejection thresholds
    position_threshold = 10.0e6 # Standard deviations
    angle_threshold = 10.0e6     # Standard deviations
    
    start_time = time.time()
    
    # Convert image to grayscale
    frame = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    
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
        MIN_AREA = 25000
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
        predicted_centroid, predicted_angle = predict_next_values(centroids, angles, timestamps[-1])
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

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    current_contour_area = None
    q+=1
    if contours:
        #! error check contours size
        sortedPairs = sorted(zip(contours, hierarchy[0]), key=lambda pair: cv2.contourArea(pair[0]), reverse=True)[:6]
        contours, hierarchy = zip(*(sortedPairs)) # adjust number of accepted contours (10)
        for i in range(0, len(contours)):
            hull = cv2.convexHull(contours[i])
            simple_hull = cv2.approxPolyDP(hull, hull_epsilon * cv2.arcLength(hull, True), True)
            if len(simple_hull) >= min_hull_points and len(simple_hull) <= max_hull_points:
                w+=1
                rect = cv2.minAreaRect(contours[i])
                (cx, cy), (width, height), angle = rect  
                
                # Ensure width is the longer dimension
                if width < height:
                    width, height = height, width
                else:
                    angle = angle-90 
                
                area = cv2.contourArea(contours[i])
                
                # Apply aspect ratio and area constraints
                if width < aspect_ratio*height and area < MAX_AREA and area > MIN_AREA:
                    e+=1
                    # rest of the detection code remains the same
                    for j in [1]:
                        # Perform outlier rejection if we have enough history
                        is_outlier = False
                        if len(centroids) >= 5:
                                                        # Get prediction for this frame based on past measurements
                            predicted_centroid, predicted_angle = predict_next_values(centroids, angles)
                            
                            if predicted_centroid is not None:
                                # Calculate distance to predicted position
                                pred_cx, pred_cy = predicted_centroid
                                distance_to_prediction = math.sqrt((cx - pred_cx)**2 + (cy - pred_cy)**2)
                                
                                # Calculate angle difference to prediction (handling wraparound)
                                angle_diff = min(abs(angle - angles[-1]), abs((angle - 180) - angles[-1]))
                                
                                if angle_diff > 20 and False:
                                    print(f"Angle outlier rejected: {angle:.1f}° - too far from previous: {angle_diff:.2f}°")
                                    is_outlier = True
                                    skipped_counter +=1
                                    continue

                                # Check if current measurement is too far from prediction
                                if distance_to_prediction > max_position_deviation:
                                    print(f"Position outlier rejected: ({cx:.1f}, {cy:.1f}) - too far from prediction: {distance_to_prediction:.2f}px")
                                    is_outlier = True
                                    skipped_counter +=0.0001
                                    continue
                            
                            # Also perform traditional statistical outlier rejection
                            # Check if area is within threshold of previous area
                            mean_area = np.mean(areas)  
                            std_area = np.std(areas)
                            if std_area > 0:  # Avoid division by zero
                                # Calculate z-score for area
                                z_area = abs(area - mean_area) / std_area
                                if z_area > area_threshold:  # 3 standard deviations
                                    print(f"Area outlier rejected: {area:.0f} - z-score: {z_area:.2f}")
                                    is_outlier = True
                                    skipped_counter +=0.000001
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
                                    skipped_counter +=0.00000001
                                    continue 
                                                                 
                    # Skip this contour if it's an outlier
                    if is_outlier:
                        openColor = (0, 0, 255)  # red
                        if skipped_counter > 25:
                            centroids.pop(0)
                            angles.pop(0)
                            areas.pop(0)
                        continue
                                            
                    if len(centroids) < 5:
                        bot_rect = rect
                        bot_cnt = contours[i]
                        openColor = (0, 255, 0)  # green
                        break
                    else:
                        (avg_cx, avg_cy) = np.mean(centroids, axis=0)  
                        avg_angle = np.mean(angles)
                        
                        # Accept the detection
                        bot_rect = rect
                        bot_cnt = contours[i]
                        openColor = (0, 255, 0)  # green
                        
                        if len(centroids) > 10:
                            centroids.pop(0)
                            angles.pop(0)
                            areas.pop(0)   
                        break                     
        
        # The rest of the function remains the same
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
    
            # Find the two points furthest apart - this will be the main axis
            max_dist = 0
            side1, side2 = None, None
            
            for i in range(len(simple_hull)):
                for j in range(i + 1, len(simple_hull)):
                    dist = np.linalg.norm(simple_hull[i] - simple_hull[j])
                    if dist > max_dist:
                        max_dist = dist
                        side1, side2 = i, j
            
            # Modified tip detection algorithm to find the smaller of the two edges 
            # intersected by the perpendicular bisector of the midline
            
            # Step 1: Calculate the main axis vector (centerline)
            base1 = simple_hull[side1]
            base2 = simple_hull[side2]
            main_axis = base2 - base1
            main_axis_norm = np.linalg.norm(main_axis)
            if main_axis_norm > 0:
                main_axis_unit = main_axis / main_axis_norm
            else:
                main_axis_unit = np.array([1.0, 0.0])  # Fallback if zero length (shouldn't happen)
            
            # NEW: Use the centroid (cx, cy) as the center point for our baseline
            # instead of the midpoint of the longest side
            centroid_point = np.array([cx, cy])
            
            # The baseline direction is the same as the main axis (parallel to longest side)
            # but now it passes through the centroid
            
            # Calculate vector perpendicular to main axis
            perp_vector = np.array([-main_axis_unit[1], main_axis_unit[0]])
            #  lets use the perpendicular vector to the base of the rect
            # perp_vector = np.array([-math.cos(angle),math.sin(angle)])
            
            # Draw the baseline through the centroid point
            baseline_pt1 = centroid_point - main_axis_unit * 200
            baseline_pt2 = centroid_point + main_axis_unit * 200
            cv2.line(cvImage, tuple(map(int, baseline_pt1)), tuple(map(int, baseline_pt2)), 
                    (0, 0, 255), 2)  # Red line for baseline
            
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
                
                # Project this vector onto the perpendicular axis to determine side
                side_projection = np.dot(to_edge_vector, perp_vector)
                
                # Calculate how aligned this edge is with the perpendicular bisector
                # (how perpendicular it is to the main axis)
                if np.linalg.norm(edge_vector) > 0:
                    edge_unit_vector = edge_vector / np.linalg.norm(edge_vector)
                    perp_alignment = abs(np.dot(edge_unit_vector, main_axis_unit))
                else:
                    perp_alignment = 1.0
                
                # Calculate perpendicular distance from the midpoint of the edge to the baseline
                perp_dist = abs(np.dot(to_edge_vector, perp_vector))
                
                # Calculate intersection parameter with perpendicular from centroid
                intersection_score = 0
                intersection_point = None

                # IMPROVED INTERSECTION ALGORITHM:
                # Calculate if the perpendicular line from centroid intersects this edge segment

                # Define line segment 1: Edge from pt1 to pt2
                # Define line segment 2: Perpendicular line from centroid in both directions

                # First point of perpendicular line - make it even longer to ensure intersections
                perp_line_start = centroid_point - perp_vector * 5000
                # Second point of perpendicular line
                perp_line_end = centroid_point + perp_vector * 5000

                # Calculate direction vectors
                d1 = pt2 - pt1  # Edge direction
                d2 = perp_line_end - perp_line_start  # Perpendicular line direction

                # Calculate the cross product to determine parallelism
                cross_product = d1[0] * d2[1] - d1[1] * d2[0]

                # If lines are not parallel (cross product not near zero)
                if abs(cross_product) > 1e-10:
                    try:
                        # Calculate parameters for intersection point using more robust formula
                        # We're solving for parameters s and t where:
                        # pt1 + s * d1 = perp_line_start + t * d2
                        
                        # Calculate vector between starting points
                        delta_p = pt1 - perp_line_start
                        
                        # Calculate parameters using Cramer's rule
                        s = (delta_p[1] * d2[0] - delta_p[0] * d2[1]) / cross_product
                        t = (delta_p[1] * d1[0] - delta_p[0] * d1[1]) / cross_product
                        
                        # Check if intersection is within the edge segment (0≤s≤1)
                        # For the perpendicular line, we don't need bounds since it's very long
                        if 0 <= s <= 1:
                            # Calculate the intersection point
                            intersection_point = pt1 + s * d1
                            
                            # Score based on how close to the middle of the edge the intersection is
                            # 1.0 means perfect bisection (middle of edge)
                            intersection_score = 1-abs(0.5 - s)
                            
                            # Draw the intersection point for debugging
                            cv2.circle(cvImage, tuple(map(int, intersection_point)), 
                                    radius=5, color=(0, 255, 255), thickness=2)
                            
                            # Draw a line from the intersection to the edge midpoint
                            cv2.line(cvImage, tuple(map(int, intersection_point)), 
                                   tuple(map(int, edge_midpoint)), (255, 255, 0), 1)
                    except:
                        # Skip calculation if numerical issues
                        pass
                
                edges.append({
                    'idx1': i,
                    'idx2': next_i,
                    'pt1': pt1,
                    'pt2': pt2,
                    'midpoint': edge_midpoint,
                    'length': edge_length,
                    'side': np.sign(side_projection),
                    'perp_alignment': perp_alignment,
                    'perp_dist': perp_dist,
                    'intersection_score': intersection_score,
                    'intersection_point': intersection_point
                })
            
            # Step 3: Group edges by side (positive or negative perpendicular projection)
            pos_side_edges = [e for e in edges if e['side'] > 0]
            neg_side_edges = [e for e in edges if e['side'] < 0]
            
            # Draw the perpendicular bisector line for visualization
            # Calculate a point 200 pixels along the perpendicular in each direction from centroid
            perp_pt1 = centroid_point + perp_vector * 200
            perp_pt2 = centroid_point - perp_vector * 200
            cv2.line(cvImage, tuple(map(int, perp_pt1)), tuple(map(int, perp_pt2)), (255, 255, 255), 1)
            
            # Draw the centroid point
            cv2.circle(cvImage, (int(cx), int(cy)), radius=6, color=(0, 0, 255), thickness=-1)
            cv2.putText(cvImage, "Centroid", (int(cx) + 10, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Step 4: Find the best edge on each side that:
            # 1. Is most likely to be intersected by the perpendicular from the centroid
            # 2. Is reasonably perpendicular to the main axis
            # 3. Is not too short (filters out noise)
            best_pos_edge = None
            best_neg_edge = None
            
            # Draw all edges with their scores for debugging
            for edge in edges:
                midpt = tuple(map(int, edge['midpoint']))
                edge_color = (100, 100, 250) if edge['side'] > 0 else (250, 100, 100)
                
                # Display tiny numbers showing intersection scores
                if edge['intersection_score'] != 0:
                    cv2.putText(cvImage, f"{edge['intersection_score']:.1f}", 
                               midpt, cv2.FONT_HERSHEY_SIMPLEX, 1, edge_color, 1)
                    # cv2.waitKey(100)  # Update the display
            # Minimum length threshold to avoid detecting noise as tips
            min_length_threshold = main_axis_norm * 0.00  # 5% of main axis length
            
            # For positive side edges
            if pos_side_edges:
                # Sort by intersection score (prefer edges intersected by perpendicular from centroid)
                pos_candidates = [e for e in pos_side_edges if e['length'] > min_length_threshold]
                pos_candidates.sort(key=lambda e: (-e['intersection_score'], e['perp_alignment'], e['length']))
                
                if pos_candidates:
                    best_pos_edge = pos_candidates[0]
            
            # For negative side edges
            if neg_side_edges:
                # Sort by intersection score (prefer edges intersected by perpendicular from centroid)
                neg_candidates = [e for e in neg_side_edges if e['length'] > min_length_threshold]
                neg_candidates.sort(key=lambda e: (-e['intersection_score']))#, e['perp_alignment'], e['length']))
                
                if neg_candidates:
                    best_neg_edge = neg_candidates[0]
            
            # Step 5: If we found edges on both sides, use them
            if best_pos_edge and best_neg_edge:
                # Set the tips using our selected edges
                tipl = best_pos_edge['pt1']
                tipr = best_pos_edge['pt2']
                tip2l = best_neg_edge['pt1']
                tip2r = best_neg_edge['pt2']
                
                # However, we want to find the smaller of these two edges
                # The length of each edge
                pos_edge_len = best_pos_edge['length']
                neg_edge_len = best_neg_edge['length']
                
                # Choose the smaller edge as our tip pair
                if neg_edge_len < pos_edge_len:
                    tipl, tipr = tip2l, tip2r
                
                # Calculate midpoint of tips for angle calculation
                mid_tips = (tipl + tipr) / 2
                
                # For angle calculation, use centroid and tip midpoint
                mid2 = centroid_point  # Centroid point
                mid1 = mid_tips  # Tips midpoint
            
            # Draw the directional arrow (from base midpoint to tip midpoint)
            mid1_tuple = tuple(mid1.astype(int))
            mid2_tuple = tuple(mid2.astype(int))
            cv2.arrowedLine(cvImage, mid2_tuple, mid1_tuple, openColor, 4)
        
            # Calculate angle from the direction vector
            v = mid1 - mid2  # Vector pointing from base to tips
            angle_vector = np.arctan2(v[1], v[0])
            angle_vector = np.degrees(angle_vector)
            
            # Calculate statistics for angles
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)
            
            # Calculate smallest angle difference accounting for wraparound
            angle_diff = min(abs(angle_vector - mean_angle), 360 - abs(angle_vector - mean_angle))
            z_angle = angle_diff / std_angle if std_angle > 0 else 10
            
            if z_angle > angle_threshold:
                print(f"Angle outlier rejected: {angle_vector:.1f}° - z-score: {z_angle:.2f}")
                skipped_counter += 0.1
                return cvImage, openColor, centroids, angles, areas, openlengths, timestamps
            else:
                centroids.append((cx,cy))
                areas.append(area)
                angles.append(angle_vector)
                timestamps.append(timestamp)
                skipped_counter = 0
                
            if len(tipl) != 0 and len(tipr) != 0:
                openlength = np.linalg.norm(tipl - tipr)*1000/150 # Convert to um if needed, adjust scaling factor as per your calibration
                if len(openlengths) >= 5:
                    avg_openlength = np.mean(openlengths)
                    std_openlength = np.std(openlengths)
                    if std_openlength > 0:
                        z_openlength = abs(openlength - avg_openlength) / std_openlength
                        if z_openlength <= 5.0:  # Accept if within 3 standard deviations
                            openlengths.append(openlength)
                            openlengths.pop(0)
                    else:
                        openlengths.append(openlength)
                        openlengths.pop(0)
                else:
                    openlengths.append(openlength)
                # print(f"cx,cy:({cx},{cy})")

                # Draw visualization elements
                tipl_tuple = tuple(tipl.astype(int))
                tipr_tuple = tuple(tipr.astype(int))
                centroidtuple = tuple(map(int, centroids[-1]))
                
                # Draw tip points
                cv2.circle(cvImage, tipl_tuple, radius=6, color=openColor, thickness=-1)
                cv2.circle(cvImage, tipr_tuple, radius=6, color=openColor, thickness=-1)
                
                # Draw centroid
                cv2.circle(cvImage, centroidtuple, radius=6, color=(0, 225, 225), thickness=-1)
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
                cv2.putText(cvImage, f"Opening: {openlength:.1f}px", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
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
    return cvImage, openColor, centroids, angles, areas, openlengths, timestamps

def publish_pose(publisher, x, y, theta, opening, timestamp=None):
        try:
            # print(f"x,y:({x},{y})")

            # Convert to mm 
            x = x * PixelToMM
            y = y * PixelToMM
            
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
            pose_msg.data = [x, y, x, qx, qy, qz, qw, opening, timestamp.to_sec()]
            
            # Publish the message
            publisher.publish(pose_msg)
        except Exception as e:
            print(f"Error publishing pose: {e}")

def image_callback(msg):
    bridge = CvBridge()
    global centroids, angles, areas, publisher, openlengths, last_image_time, timestamps
    
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
    
    processed_img, openColor, centroids, angles, areas, openlengths, timestamps = microgripperDetection(cv_image, timestamp_sec, openColor, centroids, angles, areas, openlengths, timestamps)
    
    if (processed_img is not None and centroids):
        if timestamps[-1] == timestamp_sec:
            x = centroids[-1][0] - cv_image.shape[1] / 2  # Adjust x to have 0,0 at the center of the image
            y = -centroids[-1][1] + cv_image.shape[0] / 2  # Adjust y to have 0,0 at the center of the image
            publish_pose(publisher, x, y, angles[-1], openlengths[-1], timestamp)
            cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
            if cv2.waitKey(3) & 0xFF == ord(' '):
                cv2.destroyAllWindows()
    # elif processed_img is not None:
    #     cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))


def main():
    rospy.init_node('image_processor_node', anonymous=True)
    
    # Initialize global variables
    global centroids, angles, prev_contour_area, publisher, areas, openlengths, last_image_time, timestamps
    openlengths = [0]
    areas = []
    centroids = []
    angles = []
    timestamps = []
    prev_contour_area = None
    last_image_time = time.time()
    
    # Set up the timeout timer to check every 0.5 seconds
    rospy.Timer(rospy.Duration(0.5), timeout_callback)
    
    rospy.Subscriber("/camera/basler_camera_1/image_raw", Image, image_callback)
    publisher = rospy.Publisher('/vision_feedback/pose_estimation', Float64MultiArray, queue_size=10)
    
    rospy.loginfo("MicroGripper Vision Feedback started. Will quit if no images received for {} seconds.".format(image_timeout))
    
    rospy.spin()   

if __name__ == "__main__":
    main()


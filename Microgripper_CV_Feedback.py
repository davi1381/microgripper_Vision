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

# New global variables for timestamp-based velocity estimation
last_timestamps = []
last_positions = []
last_angles = []
skiped_counter = 0

PixelToMM = 2.5 / 1280.0  # Conversion factor from pixels to mm (assuming 1280 pixels in width)


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
    MAX_AREA = 100000 
    MIN_AREA = 25000 
    

    kernel = np.ones((3,3))
    SEARCH_AREA = 175
    cropping = False
    bot_rect = None
    j = 0
    area_threshold = 10.0e6  # 20% threshold for contour area change
    
    # Position and angle outlier rejection thresholds
    position_threshold = 10.0e6 # Standard deviations
    angle_threshold = 10.0e6     # Standard deviations
    
    # 14umTest1 works
    # 14umTest3 works
    
    start_time = time.time()
    j = j + 1
    
    #color = cv2.resize(color, (1280, 720), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(cvImage, cv2.COLOR_BGR2GRAY)
    
    # Adjust picture contrast / equalize it? 
    #blurred = cv2.bilateralFilter(frame,5,10,10)
    edges = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,135,-6)
    
    edges = cv2.erode(edges, kernel)
    edges = cv2.dilate(edges, kernel, iterations=5)
    edges = cv2.erode(edges, kernel, iterations=3)
    #edges = cv2.dilate(edges, kernel, iterations=1)
    
    #edges = cv2.bitwise_or(edges, edges_orig)
    
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
    
    if contours:
        #! error check contours size
        sortedPairs = sorted(zip(contours, hierarchy[0]), key=lambda pair: cv2.contourArea(pair[0]), reverse=True)[:6]
        contours, hierarchy = zip(*(sortedPairs)) # adjust number of accepted contours (10)
        for i in range(0, len(contours)):
            hull = cv2.convexHull(contours[i])
            simple_hull = cv2.approxPolyDP(hull, 0.013 * cv2.arcLength(hull, True), True)
            if len(simple_hull) >= 3 and len(simple_hull) < 15: # was for ellipse fitting - speeds up processing but maybe remove
                # cur_cnt = cv2.approxPolyDP(contours[i], 0.03 * cv2.arcLength(contours[i], True), True)
                rect = cv2.minAreaRect(contours[i])
                (cx, cy), (width, height), angle = rect  
                #angle = angle+90
                if width < height:
                    width, height = height, width
                    #print("new width:", width)
                else:
                    angle = angle-90 
                area = cv2.contourArea(contours[i])
                #cv2.drawContours(color, [cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)], 0, openColor, 2)
                if width < 1.75*height and area < MAX_AREA and area > MIN_AREA: # adjust from 1.5
                    # Calculate contour area
                    for i in [1]:
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
                                
                                # # Define maximum allowed deviation from prediction
                                # max_position_deviation = 50  # pixels
                                # max_angle_deviation = 30  # degrees
                                if angle_diff > 20 and False:
                                    print(f"Angle outlier rejected: {angle:.1f}° - too far from previous: {angle_diff:.2f}°")
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
                                if z_area > area_threshold:  # 3 standard deviations
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
                        if skipped_counter > 25:
                            centroids.pop(0)
                            angles.pop(0)
                            areas.pop(0)
                        continue
                                            
                    if len(centroids) < 5:
                        bot_rect = rect
                        bot_cnt = contours[i]
                        openColor = (0, 255, 0)  # green
                        #print(rect)
                        break
                    else:
                        #cropping = True
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
                    
        #field = cv2.approxPolyDP(contours[0], 0.11 * cv2.arcLength(contours[0], True), True)    # add this in maybe 
        
        if bot_rect:   
            thetas = []    
            lengths = []   
            for i in range(len(simple_hull)):
                v1 = (simple_hull[i-1] - simple_hull[i]).flatten()
                v2 = (simple_hull[(i+1) % len(simple_hull)] - simple_hull[i]).flatten()
                theta = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                theta = np.degrees(np.arccos(theta))
                thetas.append((theta,i))
                lengths.append((np.linalg.norm(v1),i))
                
            #print(thetas)
            #print("lens", lengths)
            maxLength = np.max(lengths)
            
            simple_hull = np.array(simple_hull).reshape(-1, 2)
    
            max_dist = 0
            side1, side2 = None, None
            
            for i in range(len(simple_hull)):
                for j in range(i + 1, len(simple_hull)):  # Avoid redundant comparisons
                    dist = np.linalg.norm(simple_hull[i] - simple_hull[j])
                    if dist > max_dist:
                        max_dist = dist
                        side1, side2 = i, j
            
            left = lengths[side1][0]
            right = lengths[(side1+1) % len(lengths)][0] 
            if (left > right):
                tipr = simple_hull[side1-1]  
                tipl = simple_hull[(side2+1) % len(lengths)]  
            else:
                tipl = simple_hull[(side1+1) % len(lengths)]
                tipr = simple_hull[side2-1]  
            
            mid2 = (simple_hull[side1] + simple_hull[side2]) / 2
            mid1 = (tipl + tipr) / 2
            
            # Fix: Convert numpy arrays to tuples when passing to cv2.line
            mid1_tuple = tuple(mid1.astype(int))
            mid2_tuple = tuple(mid2.astype(int))
            cv2.arrowedLine(cvImage, mid2_tuple, mid1_tuple, openColor, 4)
        
            v = -mid1 + mid2
            angle_vector = np.arctan2(v[1], v[0])
            angle_vector = np.degrees(angle_vector)
            
            # Calculate statistics for angles
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)

            # Calculate statistics for angles
            mean_angle = np.mean(angles)
            std_angle = np.std(angles)

            # Calculate smallest angle difference accounting for wraparound
            angle_diff = min(abs(angle_vector - mean_angle), 360 + abs(angle_vector - mean_angle))
            z_angle = angle_diff / std_angle if std_angle > 0 else 10
            if z_angle > angle_threshold:
                print(f"Angle outlier rejected: {angle:.1f}° - z-score: {z_angle:.2f}")
                skipped_counter +=1
                return cvImage, openColor, centroids, angles, areas, openlengths

            else:
                centroids.append((cx,cy))
                areas.append(area)
                angles.append(angle_vector)
                timestamps.append(timestamp)
                skipped_counter = 0

            if len(tipl) != 0:  
                openlength = np.linalg.norm(tipl-tipr) 
                if len(openlengths)>=5:
                    avg_openlength = np.mean(openlengths) 
                    z_openlength = abs(openlength - avg_openlength) / np.std(openlengths)
                    if z_openlength > 3.0:  # 3 standard deviations
                        openlengths.append(openlength)
                    openlengths.pop(0)

                # Fix: Convert numpy arrays to tuples for cv2.circle
                tipl_tuple = tuple(tipl.astype(int))
                tipr_tuple = tuple(tipr.astype(int))
                centroidtuple = tuple(map(int,centroids[-1]))


                cv2.circle(cvImage, tipl_tuple, radius = 6, color=openColor, thickness= -1)
                cv2.circle(cvImage, tipr_tuple, radius = 6, color=openColor, thickness= -1)
                cv2.circle(cvImage, centroidtuple, radius = 6, color=(0,225,225), thickness= -1)
                cv2.drawContours(cvImage, [simple_hull], 0, openColor, 2)    
    else:
        print("No robot contours found.")
    
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
    
    #cv2.imshow("Video", color)
    #cv2.imshow("Edges", edges)
    #cv2.imshow("Histogram", equ)
    end_time = time.time()
    #print((end_time-start_time)*1000)
    return cvImage, openColor, centroids, angles, areas, openlengths, timestamps

def publish_pose(publisher, x, y, theta, opening, timestamp=None):
        # Convert the orientation to quaternion (only around z-axis)
        try:
            # x and y are already centered (adjusted in image_callback)
            
            # Convert to mm 
            x = x * PixelToMM
            y = y * PixelToMM
            
            # convert to radians
            print("theta", theta)
            theta = math.radians(90-theta)
            
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
        if timestamps[-1]==timestamp_sec:
            x = centroids[-1][0] - cv_image.shape[1] / 2  # Adjust x to have 0,0 at the center of the image
            y = centroids[-1][1] - cv_image.shape[0] / 2  # Adjust y to have 0,0 at the center of the image
            publish_pose(publisher, x, y, angles[-1], openlengths[-1], timestamp)
            cv2.imshow("Processed Image", cv2.resize(processed_img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA))
            if cv2.waitKey(2) & 0xFF == ord(' '):
                cv2.destroyAllWindows()

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


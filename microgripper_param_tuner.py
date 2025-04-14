#!/usr/bin/env python

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, Scale
import os
import sys
import time
import threading

# Import ROS-related modules
try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    print("ROS packages not found - ROS functionality will be disabled")
    ROS_AVAILABLE = False
    # Define placeholder classes for type checking
    class CvBridge:
        def imgmsg_to_cv2(self, *args, **kwargs):
            return None
        


class MicrogripperParamTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Microgripper Detection Parameter Tuner")
        self.root.geometry("800x600")
        try:
            from CVParameters3 import MICROGRIPPER_PARAMS
            self.params = MICROGRIPPER_PARAMS
            print("Using saved parameters from CVParameters1.py")
        except ImportError:
            # Parameters with default values
            self.params = {
                # Thresholding params
                # Adaptive thresholding params
                "adaptive_block_size": 135,
                "adaptive_constant": -6,
                
                # Morphological operations
                "erode_iterations": 1,
                "dilate1_iterations": 5,
                "erode2_iterations": 3,
                "dilate2_iterations": 3,
                
                # Contour processing
                "hull_epsilon": 0.013,
                "min_hull_points": 3,
                "max_hull_points": 15,
                "aspect_ratio": 1.75,
                "min_area": 25000,
                "max_area": 100000,
                
                # Edge detection (alternative method)
                "use_canny": False,
                "canny_threshold1": 100,
                "canny_threshold2": 200,
                
                # Other options
                "use_bilateral_filter": False,
                "bilateral_d": 5,
                "bilateral_sigma_color": 10,
                "bilateral_sigma_space": 10,
                
                # Square detection parameters
                "square_blur_size": 1,
                "square_canny_threshold1": 100,
                "square_canny_threshold2": 400,
                "square_epsilon": 60,
                "square_dilate_iterations": 1,
                "square_min_area": 5000,
                "square_max_area": 10000,
                "square_aspect_ratio": 1.3,
                "square_brightness_ratio": 1.5
            }
            
        # Ensure all required parameters are initialized, even if they weren't in the loaded file
        # This prevents KeyError when parameters are added to new versions
        default_params = {
            # Square detection parameters
            "square_blur_size": 1,
            "square_canny_threshold1": 100,
            "square_canny_threshold2": 400,
            "square_epsilon": 60,
            "square_dilate_iterations": 1,
            "square_min_area": 5000,
            "square_max_area": 10000,
            "square_aspect_ratio": 1.3,
            "square_brightness_ratio": 1.5
        }
        
        # Add any missing parameters from default_params
        for key, value in default_params.items():
            if key not in self.params:
                self.params[key] = value
        
        # ROS-related variables
        self.ros_initialized = False
        self.bridge = None
        self.image_sub = None
        self.ros_image = None
        self.ros_lock = threading.Lock()
        self.ros_topic = "/camera/basler_camera_1/image_raw"  # Default topic
        
        # Trackbar dictionary to store references
        self.trackbars = {}
        
        # Create GUI components
        self.create_ui()
        
        # Image display variables
        self.image = None
        self.processed_image = None
        
        # Flag to control the OpenCV window display loop
        self.running = True
        
        # Initialize ROS if available
        if ROS_AVAILABLE:
            try:
                self.init_ros()
            except Exception as e:
                print(f"Failed to initialize ROS: {e}")
        
        # Start the update loop for processing images
        self.update_loop()
        
        # Load initial image if ROS is not available
        if not ROS_AVAILABLE:
            self.load_image()

    def init_ros(self):
        """Initialize ROS node and subscribers"""
        if not rospy.core.is_initialized():
            rospy.init_node('microgripper_param_tuner', anonymous=True, disable_signals=True)
            self.ros_initialized = True
        
        self.bridge = CvBridge()
        self.subscribe_to_ros_topic(self.ros_topic)
        print(f"ROS initialized and subscribed to {self.ros_topic}")
    def find_bright_squares(self, image):
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
        blur_size = self.params["square_blur_size"]
        if blur_size % 2 == 0:  # Must be odd
            blur_size += 1
            
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Edge detection to find borders
        edges = cv2.Canny(blurred, 
                         self.params["square_canny_threshold1"], 
                         self.params["square_canny_threshold2"])
        
        # Dilate slightly to connect any broken edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=self.params["square_dilate_iterations"])
        
        # Find contours from the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw all contours on a blank image for debugging
        debug_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        
        # Filter contours to find squares of similar size
        potential_squares = []
        for contour in contours:
            # Calculate area
            hull = cv2.convexHull(contour)

            area = cv2.contourArea(hull)
            
            # Skip contours outside area range
            if area < self.params["square_min_area"] or area > self.params["square_max_area"]:
                continue
            
            # Approximate the contour to a polygon
            epsilon = self.params["square_epsilon"]/1000.0 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # Draw both the original contour and the approximation in different colors
            cv2.drawContours(debug_image, [contour], -1, (0, 255, 0), 1)  # Original in green
            cv2.drawContours(debug_image, [approx], -1, (255, 0, 255), 1)  # Approximation in magenta
            # Add text showing the number of points
            cv2.putText(debug_image, f"Points: {len(approx)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Check if it's approximately a square (4 vertices)
            if len(approx) >=3 and len(approx) <=6:
                # Get the bounding rectangle
                rect = cv2.minAreaRect(approx)
                (cx, cy), (width, height), angle = rect
                
                # Check if it's square-like (aspect ratio near 1)
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio < self.params["square_aspect_ratio"]:
                    # Calculate average brightness inside the contour
                    mask = np.zeros_like(gray)
                    cv2.drawContours(mask, [hull], 0, 255, -1)
                    mean_brightness = cv2.mean(gray, mask=mask)[0]
                    
                    # Calculate mean brightness of whole image for comparison
                    mean_image_brightness = np.mean(gray)
                    
                    # Add to potential squares if bright enough compared to image average
                    brightness_ratio = self.params.get("square_brightness_ratio", 1.5)
                    if mean_brightness > brightness_ratio * mean_image_brightness:
                        potential_squares.append({
                            'contour': hull,
                            'rect': rect,
                            'area': area,
                            'center': (cx, cy),
                            'brightness': mean_brightness
                        })
        cv2.imshow("Debug Image", debug_image)
        cv2.waitKey(1)
        # Sort by area
        potential_squares.sort(key=lambda x: x['area'], reverse=True)
        
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
            print(f"Found squares with areas: {sq1['area']:.1f} and {sq2['area']:.1f}")
            return [(sq1['center'], sq1['rect']), (sq2['center'], sq2['rect'])]
        else:
            return []
        
    def subscribe_to_ros_topic(self, topic):
        """Subscribe to a ROS image topic"""
        # Unsubscribe from current topic if any
        if self.image_sub is not None:
            self.image_sub.unregister()
        
        # Subscribe to new topic
        try:
            self.image_sub = rospy.Subscriber(
                topic,
                Image,
                self.ros_image_callback,
                queue_size=1
            )
            self.ros_topic = topic
            print(f"Subscribed to {topic}")
        except Exception as e:
            print(f"Failed to subscribe to {topic}: {e}")

    def ros_image_callback(self, msg):
        """Callback function for ROS image messages"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.ros_lock:
                self.ros_image = cv_image
        except Exception as e:
            print(f"Error processing ROS image: {e}")

    def create_ui(self):
        # Create main frame for controls (the images will be displayed with OpenCV)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add tab control for organizing parameters
        tab_control = ttk.Notebook(main_frame)
        tab_control.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different parameter groups
        main_tab = ttk.Frame(tab_control)
        squares_tab = ttk.Frame(tab_control)
        
        tab_control.add(main_tab, text="Microgripper Detection")
        tab_control.add(squares_tab, text="Square Detection")
        
        # Control panel for main parameters
        control_frame = ttk.LabelFrame(main_tab, text="Parameters")
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollable canvas for parameters (for small screens)
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        
        scroll_frame.bind(
            "<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add ROS topic input if ROS is available
        if ROS_AVAILABLE:
            ros_frame = ttk.LabelFrame(scroll_frame, text="ROS Settings")
            ros_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(ros_frame, text="ROS Topic:").pack(side=tk.LEFT, padx=5)
            self.topic_var = tk.StringVar(value=self.ros_topic)
            topic_entry = ttk.Entry(ros_frame, textvariable=self.topic_var, width=30)
            topic_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            ttk.Button(ros_frame, text="Subscribe", 
                      command=lambda: self.subscribe_to_ros_topic(self.topic_var.get())).pack(side=tk.RIGHT, padx=5)
            
            # Add a checkbox to enable/disable ROS feed
            self.ros_enabled_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(ros_frame, text="Enable ROS Feed", 
                           variable=self.ros_enabled_var).pack(anchor=tk.W, padx=5)

        # Group thresholding parameters
        thresh_frame = ttk.LabelFrame(scroll_frame, text="Thresholding")
        thresh_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Adaptive thresholding params
        self.add_trackbar(thresh_frame, "Block Size", "adaptive_block_size", 3, 301, 2)
        self.add_trackbar(thresh_frame, "Constant", "adaptive_constant", -20, 20, 1)
        
        # Edge detection option
        edge_frame = ttk.LabelFrame(scroll_frame, text="Edge Detection")
        edge_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.canny_var = tk.BooleanVar(value=self.params["use_canny"])
        ttk.Checkbutton(edge_frame, text="Use Canny Edge Detection", 
                        variable=self.canny_var).pack(anchor=tk.W)
        
        self.add_trackbar(edge_frame, "Threshold1", "canny_threshold1", 0, 500, 1)
        self.add_trackbar(edge_frame, "Threshold2", "canny_threshold2", 0, 500, 1)
        
        # Morphology parameters
        morph_frame = ttk.LabelFrame(scroll_frame, text="Morphological Operations")
        morph_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.add_trackbar(morph_frame, "Erode Iterations", "erode_iterations", 0, 10, 1)
        self.add_trackbar(morph_frame, "Dilate1 Iterations", "dilate1_iterations", 0, 10, 1)
        self.add_trackbar(morph_frame, "Erode2 Iterations", "erode2_iterations", 0, 10, 1)
        self.add_trackbar(morph_frame, "Dilate2 Iterations", "dilate2_iterations", 0, 10, 1)
        
        # Bilateral filter option
        filter_frame = ttk.LabelFrame(scroll_frame, text="Filtering")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.bilateral_var = tk.BooleanVar(value=self.params["use_bilateral_filter"])
        ttk.Checkbutton(filter_frame, text="Use Bilateral Filter", 
                        variable=self.bilateral_var).pack(anchor=tk.W)
        
        self.add_trackbar(filter_frame, "Diameter", "bilateral_d", 1, 20, 2)
        self.add_trackbar(filter_frame, "Sigma Color", "bilateral_sigma_color", 1, 100, 1)
        self.add_trackbar(filter_frame, "Sigma Space", "bilateral_sigma_space", 1, 100, 1)
        
        # Contour parameters
        contour_frame = ttk.LabelFrame(scroll_frame, text="Contour Processing")
        contour_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.add_trackbar(contour_frame, "Hull Epsilon (x0.001)", "hull_epsilon", 1, 50, lambda x: x/1000)
        self.add_trackbar(contour_frame, "Min Hull Points", "min_hull_points", 3, 10, 1)
        self.add_trackbar(contour_frame, "Max Hull Points", "max_hull_points", 5, 30, 1)
        self.add_trackbar(contour_frame, "Aspect Ratio (x0.1)", "aspect_ratio", 10, 50, lambda x: x/10)
        self.add_trackbar(contour_frame, "Min Area (x1000)", "min_area", 1, 100, lambda x: x*1000)
        self.add_trackbar(contour_frame, "Max Area (x1000)", "max_area", 10, 500, lambda x: x*1000)
        
        # Status bar
        status_frame = ttk.Frame(scroll_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        
        # Action buttons
        button_frame = ttk.Frame(scroll_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Load Image", 
                   command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Parameters", 
                   command=self.save_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset Parameters", 
                   command=self.reset_parameters).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Save Current Image", 
                   command=self.save_current_image).pack(side=tk.RIGHT, padx=5)

        # Setup Square Detection tab
        square_control_frame = ttk.LabelFrame(squares_tab, text="Square Detection Parameters")
        square_control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add scrollable canvas for square detection parameters
        square_canvas = tk.Canvas(square_control_frame)
        square_scrollbar = ttk.Scrollbar(square_control_frame, orient=tk.VERTICAL, command=square_canvas.yview)
        square_scroll_frame = ttk.Frame(square_canvas)
        
        square_scroll_frame.bind(
            "<Configure>", 
            lambda e: square_canvas.configure(scrollregion=square_canvas.bbox("all"))
        )
        square_canvas.create_window((0, 0), window=square_scroll_frame, anchor=tk.NW)
        square_canvas.configure(yscrollcommand=square_scrollbar.set)
        
        square_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        square_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add square detection parameters
        if "square_blur_size" not in self.params:
            # Add default parameters for square detection if they don't exist
            self.params.update({
                "square_blur_size": 1,
                "square_canny_threshold1": 100,
                "square_canny_threshold2": 400,
                "square_epsilon": 60,
                "square_dilate_iterations": 1,
                "square_min_area": 5000,
                "square_max_area": 10000,
                "square_aspect_ratio": 1.5,
                "square_brightness_ratio": 1.5
            })
        
        # Create parameter groups for square detection
        preprocessing_frame = ttk.LabelFrame(square_scroll_frame, text="Preprocessing")
        preprocessing_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.add_trackbar(preprocessing_frame, "Blur Size (must be odd)", "square_blur_size", 1, 11, 2)
        self.add_trackbar(preprocessing_frame, "Canny Threshold 1", "square_canny_threshold1", 50, 300, 1)
        self.add_trackbar(preprocessing_frame, "Canny Threshold 2", "square_canny_threshold2", 100, 600, 1)
        self.add_trackbar(preprocessing_frame, "epsilon", "square_epsilon", 0, 1000, 1)
        self.add_trackbar(preprocessing_frame, "Dilate Iterations", "square_dilate_iterations", 0, 5, 1)
        
        filtering_frame = ttk.LabelFrame(square_scroll_frame, text="Square Filtering")
        filtering_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.add_trackbar(filtering_frame, "Minimum Area", "square_min_area", 500, 5000, 1)
        self.add_trackbar(filtering_frame, "Maximum Area", "square_max_area", 3000, 10000, 1)
        self.add_trackbar(filtering_frame, "Max Aspect Ratio", "square_aspect_ratio", 10, 20, lambda x: x/10)
        self.add_trackbar(filtering_frame, "Min Brightness Ratio", "square_brightness_ratio", 10, 30, lambda x: x/10)
        
        # Status and action buttons for Square Detection tab
        square_status_frame = ttk.Frame(square_scroll_frame)
        square_status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        square_button_frame = ttk.Frame(square_scroll_frame)
        square_button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(square_button_frame, text="Test Current Image", 
                  command=self.update_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(square_button_frame, text="Save Parameters", 
                  command=self.save_parameters).pack(side=tk.LEFT, padx=5)

    def add_trackbar(self, parent, label_text, param_name, min_val, max_val, scale_func):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(frame, text=label_text).pack(side=tk.LEFT)
        
        # Calculate dynamic range based on current parameter value
        current_val = self.params[param_name]
        if callable(scale_func):
            # For lambda functions, convert to display value
            display_val = int(current_val / scale_func(1)) if scale_func(1) != 0 else current_val
            # Set dynamic min/max to 1/10 and 10x the current value
            dynamic_min = max(1, int(display_val / 10))  # Avoid going below 1
            dynamic_max = int(display_val * 10)
            
            scale = tk.Scale(frame, from_=dynamic_min, to=dynamic_max, orient=tk.HORIZONTAL, 
                            length=200, command=lambda v, p=param_name, f=scale_func: self.update_param(p, float(v), f))
        else:
            # For simple integer scaling
            display_val = int(current_val / scale_func) if scale_func != 0 else current_val
            # Set dynamic min/max to 1/10 and 10x the current value
            dynamic_min = max(1, int(display_val / 10))  # Avoid going below 1
            dynamic_max = int(display_val * 10)
            
            scale = tk.Scale(frame, from_=dynamic_min, to=dynamic_max, orient=tk.HORIZONTAL, 
                            length=200, command=lambda v, p=param_name, f=scale_func: self.update_param(p, int(v), f))
        
        # Set the current value
        scale.set(display_val)
        scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Store reference to scale widget
        self.trackbars[param_name] = scale

    def update_param(self, param_name, value, scale_func):
        if callable(scale_func):
            self.params[param_name] = scale_func(value)
        else:
            self.params[param_name] = value * scale_func

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        
        if file_path:
            try:
                self.image = cv2.imread(file_path)
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error loading image: {e}")
                self.status_var.set(f"Error loading image: {e}")

    def save_current_image(self):
        """Save the current processed image to a file"""
        if self.processed_image is None:
            self.status_var.set("No processed image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=".png",
            filetypes=(
                ("PNG files", "*.png"), 
                ("JPEG files", "*.jpg"), 
                ("All files", "*.*")
            )
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                self.status_var.set(f"Saved processed image to {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Error saving image: {e}")
                self.status_var.set(f"Error saving image: {e}")

    def update_processing(self):
        """Process the current image with current parameters"""
        # Get image to process (from ROS or loaded file)
        img_to_process = None
        
        if ROS_AVAILABLE and self.ros_enabled_var.get():
            with self.ros_lock:
                if self.ros_image is not None:
                    img_to_process = self.ros_image.copy()
        else:
            if self.image is not None:
                img_to_process = self.image.copy()
        
        if img_to_process is None:
            return
        
        start_time = time.time()
        
        # Get current parameter values from UI
        self.params["use_canny"] = self.canny_var.get()
        self.params["use_bilateral_filter"] = self.bilateral_var.get()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY)
        
        # Create a debug image to show square detection steps
        debug_squares = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        
        # Square detection preprocessing steps
        blur_size = self.params["square_blur_size"]
        if blur_size % 2 == 0:  # Must be odd
            blur_size += 1
            
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Edge detection to find borders
        square_edges = cv2.Canny(blurred, 
                             self.params["square_canny_threshold1"], 
                             self.params["square_canny_threshold2"])
        
        # Dilate slightly to connect any broken edges
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(square_edges, kernel, 
                               iterations=self.params["square_dilate_iterations"])
        
        # Find bright squares for registration/calibration
        square_results = self.find_bright_squares(gray)
        
        # Copy the input image for visualization
        result = img_to_process.copy()
        
        # Add square detection debugging to debug_squares image
        debug_squares[:, :, 0] = dilated_edges  # Blue channel - dilated edges
        debug_squares[:, :, 1] = square_edges   # Green channel - original edges
        
        # Draw the detected squares with distinctive appearance
        if square_results:
            for i, (center, rect) in enumerate(square_results):
                # Convert data types for visualization
                cx, cy = center
                cx, cy = int(cx), int(cy)
                
                # Draw box around square in magenta on both result and debug
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(result, [box], 0, (255, 0, 255), 2)  # Magenta
                cv2.drawContours(debug_squares, [box], 0, (255, 255, 255), 2)  # White
                
                # Draw center point
                cv2.circle(result, (cx, cy), 7, (255, 0, 255), -1)
                cv2.circle(debug_squares, (cx, cy), 7, (255, 255, 255), -1)
                
                # Add label
                cv2.putText(result, f"Square {i+1}", (cx - 30, cy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Show area
                area = cv2.contourArea(box)
                cv2.putText(result, f"Area: {int(area)}", (cx - 30, cy + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Apply bilateral filter if enabled
        if self.params["use_bilateral_filter"]:
            gray = cv2.bilateralFilter(
                gray,
                self.params["bilateral_d"],
                self.params["bilateral_sigma_color"],
                self.params["bilateral_sigma_space"]
            )
        
        # Edge detection
        if self.params["use_canny"]:
            edges = cv2.Canny(
                gray, 
                self.params["canny_threshold1"],
                self.params["canny_threshold2"]
            )
        else:
            # Adaptive thresholding
            block_size = self.params["adaptive_block_size"]
            if block_size % 2 == 0:  # Must be odd
                block_size += 1
                
            edges = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                self.params["adaptive_constant"]
            )
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        if self.params["erode_iterations"] > 0 and self.params["use_canny"] == False:
            edges = cv2.erode(edges, kernel, iterations=self.params["erode_iterations"])
        
        if self.params["dilate1_iterations"] > 0:
            edges = cv2.dilate(edges, kernel, iterations=self.params["dilate1_iterations"])
        
        if self.params["erode2_iterations"] > 0:
            edges = cv2.erode(edges, kernel, iterations=self.params["erode2_iterations"])
        
        if self.params["dilate2_iterations"] > 0:
            edges = cv2.dilate(edges, kernel, iterations=self.params["dilate2_iterations"])
        
        # Find and process contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort by contour area (largest first)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Process each contour up to a limit
            max_contours_to_check = min(6, len(sorted_contours))
            
            for i in range(max_contours_to_check):
                hull = cv2.convexHull(sorted_contours[i])
                hull_epsilon = self.params["hull_epsilon"] * cv2.arcLength(hull, True)
                simple_hull = cv2.approxPolyDP(hull, hull_epsilon, True)
                
                if (len(simple_hull) >= self.params["min_hull_points"] and 
                    len(simple_hull) <= self.params["max_hull_points"]):
                    
                    rect = cv2.minAreaRect(sorted_contours[i])
                    (cx, cy), (width, height), angle = rect
                    
                    M = cv2.moments(sorted_contours[i])
                    if M["m00"] != 0:
                        cx2 = int(M["m10"] / M["m00"])
                        cy2 = int(M["m01"] / M["m00"])

                    # Ensure width is the longer dimension
                    if width < height:
                        width, height = height, width
                    else:
                        angle = angle - 90
                    
                    area = cv2.contourArea(sorted_contours[i])
                    aspect_ratio = width / height if height > 0 else float('inf')
                    
                    # Check constraints
                    if (aspect_ratio < self.params["aspect_ratio"] and 
                        area < self.params["max_area"] and 
                        area > self.params["min_area"]):
                        
                        # Draw the contour and hull
                        cv2.drawContours(result, [sorted_contours[i]], -1, (0, 255, 0), 2)
                        cv2.drawContours(result, [simple_hull], -1, (255, 0, 0), 2)
                        
                        # Draw the rectangle
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
                        
                        # Draw the centroid
                        cv2.circle(result, (int(cx), int(cy)), 5, (255, 255, 0), -1)
                        cv2.circle(result, (int(cx2), int(cy2)), 5, (255, 155, 0), -1)

                        
                        # Add text with details
                        text = f"Area: {area:.0f}, AR: {aspect_ratio:.2f}, Angle: {angle:.1f}"
                        cv2.putText(result, text, (int(cx), int(cy) - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Try to identify the tips and base for the gripper
                        try:
                            simple_hull = np.array(simple_hull).reshape(-1, 2)
                            # Find the points furthest apart - main axis
                            max_dist = 0
                            side1, side2 = None, None
                            
                            for i in range(len(simple_hull)):
                                for j in range(i + 1, len(simple_hull)):
                                    dist = np.linalg.norm(simple_hull[i] - simple_hull[j])
                                    if dist > max_dist:
                                        max_dist = dist
                                        side1, side2 = i, j
                            
                            if side1 is not None and side2 is not None:
                                # Highlight the main axis points
                                p1 = tuple(map(int, simple_hull[side1]))
                                p2 = tuple(map(int, simple_hull[side2]))
                                cv2.circle(result, p1, 8, (0, 128, 255), -1)
                                cv2.circle(result, p2, 8, (0, 128, 255), -1)
                                cv2.line(result, p1, p2, (0, 128, 255), 2)
                        except Exception as e:
                            print(f"Error processing hull: {e}")
        
        # Display processing time
        processing_time = (time.time() - start_time) * 1000
        cv2.putText(result, f"Processing time: {processing_time:.1f} ms", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Create different display layouts based on active tab
        if hasattr(self, 'root') and hasattr(self.root, 'focus_get'):
            current_tab = self.root.focus_get()
            tab_name = current_tab.winfo_name() if current_tab else ""
            
            if "squares" in tab_name.lower():
                # For Square Detection tab, show square detection debug images
                edges_color = cv2.cvtColor(square_edges, cv2.COLOR_GRAY2BGR)
                dilated_color = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((result, debug_squares))
                cv2.imshow("Square Detection Steps", np.vstack((edges_color, dilated_color)))
            else:
                # For main tab, show edge detection results
                edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                combined = np.hstack((result, edges_color))
                if cv2.getWindowProperty("Square Detection Steps", cv2.WND_PROP_VISIBLE) > 0:
                    cv2.destroyWindow("Square Detection Steps")
        else:
            # Default display layout
            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((result, edges_color))
        
        # Store processed image
        self.processed_image = combined
        
        # Display using OpenCV windows
        if img_to_process is not None:
            cv2.imshow("Original", cv2.resize(img_to_process, (640, 480)))
        
        if self.processed_image is not None:
            # Resize if too large for screen
            h, w = self.processed_image.shape[:2]
            max_width = 1200
            if w > max_width:
                scale = max_width / w
                display_img = cv2.resize(self.processed_image, (int(w * scale), int(h * scale)))
            else:
                display_img = self.processed_image
                
            cv2.imshow("Processed", display_img)
        
        return combined

    def update_loop(self):
        """Main update loop for image processing"""
        try:
            if self.running:
                self.update_processing()
                
                # Process ROS messages if initialized
                if ROS_AVAILABLE and self.ros_initialized:
                    try:
                        rospy.rostime.wallsleep(0.001)
                    except:
                        pass
                
                # Process OpenCV window events
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.on_closing()
                    return
        except Exception as e:
            print(f"Error in update loop: {e}")
        
        # Schedule the next update
        if self.running:
            self.root.after(50, self.update_loop)  # Update at ~20 FPS

    def save_parameters(self):
        file_path = filedialog.asksaveasfilename(
            title="Save parameters",
            defaultextension=".py",
            filetypes=(("Python files", "*.py"), ("All files", "*.*"))
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write("# Microgripper Detection Parameters\n")
                f.write("# Generated on " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
                
                f.write("MICROGRIPPER_PARAMS = {\n")
                for key, value in self.params.items():
                    f.write(f"    '{key}': {value},\n")
                f.write("}\n")
                
                # Also write a function to apply these parameters
                f.write("""
def apply_parameters_to_image(image):
    \"\"\"
    Apply the saved parameters to process an image
    
    Args:
        image: OpenCV BGR image
        
    Returns:
        edges: Processed binary image
        result: Visualization of detection
    \"\"\"
    import cv2
    import numpy as np
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter if enabled
    if MICROGRIPPER_PARAMS['use_bilateral_filter']:
        gray = cv2.bilateralFilter(
            gray,
            MICROGRIPPER_PARAMS['bilateral_d'],
            MICROGRIPPER_PARAMS['bilateral_sigma_color'],
            MICROGRIPPER_PARAMS['bilateral_sigma_space']
        )
    
    # Edge detection
    if MICROGRIPPER_PARAMS['use_canny']:
        edges = cv2.Canny(
            gray, 
            MICROGRIPPER_PARAMS['canny_threshold1'],
            MICROGRIPPER_PARAMS['canny_threshold2']
        )
    else:
        # Adaptive thresholding
        block_size = MICROGRIPPER_PARAMS['adaptive_block_size']
        if block_size % 2 == 0:  # Must be odd
            block_size += 1
            
        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block_size,
            MICROGRIPPER_PARAMS['adaptive_constant']
        )
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    if MICROGRIPPER_PARAMS['erode_iterations'] > 0:
        edges = cv2.erode(edges, kernel, iterations=MICROGRIPPER_PARAMS['erode_iterations'])
    
    if MICROGRIPPER_PARAMS['dilate1_iterations'] > 0:
        edges = cv2.dilate(edges, kernel, iterations=MICROGRIPPER_PARAMS['dilate1_iterations'])
    
    if MICROGRIPPER_PARAMS['erode2_iterations'] > 0:
        edges = cv2.erode(edges, kernel, iterations=MICROGRIPPER_PARAMS['erode2_iterations'])
    
    if MICROGRIPPER_PARAMS['dilate2_iterations'] > 0:
        edges = cv2.dilate(edges, kernel, iterations=MICROGRIPPER_PARAMS['dilate2_iterations'])
    
    return edges
""")
                
            self.status_var.set(f"Parameters saved to {os.path.basename(file_path)}")
            print(f"Parameters saved to {file_path}")

    def reset_parameters(self):
        # Reset to default values
        self.params = {
            "adaptive_block_size": 135,
            "adaptive_constant": -6,
            "erode_iterations": 1,
            "dilate1_iterations": 5,
            "erode2_iterations": 3,
            "dilate2_iterations": 3,
            "hull_epsilon": 0.013,
            "min_hull_points": 3,
            "max_hull_points": 15,
            "aspect_ratio": 1.75,
            "min_area": 25000,
            "max_area": 100000,
            "use_canny": False,
            "canny_threshold1": 100,
            "canny_threshold2": 200,
            "use_bilateral_filter": False,
            "bilateral_d": 5,
            "bilateral_sigma_color": 10,
            "bilateral_sigma_space": 10,
            "square_blur_size": 3,
            "square_canny_threshold1": 100,
            "square_canny_threshold2": 400,
            "square_epsilon": 10,
            "square_dilate_iterations": 1,
            "square_min_area": 2000,
            "square_max_area": 5000,
            "square_aspect_ratio": 1.3,
            "square_min_brightness": 150
        }
        
        # Update trackbars to match default values
        for param_name, track in self.trackbars.items():
            if param_name == "hull_epsilon":
                track.set(int(self.params[param_name] * 1000))
            elif param_name == "aspect_ratio" or param_name == "square_aspect_ratio":
                track.set(int(self.params[param_name] * 10))
            elif param_name in ["min_area", "max_area", "square_min_area", "square_max_area"]:
                track.set(int(self.params[param_name] / 1000))
            else:
                track.set(self.params[param_name])
        
        # Update checkboxes
        self.canny_var.set(self.params["use_canny"])
        self.bilateral_var.set(self.params["use_bilateral_filter"])
        
        self.status_var.set("Parameters reset to default values")

    def on_closing(self):
        """Clean up resources when the application is closed"""
        self.running = False
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        if ROS_AVAILABLE and self.ros_initialized:
            # Ensure ROS is properly shut down
            try:
                rospy.signal_shutdown("Application closed")
            except:
                pass
                
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = MicrogripperParamTuner(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
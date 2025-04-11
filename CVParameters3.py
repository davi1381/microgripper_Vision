# Microgripper Detection Parameters
# Generated on 2025-04-10 19:17:29

MICROGRIPPER_PARAMS = {
    'adaptive_block_size': 566,
    'adaptive_constant': -6,
    'erode_iterations': 4,
    'dilate1_iterations': 8,
    'erode2_iterations': 4,
    'dilate2_iterations': 0,
    'hull_epsilon': 0.018,
    'min_hull_points': 3,
    'max_hull_points': 10,
    'aspect_ratio': 1.8,
    'min_area': 40000.0,
    'max_area': 65000.0,
    'use_canny': False,
    'canny_threshold1': 100,
    'canny_threshold2': 200,
    'use_bilateral_filter': False,
    'bilateral_d': 4,
    'bilateral_sigma_color': 10,
    'bilateral_sigma_space': 10,
    'square_blur_size': 3,
    'square_canny_threshold1': 100,
    'square_canny_threshold2': 400,
    'square_dilate_iterations': 1,
    'square_min_area': 5238,
    'square_max_area': 11300,
    'square_aspect_ratio': 1.3,
    'square_min_brightness': 150,
    'square_epsilon': 40,
    'square_brightness_ratio': 1,
}

def apply_parameters_to_image(image):
    """
    Apply the saved parameters to process an image
    
    Args:
        image: OpenCV BGR image
        
    Returns:
        edges: Processed binary image
        result: Visualization of detection
    """
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

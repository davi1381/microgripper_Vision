# Microgripper Detection Parameters
# Generated on 2025-03-27 23:48:40

MICROGRIPPER_PARAMS = {
    'adaptive_block_size': 546,
    'adaptive_constant': -9,
    'erode_iterations': 3,
    'dilate1_iterations': 9,
    'erode2_iterations': 10,
    'dilate2_iterations': 10,
    'hull_epsilon': 0.02,
    'min_hull_points': 4,
    'max_hull_points': 10,
    'aspect_ratio': 3.5,
    'min_area': 28000.0,
    'max_area': 71000.0,
    'use_canny': False,
    'canny_threshold1': 270,
    'canny_threshold2': 243,
    'use_bilateral_filter': False,
    'bilateral_d': 12,
    'bilateral_sigma_color': 5,
    'bilateral_sigma_space': 48,
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

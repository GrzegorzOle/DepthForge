#!/usr/bin/env python3
"""
Test script for DepthForge
"""

import cv2
import numpy as np
from pathlib import Path
import os

def create_sample_image():
    """Create a sample image for testing"""
    # Create a gradient image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a gradient from black to white
    for i in range(640):
        intensity = int(255 * i / 640)
        img[:, i] = [intensity, intensity, intensity]
    
    # Add some shapes to make it more interesting
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)
    cv2.line(img, (50, 400), (600, 400), (0, 0, 255), 5)
    
    return img

def main():
    # Create sample image
    sample_img = create_sample_image()
    
    # Save sample image
    sample_path = "data/sample_input.jpg"
    cv2.imwrite(sample_path, sample_img)
    print(f"Created sample image: {sample_path}")
    
    # Display image info
    height, width, channels = sample_img.shape
    print(f"Image dimensions: {width}x{height} pixels")
    print(f"Channels: {channels}")

if __name__ == "__main__":
    main()
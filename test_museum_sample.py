#!/usr/bin/env python3
"""
Test script for DepthForge with OpenVINO integration
"""

import cv2
import numpy as np
from pathlib import Path
import os

def create_museum_style_image():
    """Create a sample museum-style image for testing"""
    # Create an image with different depths - simulating a museum painting
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create a gradient background (representing distant background)
    for i in range(640):
        intensity = int(255 * i / 640)
        img[:, i] = [intensity//2, intensity//3, intensity//4]
    
    # Add some objects at different depths
    # Foreground object (closer)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle - close
    cv2.putText(img, "Foreground", (110, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Midground object (medium distance)
    cv2.circle(img, (400, 300), 50, (0, 255, 0), -1)  # Green circle - medium
    cv2.putText(img, "Midground", (350, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Background object (farther)
    cv2.rectangle(img, (500, 350), (580, 430), (0, 0, 255), -1)  # Red rectangle - far
    cv2.putText(img, "Background", (510, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add some texture to make it look more like a painting
    for i in range(0, 640, 10):
        for j in range(0, 480, 10):
            if np.random.random() > 0.7:
                cv2.circle(img, (i, j), 1, (200, 200, 200), -1)
    
    return img

def main():
    # Create sample museum-style image
    sample_img = create_museum_style_image()
    
    # Save sample image
    sample_path = "data/museum_sample_input.jpg"
    cv2.imwrite(sample_path, sample_img)
    print(f"Created museum-style sample image: {sample_path}")
    
    # Display image info
    height, width, channels = sample_img.shape
    print(f"Image dimensions: {width}x{height} pixels")
    print(f"Channels: {channels}")

if __name__ == "__main__":
    main()
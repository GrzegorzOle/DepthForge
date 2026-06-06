#!/usr/bin/env python3
"""
3D Model Generator from Depth Maps
Creates 3D printable models from depth maps for tactile visualization
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_3d_model_from_depth_map(depth_map_path: str, output_path: str, 
                                 scale_factor: float = 1.0, height_factor: float = 1.0):
    """
    Create a 3D model from a depth map
    
    Args:
        depth_map_path: Path to the depth map image
        output_path: Path to save the 3D model
        scale_factor: Scale factor for the 3D model
        height_factor: Height scaling factor
    """
    logger.info(f"Creating 3D model from depth map: {depth_map_path}")
    
    # Load depth map
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError(f"Could not load depth map from {depth_map_path}")
    
    # Get dimensions
    height, width = depth_map.shape
    logger.info(f"Depth map dimensions: {width}x{height}")
    
    # Normalize depth values to 0-1 range
    normalized_depth = depth_map.astype(np.float32) / 255.0
    
    # Create 3D coordinates
    # We'll create a simple 3D mesh representation
    vertices = []
    faces = []
    
    # Create vertices (x, y, z) where z represents depth
    for y in range(height):
        for x in range(width):
            # Z coordinate is based on depth value
            z = normalized_depth[y, x] * height_factor
            
            # Store vertex position
            vertices.append([x * scale_factor, y * scale_factor, z])
    
    # Convert to numpy array for easier handling
    vertices = np.array(vertices, dtype=np.float32)
    
    # Save as simple text file (can be imported by 3D software)
    # For a more complete solution, we would export to STL or OBJ format
    
    # Create a simplified 3D representation
    # This creates a heightfield representation that can be 3D printed
    heightfield = normalized_depth * height_factor
    
    # Save as a simple text file that can be used with 3D modeling software
    with open(output_path, 'w') as f:
        f.write("# 3D Heightfield Model from Depth Map\n")
        f.write(f"# Dimensions: {width}x{height}\n")
        f.write(f"# Scale factor: {scale_factor}\n")
        f.write(f"# Height factor: {height_factor}\n")
        f.write("\n")
        f.write("Vertices:\n")
        
        # Write vertices (x, y, z)
        for y in range(height):
            for x in range(width):
                z = heightfield[y, x]
                f.write(f"{x * scale_factor:.2f} {y * scale_factor:.2f} {z:.2f}\n")
    
    logger.info(f"3D model saved to {output_path}")
    return vertices

def create_stl_model_from_depth_map(depth_map_path: str, output_path: str, 
                                  height_factor: float = 1.0):
    """
    Create a basic STL model from depth map (simplified version)
    
    Args:
        depth_map_path: Path to the depth map image
        output_path: Path to save the STL model
        height_factor: Height scaling factor
    """
    logger.info(f"Creating STL model from depth map: {depth_map_path}")
    
    # Load depth map
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError(f"Could not load depth map from {depth_map_path}")
    
    # Get dimensions
    height, width = depth_map.shape
    
    # Normalize depth values to 0-1 range
    normalized_depth = depth_map.astype(np.float32) / 255.0
    
    # Create simple 3D model (heightfield)
    # For a proper STL export, we would use a library like numpy-stl
    # Here we'll create a simple representation
    
    # Create a simple heightfield grid
    with open(output_path, 'w') as f:
        f.write("solid depth_map_model\n")
        
        # For simplicity, we'll create a basic grid of triangles
        # This is a very simplified version - in practice you'd want to use 
        # a proper 3D library like numpy-stl for real STL generation
        
        # Create a simple representation of heightfield
        for y in range(0, height-1, 2):
            for x in range(0, width-1, 2):
                # Get depth values for 4 corners
                z1 = normalized_depth[y, x] * height_factor
                z2 = normalized_depth[y, x+1] * height_factor
                z3 = normalized_depth[y+1, x] * height_factor
                z4 = normalized_depth[y+1, x+1] * height_factor
                
                # Create two triangles for this quad
                # Triangle 1
                f.write(f"  facet normal 0 0 1\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {x} {y} {z1}\n")
                f.write(f"      vertex {x+1} {y} {z2}\n")
                f.write(f"      vertex {x} {y+1} {z3}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
                
                # Triangle 2
                f.write(f"  facet normal 0 0 1\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {x+1} {y+1} {z4}\n")
                f.write(f"      vertex {x+1} {y} {z2}\n")
                f.write(f"      vertex {x} {y+1} {z3}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
        
        f.write("endsolid depth_map_model")
    
    logger.info(f"STL model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="3D Model Generator from Depth Maps")
    parser.add_argument("--input", "-i", required=True, help="Input depth map path")
    parser.add_argument("--output", "-o", required=True, help="Output 3D model path")
    parser.add_argument("--format", "-f", choices=['txt', 'stl'], default='txt', 
                       help="Output format (txt or stl)")
    parser.add_argument("--height-factor", "-h", type=float, default=1.0, 
                       help="Height scaling factor")
    parser.add_argument("--scale-factor", "-s", type=float, default=1.0, 
                       help="Scale factor for 3D model")
    
    args = parser.parse_args()
    
    if args.format == 'txt':
        create_3d_model_from_depth_map(args.input, args.output, args.scale_factor, args.height_factor)
    elif args.format == 'stl':
        create_stl_model_from_depth_map(args.input, args.output, args.height_factor)
    
    print(f"3D model generated successfully!")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")

if __name__ == "__main__":
    main()
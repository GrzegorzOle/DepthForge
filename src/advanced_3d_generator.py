#!/usr/bin/env python3
"""
Advanced 3D Model Generator from Depth Maps
Creates 3D printable models from depth maps for tactile visualization
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import logging
from typing import Tuple
from stl import mesh
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_advanced_3d_model_from_depth_map(depth_map_path: str, output_path: str, 
                                          height_factor: float = 1.0, resolution: int = 1):
    """
    Create a detailed 3D model from a depth map using numpy-stl
    
    Args:
        depth_map_path: Path to the depth map image
        output_path: Path to save the STL model
        height_factor: Height scaling factor
        resolution: Resolution factor (lower = higher resolution)
    """
    logger.info(f"Creating advanced 3D model from depth map: {depth_map_path}")
    
    # Load depth map
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError(f"Could not load depth map from {depth_map_path}")
    
    # Get dimensions
    height, width = depth_map.shape
    logger.info(f"Depth map dimensions: {width}x{height}")
    
    # Normalize depth values to 0-1 range
    normalized_depth = depth_map.astype(np.float32) / 255.0
    
    # Create vertices and triangles for STL mesh
    # We'll create a mesh that represents the heightfield
    
    # For performance reasons, reduce resolution if needed
    step = max(1, resolution)
    
    # Calculate number of vertices
    new_width = width // step
    new_height = height // step
    
    # Create vertices
    vertices = []
    vertex_map = {}  # Map to avoid duplicate vertices
    
    # Create vertices
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Get depth value
            z = normalized_depth[y, x] * height_factor
            
            # Store vertex
            vertices.append([x, y, z])
            vertex_map[(x, y)] = len(vertices) - 1
    
    # Create triangles (faces)
    faces = []
    
    # Create triangular mesh
    for y in range(0, height - step, step):
        for x in range(0, width - step, step):
            # Get indices for four corners
            idx1 = vertex_map.get((x, y))
            idx2 = vertex_map.get((x + step, y))
            idx3 = vertex_map.get((x, y + step))
            idx4 = vertex_map.get((x + step, y + step))
            
            # Check if all vertices exist
            if all(idx is not None for idx in [idx1, idx2, idx3, idx4]):
                # Create two triangles for this quad
                # Triangle 1: (idx1, idx2, idx3)
                faces.append([[idx1, idx2, idx3]])
                # Triangle 2: (idx2, idx4, idx3)
                faces.append([[idx2, idx4, idx3]])
    
    # Convert to numpy arrays
    vertices_array = np.array(vertices, dtype=np.float32)
    faces_array = np.array(faces, dtype=np.int32)
    
    # Create mesh
    if len(faces_array) > 0:
        # Create the mesh
        cube = mesh.Mesh(np.zeros(faces_array.shape[0], dtype=mesh.Mesh.dtype))
        
        for i, face in enumerate(faces_array):
            for j in range(3):
                cube.vectors[i][j] = vertices_array[face[0][j]]
        
        # Save to STL file
        cube.save(output_path)
        logger.info(f"STL model saved to {output_path}")
        logger.info(f"Model contains {len(faces_array)} triangles")
        return True
    else:
        logger.warning("No faces created for the mesh")
        return False

def create_simple_3d_model_from_depth_map(depth_map_path: str, output_path: str, 
                                        height_factor: float = 1.0):
    """
    Create a simple 3D model from depth map using text format
    
    Args:
        depth_map_path: Path to the depth map image
        output_path: Path to save the model
        height_factor: Height scaling factor
    """
    logger.info(f"Creating simple 3D model from depth map: {depth_map_path}")
    
    # Load depth map
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)
    if depth_map is None:
        raise ValueError(f"Could not load depth map from {depth_map_path}")
    
    # Get dimensions
    height, width = depth_map.shape
    
    # Normalize depth values to 0-1 range
    normalized_depth = depth_map.astype(np.float32) / 255.0
    
    # Create a simple heightfield representation
    with open(output_path, 'w') as f:
        f.write("# Simple 3D Heightfield Model from Depth Map\n")
        f.write(f"# Dimensions: {width}x{height}\n")
        f.write(f"# Height factor: {height_factor}\n")
        f.write("\n")
        f.write("# Vertices (x, y, z)\n")
        
        # Write vertices
        for y in range(height):
            for x in range(width):
                z = normalized_depth[y, x] * height_factor
                f.write(f"{x} {y} {z}\n")
        
        f.write("\n# Faces (triangles)\n")
        # Create simple triangular mesh
        for y in range(height-1):
            for x in range(width-1):
                # Two triangles for each quad
                # Triangle 1
                f.write(f"3 {y*width+x} {(y)*width+(x+1)} {(y+1)*width+x}\n")
                # Triangle 2
                f.write(f"3 {(y)*width+(x+1)} {(y+1)*width+(x+1)} {(y+1)*width+x}\n")
    
    logger.info(f"Simple 3D model saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Advanced 3D Model Generator from Depth Maps")
    parser.add_argument("--input", "-i", required=True, help="Input depth map path")
    parser.add_argument("--output", "-o", required=True, help="Output 3D model path")
    parser.add_argument("--format", "-f", choices=['stl', 'txt'], default='stl', 
                       help="Output format (stl or txt)")
    parser.add_argument("--height-scale", type=float, default=1.0, 
                       help="Height scaling factor")
    parser.add_argument("--resolution", "-r", type=int, default=1, 
                       help="Resolution factor (higher = lower resolution)")
    
    args = parser.parse_args()
    
    if args.format == 'stl':
        success = create_advanced_3d_model_from_depth_map(
            args.input, args.output, args.height_scale, args.resolution
        )
        if success:
            print(f"STL 3D model generated successfully!")
        else:
            print("Failed to generate STL 3D model")
    elif args.format == 'txt':
        create_simple_3d_model_from_depth_map(args.input, args.output, args.height_scale)
        print(f"Simple 3D model generated successfully!")
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")

if __name__ == "__main__":
    main()
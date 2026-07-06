#!/usr/bin/env python3
"""
DepthForge - Depth Map Generation for Museum Images
Specialized for 3D tactile visualization for the visually impaired
With OpenVINO integration

This project generates depth maps from museum images to create 3D tactile visualizations
for people with visual impairments. It uses OpenVINO for efficient depth estimation
and creates models suitable for 3D printing.

Author: DepthForge Team
Version: 0.1.0
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Optional
import logging

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DepthForge:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize DepthForge with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_directories()
        if _TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
        else:
            self.device = "cpu"
            logger.info("PyTorch not available – using OpenVINO/CPU pipeline only")
        
        # Initialize OpenVINO if available
        self.ov_core = None
        try:
            import openvino as ov
            self.ov_core = ov.Core()
            logger.info("OpenVINO initialized successfully")
        except ImportError:
            logger.warning("OpenVINO not available, falling back to CPU processing")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.config['directories']['src'],
            self.config['directories']['data'],
            self.config['directories']['models'],
            self.config['directories']['output']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        # Use frombuffer + imdecode to handle Unicode/non-ASCII paths on Windows
        buf = np.frombuffer(open(image_path, "rb").read(), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for depth map generation
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Store original dimensions
        self.original_height, self.original_width = image.shape[:2]
        
        # Resize if needed according to config
        if self.config['processing']['resize_input']:
            width = self.config['processing']['resize_width']
            height = self.config['processing']['resize_height']
            image = cv2.resize(image, (width, height))
        
        return image
    
    def generate_depth_map_midas(self, image: np.ndarray) -> np.ndarray:
        """
        Generate depth map using MiDaS-like approach
        
        Args:
            image: Input image
            
        Returns:
            Generated depth map
        """
        logger.info("Generating depth map using MiDaS approach...")
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Create a synthetic depth map based on image complexity and color variation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create depth map based on intensity variations
        # Brighter areas = closer objects, darker areas = farther objects
        depth_map = 255 - gray  # Invert to make bright areas represent close objects
        
        # Apply some smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        # Normalize to 0-255 range
        if depth_map.max() > 0:
            depth_map = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        logger.info("Depth map generated successfully")
        return depth_map
    
    def _load_openvino_model(self, model_xml_path: str):
        """
        Loads an OpenVINO IR model into memory and returns the compiled model.
        Results are cached – the model is loaded only once.

        Args:
            model_xml_path: Path to the OpenVINO model .xml file

        Returns:
            Compiled OpenVINO model (CompiledModel) or None on error
        """
        if not hasattr(self, '_ov_compiled_models'):
            self._ov_compiled_models = {}

        if model_xml_path in self._ov_compiled_models:
            return self._ov_compiled_models[model_xml_path]

        xml_path = Path(model_xml_path)
        if not xml_path.exists():
            logger.error(f"OpenVINO model does not exist: {xml_path}")
            return None

        try:
            device = self.config['model']['depth_estimation'].get('device', 'CPU')
            logger.info(f"Loading OpenVINO model: {xml_path} on {device}")
            compiled = self.ov_core.compile_model(str(xml_path), device)
            self._ov_compiled_models[model_xml_path] = compiled
            logger.info(f"Model loaded successfully: {xml_path.name}")
            return compiled
        except Exception as e:
            logger.error(f"Error loading OpenVINO model: {e}")
            return None

    def generate_depth_map_openvino(self, image: np.ndarray) -> np.ndarray:
        """
        Generates a depth map using an OpenVINO IR model (MiDaS or DPT).

        Priority order:
          1. Backend specified in config ('midas' or 'dpt')
          2. Second backend as fallback
          3. Fallback to synthetic MiDaS-like method

        Args:
            image: Input image (BGR, uint8)

        Returns:
            Depth map (uint8, 0-255)
        """
        logger.info("Generating depth map with OpenVINO...")

        de_cfg = self.config['model']['depth_estimation']
        backend = de_cfg.get('backend', 'midas').lower()

        # Order of models to try
        model_order = (
            [('midas', de_cfg.get('midas_model_path')),
             ('dpt',   de_cfg.get('dpt_model_path'))]
            if backend == 'midas'
            else
            [('dpt',   de_cfg.get('dpt_model_path')),
             ('midas', de_cfg.get('midas_model_path'))]
        )

        for name, model_path in model_order:
            if not model_path:
                continue
            compiled = self._load_openvino_model(model_path)
            if compiled is None:
                continue

            try:
                depth = self._run_openvino_inference(compiled, image, name)
                logger.info(f"Depth map generated by model: {name}")
                return depth
            except Exception as e:
                logger.warning(f"Inference {name} failed: {e}")

        logger.warning("All OpenVINO models unavailable – falling back to synthetic method")
        return self.generate_depth_map_midas(image)

    def _run_openvino_inference(self, compiled_model, image: np.ndarray,
                                model_name: str) -> np.ndarray:
        """
        Runs inference on a compiled OpenVINO model.
        The image is scaled to the model's input size, then the result is
        scaled back to the original resolution.

        Args:
            compiled_model: Compiled OpenVINO model
            image: Input image (BGR, uint8)
            model_name: 'midas' or 'dpt'

        Returns:
            Depth map (uint8, 0-255) at the input image resolution
        """
        input_shape = compiled_model.input(0).shape
        if len(input_shape) == 4:
            _, _, h, w = input_shape
        else:
            h, w = 256, 256

        orig_h, orig_w = image.shape[:2]

        # BGR -> RGB conversion
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(rgb, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4)

        # Normalization (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (resized.astype(np.float32) / 255.0 - mean) / std

        # HWC -> CHW -> NCHW
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

        # Inference
        result    = compiled_model([tensor])
        depth_raw = result[compiled_model.output(0)]

        # Flatten to 2D
        if depth_raw.ndim == 4:
            depth_raw = depth_raw[0, 0]
        elif depth_raw.ndim == 3:
            depth_raw = depth_raw[0]

        # Normalize to uint8
        d_min, d_max = depth_raw.min(), depth_raw.max()
        if d_max - d_min > 1e-6:
            depth_norm = ((depth_raw - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_raw, dtype=np.uint8)

        # Resize back to original image size
        depth_resized = cv2.resize(depth_norm, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        return depth_resized

    def _run_openvino_inference_f32(self, compiled_model, image: np.ndarray,
                                    model_name: str) -> np.ndarray:
        """
        Runs inference and returns a raw float32 disparity map (NOT normalized to uint8).
        Used by the scale-shift alignment fusion pipeline.

        Args:
            compiled_model: Compiled OpenVINO model
            image: Input image (BGR, uint8)
            model_name: 'midas' or 'dpt'

        Returns:
            Raw depth/disparity map (float32) at the input image resolution.
        """
        input_shape = compiled_model.input(0).shape
        if len(input_shape) == 4:
            _, _, h, w = input_shape
        else:
            h, w = 256, 256

        orig_h, orig_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (int(w), int(h)), interpolation=cv2.INTER_LANCZOS4)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (resized.astype(np.float32) / 255.0 - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

        result    = compiled_model([tensor])
        depth_raw = result[compiled_model.output(0)]

        if depth_raw.ndim == 4:
            depth_raw = depth_raw[0, 0]
        elif depth_raw.ndim == 3:
            depth_raw = depth_raw[0]

        # Return raw float32 — no quantisation, full precision for MWK alignment
        depth_f32 = depth_raw.astype(np.float32)
        depth_resized = cv2.resize(depth_f32, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LANCZOS4)
        return depth_resized

    def enhance_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Enhance the depth map for better tactile visualization
        
        Args:
            depth_map: Raw depth map
            
        Returns:
            Enhanced depth map
        """
        logger.info("Enhancing depth map for tactile visualization...")
        
        # Apply histogram equalization for better contrast
        enhanced = cv2.equalizeHist(depth_map)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better detail
        clahe = cv2.createCLAHE(clipLimit=self.config['processing']['enhancement']['clahe_clip_limit'], 
                               tileGridSize=tuple(self.config['processing']['enhancement']['clahe_tile_size']))
        enhanced = clahe.apply(enhanced)
        
        # Apply morphological operations to smooth the depth map
        kernel = np.ones(tuple(self.config['processing']['enhancement']['morphology_kernel_size']), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        logger.info("Depth map enhanced successfully")
        return enhanced
    
    def resize_depth_map(self, depth_map: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Resize depth map to match original image dimensions
        
        Args:
            depth_map: Depth map to resize
            target_width: Target width
            target_height: Target height
            
        Returns:
            Resized depth map
        """
        logger.info(f"Resizing depth map from {depth_map.shape[1]}x{depth_map.shape[0]} to {target_width}x{target_height}")
        
        resized = cv2.resize(depth_map, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def save_depth_map(self, depth_map: np.ndarray, output_path: str):
        """
        Save depth map to file
        
        Args:
            depth_map: Depth map to save
            output_path: Path to save the depth map
        """
        cv2.imwrite(output_path, depth_map)
        logger.info(f"Depth map saved to {output_path}")
    
    def save_tactile_visualization(self, depth_map: np.ndarray, output_path: str):
        """
        Save tactile visualization suitable for 3D printing
        
        Args:
            depth_map: Depth map to convert
            output_path: Path to save the tactile visualization
        """
        # Convert to a format suitable for tactile visualization
        # This creates a grayscale representation that can be used for 3D printing
        tactile_map = depth_map.copy()
        
        # Ensure it's in the right format for 3D printing
        # Values from 0-255 where 0 = deepest, 255 = highest
        
        # Save as PNG for maximum compatibility
        cv2.imwrite(output_path, tactile_map)
        logger.info(f"Tactile visualization saved to {output_path}")
    
    def process_single_image(self, input_path: str, output_path: Optional[str] = None,
                           enhanced_output: Optional[str] = None, tactile_output: Optional[str] = None):
        """
        Process a single image and generate depth map
        
        Args:
            input_path: Path to input image
            output_path: Path to save output depth map (optional)
            enhanced_output: Path to save enhanced depth map (optional)
            tactile_output: Path to save tactile visualization (optional)
        """
        logger.info(f"Processing image: {input_path}")
        
        # Load and preprocess image
        image = self.load_image(input_path)
        processed_image = self.preprocess_image(image)
        
        # Generate depth map using OpenVINO or fallback method
        if self.ov_core is not None and self.config['model']['depth_estimation']['use_openvino']:
            depth_map = self.generate_depth_map_openvino(processed_image)
        else:
            depth_map = self.generate_depth_map_midas(processed_image)
        
        # Enhance the depth map for better tactile visualization
        enhanced_depth_map = self.enhance_depth_map(depth_map)
        
        # Resize depth maps back to original dimensions if needed
        if self.config['processing']['resize_input']:
            depth_map = self.resize_depth_map(depth_map, self.original_width, self.original_height)
            enhanced_depth_map = self.resize_depth_map(enhanced_depth_map, self.original_width, self.original_height)
        
        # Determine output paths
        if output_path is None:
            input_file = Path(input_path)
            output_path = f"{self.config['directories']['output']}/{input_file.stem}_depth.{self.config['processing']['output_format']}"
        
        if enhanced_output is None:
            input_file = Path(input_path)
            enhanced_output = f"{self.config['directories']['output']}/{input_file.stem}_enhanced.{self.config['processing']['output_format']}"
        
        if tactile_output is None:
            input_file = Path(input_path)
            tactile_output = f"{self.config['directories']['output']}/{input_file.stem}_tactile.{self.config['processing']['output_format']}"
        
        # Save depth maps
        self.save_depth_map(depth_map, output_path)
        self.save_depth_map(enhanced_depth_map, enhanced_output)
        self.save_tactile_visualization(enhanced_depth_map, tactile_output)
        
        return depth_map, enhanced_depth_map
    
    def process_batch(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Process batch of images
        
        Args:
            input_dir: Directory with input images
            output_dir: Directory for output depth maps
        """
        if input_dir is None:
            input_dir = self.config['directories']['data']
        if output_dir is None:
            output_dir = self.config['directories']['output']
            
        # Supported image formats
        supported_formats = self.config['processing']['input_format'].split(',')
        
        # Find all image files
        image_files = []
        for fmt in supported_formats:
            image_files.extend(Path(input_dir).glob(f"*{fmt}"))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_file in image_files:
            output_file = f"{output_dir}/{image_file.stem}_depth.{self.config['processing']['output_format']}"
            enhanced_file = f"{output_dir}/{image_file.stem}_enhanced.{self.config['processing']['output_format']}"
            tactile_file = f"{output_dir}/{image_file.stem}_tactile.{self.config['processing']['output_format']}"
            
            self.process_single_image(str(image_file), output_file, enhanced_file, tactile_file)

def main():
    parser = argparse.ArgumentParser(description="DepthForge - Depth Map Generation for Museum Images")
    parser.add_argument("--input", "-i", help="Input image path")
    parser.add_argument("--output", "-o", help="Output depth map path")
    parser.add_argument("--enhanced-output", help="Output enhanced depth map path")
    parser.add_argument("--tactile-output", help="Output tactile visualization path")
    parser.add_argument("--batch", "-b", action="store_true", help="Process batch of images")
    parser.add_argument("--input-dir", help="Input directory for batch processing")
    parser.add_argument("--output-dir", help="Output directory for batch processing")
    
    args = parser.parse_args()
    
    # Initialize DepthForge
    depth_forge = DepthForge()
    
    if args.batch:
        depth_forge.process_batch(args.input_dir, args.output_dir)
    elif args.input:
        depth_forge.process_single_image(args.input, args.output, args.enhanced_output, args.tactile_output)
    else:
        print("Please specify either --input or --batch flag")
        return

if __name__ == "__main__":
    main()
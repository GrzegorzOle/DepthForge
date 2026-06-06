# DepthForge - GIMP Plugin Integration

## DepthForge GIMP Plugin

The DepthForge project can be extended with a GIMP plugin, enabling depth map generation directly within the graphical environment.

## Requirements:

- GIMP 2.10+ with Python-Fu support
- Python 3.8+
- All DepthForge project libraries

## Plugin installation:

1. **Copy the plugin file:**
   ```bash
   cp src/gimp_plugin.py /path/to/gimp/plug-ins/
   ```

2. **Grant permissions:**
   ```bash
   chmod +x /path/to/gimp/plug-ins/gimp_plugin.py
   ```

3. **Start GIMP and reload plugins:**
   - `Filters` → `DepthForge` → `Depth Map Generator`

## Usage:

### Command line:
```bash
python src/gimp_plugin.py --input image.jpg --output depth.png --enhancement 75
```

### Creating the plugin:
```bash
python src/gimp_plugin.py --create-plugin --plugin-dir gimp_plugins
```

## Plugin features:

1. **Depth map generation** from images
2. **Applying filters** and corrections
3. **GIMP system integration**
4. **Support for various image formats**

## Example usage in GIMP:

1. Open an image in GIMP
2. Select `Filters` → `DepthForge` → `Depth Map Generator`
3. Enter settings (enhancement level, size)
4. Generate the depth map
5. Use the map for further editing or export

## Required libraries:

```bash
pip install gimp
```

## Available options:

- `--input` / `-i`: Path to the input image
- `--output` / `-o`: Path to the output file
- `--enhancement` / `-e`: Enhancement level (0-100)
- `--create-plugin`: Creates plugin files for GIMP
- `--plugin-dir`: Directory for plugin files

## Version for GIMP 3.x:

The plugin has been designed with GIMP 3.x compatibility in mind, but may require minor modifications depending on the specific version.

## GIMP integration:

The plugin integrates with GIMP via:
- Python-Fu support
- Reading and writing images in PNG format
- Layer and channel handling
- Integration with GIMP's filter menu

## Testing:

```bash
# Test with a sample image
python src/gimp_plugin.py --input data/sample_input.jpg --output output/gimp_test.png --enhancement 60
```

## Example usage in GIMP:

1. Open an image in GIMP
2. Select `Filters` → `DepthForge` → `Depth Map Generator`
3. Set the enhancement level (e.g. 75)
4. Click "OK"
5. You will receive the depth map as a new layer

## Technical support:

If you encounter problems with the plugin installation:
1. Make sure GIMP has Python-Fu support enabled
2. Verify that all required libraries are installed
3. Check the plugin file paths
4. Run GIMP with administrator privileges (if required)

## Future development:

The plugin can be extended in the future with:
- Support for additional image formats
- Extra filters and effects
- Integration with 3D printing systems
- Simultaneous processing of multiple layers

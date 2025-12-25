# K-means Color Quantizer

High-quality image color reduction using K-means clustering algorithm. Produces superior results compared to standard palette reduction methods.

![Example Comparison](example.png) | ![K-means Result](example_kmeans.png)
:---:|:---:
Original | After K-means (256 colors)

## Features

- **Smart palette generation** using K-means clustering for optimal color selection
- **Memory-efficient** batched processing for large images
- **Optional Floyd-Steinberg dithering** for smoother gradients
- **Transparency support** with multiple modes:
  - Replace transparency with custom color (magenta/pink/green/etc.) at palette index 255
  - Export alpha channel as separate grayscale mask
  - Perfect for retro game sprites (BennuGD, Streets of Rage Remake, Genesis/Mega Drive)
- **Flexible output**: PNG or BMP format
- **Palette export** to Adobe Color Table (.ACT) format
- **Configurable color count** (2-256 colors)

## Installation

### Requirements

- Python 3.6 or newer
- NumPy
- Pillow (PIL)

### Install dependencies

```bash
pip install numpy pillow
```

## Usage

### Basic Examples

**Convert to 256 colors:**
```bash
python kmeans_8bit.py input.png
```

**Convert to 16 colors:**
```bash
python kmeans_8bit.py input.png -c 16
```

**Apply dithering:**
```bash
python kmeans_8bit.py input.png -d
```

**Save as BMP:**
```bash
python kmeans_8bit.py input.png -f BMP -o output.bmp
```

### Transparency Handling

**Replace transparency with magenta (for game sprites):**
```bash
python kmeans_8bit.py sprite.png --trans-color magenta
```

**Genesis/Mega Drive sprites (15 colors + transparency):**
```bash
python kmeans_8bit.py character.png -c 15 --trans-color magenta -f BMP
```

**Save alpha channel as separate mask:**
```bash
python kmeans_8bit.py sprite.png --save-alpha
```
This creates:
- `sprite_8bit.png` - with magenta replacing transparency
- `sprite_8bit_alpha.png` - grayscale alpha mask

**Custom transparency color:**
```bash
python kmeans_8bit.py sprite.png --trans-color pink
python kmeans_8bit.py sprite.png --trans-color 0,255,0    # Custom RGB green
```

Available presets: `magenta`, `pink`, `green`, `cyan`, `black`

### Advanced Options

**Export palette:**
```bash
python kmeans_8bit.py input.png --save-palette
```
Creates an Adobe Color Table (.ACT) file compatible with Photoshop and other tools.

**Adjust transparency threshold:**
```bash
python kmeans_8bit.py sprite.png --trans-color magenta --alpha-threshold 64
```
Pixels with alpha < 64 are considered transparent (default: 128).

**Memory optimization for large images:**
```bash
python kmeans_8bit.py large.png -b 2000
```
Uses smaller batches (lower RAM usage).

**Control K-means iterations:**
```bash
python kmeans_8bit.py input.png -i 50
```

## Command-Line Options

```
positional arguments:
  input                 Input PNG/BMP/JPG image path

options:
  -h, --help            Show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output image path (auto-generated if omitted)
  -c COLORS, --colors COLORS
                        Number of colors (2-256, default: 256)
  -d, --dither          Apply Floyd-Steinberg dithering
  -f {PNG,BMP}, --format {PNG,BMP}
                        Output format (default: PNG)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Pixels per batch (default: 5000)
  --save-alpha          Save alpha mask file (auto-enables --trans-color magenta)
  --trans-color COLOR   Replace transparency with color at index 255
                        (magenta/pink/green/cyan/black or R,G,B)
  --alpha-threshold N   Alpha threshold for transparency (0-255, default: 128)
  --save-palette        Save palette as .ACT file
  -i ITERATIONS, --iterations ITERATIONS
                        K-means iterations (default: 30)
```

## Use Cases

### Retro Game Development

**BennuGD / Streets of Rage Remake sprites:**
```bash
python kmeans_8bit.py character.png --trans-color magenta -f BMP
```

**Sega Genesis/Mega Drive sprites (15 colors + transparency):**
```bash
python kmeans_8bit.py sprite.png -c 15 --trans-color magenta -f BMP
```

**Export with alpha mask for editing:**
```bash
python kmeans_8bit.py sprite.png -c 15 --save-alpha -f BMP
```
Creates both the sprite BMP and a separate alpha mask BMP.

### Photo Conversion

**High-quality 256-color image:**
```bash
python kmeans_8bit.py photo.jpg -d
```

**Poster-style with fewer colors:**
```bash
python kmeans_8bit.py photo.jpg -c 32 -d
```

**Export palette for reuse:**
```bash
python kmeans_8bit.py photo.jpg --save-palette
```

## How It Works

1. **Load image** and separate alpha channel if present
2. **K-means clustering** analyzes pixel colors to find optimal palette
   - When using transparency replacement, transparent pixels are excluded from analysis
   - This ensures the palette is optimized for visible colors only
3. **Map pixels** to nearest palette colors
4. **Optional dithering** for smoother color transitions
5. **Save output** as indexed color image (PNG/BMP)
6. **Optional alpha export** as separate grayscale mask

## Technical Details

- **Algorithm**: Mini-batch K-means clustering
- **Color space**: RGB
- **Palette size**: 256 colors maximum (8-bit indexed)
- **Dithering**: Floyd-Steinberg error diffusion
- **Transparency**: Stored at palette index 255 when using `--trans-color`
- **Memory usage**: Configurable via batch size parameter

## Comparison with Other Methods

| Method | Quality | Speed | Notes |
|--------|---------|-------|-------|
| K-means (this tool) | ★★★★★ | ★★★☆☆ | Best quality, optimized per image |
| Median Cut | ★★★☆☆ | ★★★★☆ | Fast, decent quality |
| Octree | ★★★☆☆ | ★★★★★ | Very fast, good for photos |
| Fixed Palette | ★★☆☆☆ | ★★★★★ | Fast but limited |

K-means produces **superior quality** because it analyzes the actual color distribution in your image to generate an optimal palette, rather than using generic algorithms.

## Known Limitations

- RGB color space only (no CMYK support)
- Single-threaded processing
- K-means convergence can vary with random initialization

## License

MIT License - See LICENSE file for details

## Credits

Developed for high-quality palette reduction with support for retro game development workflows.

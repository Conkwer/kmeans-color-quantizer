#!/usr/bin/env python3
"""
PNG/BMP 24-bit to 8-bit Converter using K-means Clustering
Memory-optimized version for large images
"""

import numpy as np
from PIL import Image
import argparse
import os
import sys
from pathlib import Path

def find_nearest_color_batched(pixels, palette, batch_size=5000):
    """Find nearest palette color for each pixel using batched processing"""
    n_pixels = len(pixels)
    labels = np.zeros(n_pixels, dtype=np.uint8)
    
    for i in range(0, n_pixels, batch_size):
        end_idx = min(i + batch_size, n_pixels)
        batch = pixels[i:end_idx]
        
        # Calculate distances for this batch only
        distances = np.sum((batch[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2, axis=2)
        labels[i:end_idx] = np.argmin(distances, axis=1)
    
    return labels


def kmeans_color_quantization(image_array, n_colors=256, max_iter=30, batch_size=5000):
    """
    Memory-efficient K-means clustering for optimal color palette generation
    """
    h, w, c = image_array.shape
    has_alpha = (c == 4)
    
    if has_alpha:
        alpha_channel = image_array[:, :, 3]
        rgb_data = image_array[:, :, :3]
    else:
        rgb_data = image_array
        alpha_channel = None
    
    pixels = rgb_data.reshape(-1, 3).astype(np.float32)
    n_pixels = len(pixels)
    
    print(f"Processing {h}x{w} image ({n_pixels:,} pixels)...")
    
    # Simple random initialization (faster and more memory efficient)
    print(f"Initializing {n_colors} color palette...")
    indices = np.random.choice(n_pixels, min(n_colors, n_pixels), replace=False)
    centroids = pixels[indices].copy().astype(np.float32)
    
    # Pad if needed
    if len(centroids) < n_colors:
        padding = np.tile(centroids, (n_colors // len(centroids) + 1, 1))[:n_colors - len(centroids)]
        centroids = np.vstack([centroids, padding])
    
    # K-means iteration with batched processing
    print("Running K-means clustering...")
    for iteration in range(max_iter):
        # Assign labels in batches
        labels = find_nearest_color_batched(pixels, centroids, batch_size)
        
        # Update centroids
        new_centroids = np.zeros((n_colors, 3), dtype=np.float32)
        
        for i in range(n_colors):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = pixels[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]
        
        # Check convergence
        centroid_shift = np.max(np.abs(centroids - new_centroids))
        centroids = new_centroids
        
        if (iteration + 1) % 5 == 0:
            print(f"  Iteration {iteration + 1}/{max_iter}, max shift: {centroid_shift:.2f}")
        
        if centroid_shift < 0.5:
            print(f"  Converged after {iteration + 1} iterations")
            break
    
    # Final assignment
    print("Generating final indexed image...")
    labels = find_nearest_color_batched(pixels, centroids, batch_size)
    
    palette = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
    indexed_image = labels.reshape(h, w)
    
    return indexed_image, palette, alpha_channel


def floyd_steinberg_dither(image_array, palette):
    """
    Apply Floyd-Steinberg error diffusion dithering
    """
    h, w, c = image_array.shape
    img_float = image_array[:, :, :3].astype(np.float32)
    result = np.zeros((h, w), dtype=np.uint8)
    
    print("Applying Floyd-Steinberg dithering...")
    
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x]
            
            # Find nearest palette color
            distances = np.sum((palette - old_pixel) ** 2, axis=1)
            idx = np.argmin(distances)
            new_pixel = palette[idx].astype(np.float32)
            
            result[y, x] = idx
            error = old_pixel - new_pixel
            
            # Distribute error to neighbors (Floyd-Steinberg weights)
            if x + 1 < w:
                img_float[y, x + 1] += error * 7/16
            if y + 1 < h:
                if x > 0:
                    img_float[y + 1, x - 1] += error * 3/16
                img_float[y + 1, x] += error * 5/16
                if x + 1 < w:
                    img_float[y + 1, x + 1] += error * 1/16
        
        if (y + 1) % 50 == 0 or y == h - 1:
            print(f"  Row {y + 1}/{h} ({100*(y+1)/h:.1f}%)")
    
    return result


def convert_image(input_path, output_path=None, n_colors=256, dither=False, 
                 output_format='PNG', batch_size=5000):
    """
    Convert 24-bit image to 8-bit indexed color
    """
    if n_colors < 2 or n_colors > 256:
        raise ValueError("n_colors must be between 2 and 256")
    
    print(f"Loading: {input_path}")
    img = Image.open(input_path)
    
    # Convert to RGB
    if img.mode == 'P':
        img = img.convert('RGB')
    elif img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    
    img_array = np.array(img)
    
    # Perform K-means quantization
    indexed_img, palette, alpha = kmeans_color_quantization(
        img_array, n_colors=n_colors, batch_size=batch_size
    )
    
    # Apply dithering if requested
    if dither:
        indexed_img = floyd_steinberg_dither(img_array, palette)
    
    # Create PIL Image with palette
    result = Image.fromarray(indexed_img, mode='P')
    
    # Set palette
    palette_flat = palette.flatten().tolist()
    palette_flat += [0] * (768 - len(palette_flat))
    result.putpalette(palette_flat)
    
    # Generate output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.with_name(input_path_obj.stem + f'_8bit.{output_format.lower()}')
    
    # Save
    result.save(output_path, format=output_format)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Colors used: {len(np.unique(indexed_img))}/{n_colors}")
    
    return result, palette


def save_palette_act(palette, output_path):
    """Save palette as Adobe Color Table (.ACT) format"""
    act_data = np.zeros(768, dtype=np.uint8)
    n_colors = len(palette)
    
    for i in range(n_colors):
        act_data[i*3:(i+1)*3] = palette[i]
    
    with open(output_path, 'wb') as f:
        f.write(act_data.tobytes())
    
    print(f"✓ Saved palette: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert 24-bit PNG/BMP to 8-bit indexed color using K-means',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.png                          # Convert to 256 colors
  %(prog)s photo.png -c 128                   # Use 128 colors
  %(prog)s photo.png -d                       # Apply dithering
  %(prog)s photo.png -o output.bmp -f BMP     # Save as BMP
  %(prog)s photo.png --save-palette           # Also save .ACT palette
  %(prog)s photo.png -b 2000                  # Use smaller batches (less RAM)
        """
    )
    
    parser.add_argument('input', help='Input PNG/BMP/JPG image path')
    parser.add_argument('-o', '--output', help='Output image path (auto-generated if omitted)')
    parser.add_argument('-c', '--colors', type=int, default=256, 
                        help='Number of colors (2-256, default: 256)')
    parser.add_argument('-d', '--dither', action='store_true',
                        help='Apply Floyd-Steinberg dithering')
    parser.add_argument('-f', '--format', choices=['PNG', 'BMP'], default='PNG',
                        help='Output format (default: PNG)')
    parser.add_argument('-b', '--batch-size', type=int, default=5000,
                        help='Pixels per batch - lower uses less RAM (default: 5000)')
    parser.add_argument('--save-palette', action='store_true',
                        help='Save palette as .ACT file')
    parser.add_argument('-i', '--iterations', type=int, default=30,
                        help='K-means iterations (default: 30)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        result, palette = convert_image(
            args.input, 
            args.output, 
            n_colors=args.colors,
            dither=args.dither,
            output_format=args.format,
            batch_size=args.batch_size
        )
        
        if args.save_palette:
            if args.output:
                palette_path = Path(args.output).with_suffix('.act')
            else:
                palette_path = Path(args.input).with_name(Path(args.input).stem + '_8bit.act')
            save_palette_act(palette, palette_path)
        
        print(f"\n✓ Conversion complete!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

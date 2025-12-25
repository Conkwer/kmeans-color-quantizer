#!/usr/bin/env python3
"""
PNG/BMP 24-bit to 8-bit Converter using K-means Clustering
Memory-optimized version with alpha mask export and transparency color replacement
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


def kmeans_color_quantization(image_array, n_colors=256, max_iter=30, batch_size=5000, exclude_transparent=False, alpha_threshold=128):
    """
    Memory-efficient K-means clustering for optimal color palette generation
    Separates alpha channel if present

    Args:
        exclude_transparent: if True, don't include transparent pixels in K-means
    """
    h, w, c = image_array.shape
    has_alpha = (c == 4)

    if has_alpha:
        alpha_channel = image_array[:, :, 3].copy()
        rgb_data = image_array[:, :, :3]
        print(f"  Detected alpha channel")
    else:
        rgb_data = image_array
        alpha_channel = None

    pixels = rgb_data.reshape(-1, 3).astype(np.float32)
    n_pixels = len(pixels)

    # If excluding transparent pixels, filter them out for K-means
    if exclude_transparent and has_alpha:
        opaque_mask = alpha_channel.flatten() >= alpha_threshold
        pixels_for_kmeans = pixels[opaque_mask]
        print(f"Processing {h}x{w} image ({len(pixels_for_kmeans):,} opaque pixels for palette)...")
    else:
        pixels_for_kmeans = pixels
        print(f"Processing {h}x{w} image ({n_pixels:,} pixels)...")

    # Simple random initialization
    print(f"Initializing {n_colors} color palette...")
    n_samples = len(pixels_for_kmeans)
    indices = np.random.choice(n_samples, min(n_colors, n_samples), replace=False)
    centroids = pixels_for_kmeans[indices].copy().astype(np.float32)

    # Pad if needed
    if len(centroids) < n_colors:
        padding = np.tile(centroids, (n_colors // len(centroids) + 1, 1))[:n_colors - len(centroids)]
        centroids = np.vstack([centroids, padding])

    # K-means iteration
    print("Running K-means clustering...")
    for iteration in range(max_iter):
        # Assign labels in batches
        labels = find_nearest_color_batched(pixels_for_kmeans, centroids, batch_size)

        # Update centroids
        new_centroids = np.zeros((n_colors, 3), dtype=np.float32)

        for i in range(n_colors):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = pixels_for_kmeans[mask].mean(axis=0)
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

    # Final assignment for ALL pixels (including transparent ones)
    print("Generating final indexed image...")
    labels = find_nearest_color_batched(pixels, centroids, batch_size)

    palette = np.clip(np.round(centroids), 0, 255).astype(np.uint8)
    indexed_image = labels.reshape(h, w)

    return indexed_image, palette, alpha_channel


def floyd_steinberg_dither(image_array, palette):
    """Apply Floyd-Steinberg error diffusion dithering"""
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

            # Distribute error to neighbors
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


def save_alpha_mask(alpha_channel, output_path, format='BMP'):
    """Save alpha channel as separate grayscale image"""
    alpha_img = Image.fromarray(alpha_channel, mode='L')
    alpha_img.save(output_path, format=format)
    print(f"  Alpha mask: {output_path}")


def replace_transparent_with_color(indexed_img, palette, alpha_channel, trans_color, alpha_threshold=128):
    """
    Replace transparent pixels with a specific color at palette index 255
    Expands palette to 256 colors if needed
    """
    h, w = indexed_img.shape

    # Expand palette to 256 colors if it's smaller
    current_size = len(palette)
    if current_size < 256:
        # Create new 256-color palette
        new_palette = np.zeros((256, 3), dtype=np.uint8)
        new_palette[:current_size] = palette
        palette = new_palette
        print(f"  Expanded palette from {current_size} to 256 colors")

    # Use last palette index for transparency color
    trans_idx = 255
    palette[trans_idx] = np.array(trans_color, dtype=np.uint8)
    print(f"  Transparency color RGB{trans_color} at palette index {trans_idx}")

    # Replace transparent pixels
    transparent_mask = alpha_channel < alpha_threshold
    indexed_img[transparent_mask] = trans_idx

    num_transparent = np.sum(transparent_mask)
    print(f"  Replaced {num_transparent:,} transparent pixels")

    return indexed_img, palette


def convert_image(input_path, output_path=None, n_colors=256, dither=False, 
                 output_format='PNG', batch_size=5000, save_alpha=False, 
                 transparency_color=None, alpha_threshold=128):
    """
    Convert 24-bit image to 8-bit indexed color
    """
    if n_colors < 2 or n_colors > 256:
        raise ValueError("n_colors must be between 2 and 256")

    # If save_alpha is True, automatically enable transparency color (magenta)
    if save_alpha and transparency_color is None:
        transparency_color = (255, 0, 255)  # Magenta
        print("--save-alpha enabled: using magenta (255, 0, 255) for transparency")

    print(f"Loading: {input_path}")
    img = Image.open(input_path)

    # Convert to RGB or RGBA
    if img.mode == 'P':
        if 'transparency' in img.info:
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
    elif img.mode in ('RGBA', 'LA'):
        if img.mode == 'LA':
            img = img.convert('RGBA')
    else:
        img = img.convert('RGB')

    img_array = np.array(img)

    # Perform K-means quantization
    # Exclude transparent pixels from palette generation if using transparency color
    indexed_img, palette, alpha = kmeans_color_quantization(
        img_array, 
        n_colors=n_colors, 
        batch_size=batch_size,
        exclude_transparent=(transparency_color is not None),
        alpha_threshold=alpha_threshold
    )

    # Apply dithering if requested
    if dither:
        indexed_img = floyd_steinberg_dither(img_array, palette)

    # Replace transparent pixels with color if specified
    if transparency_color and alpha is not None:
        indexed_img, palette = replace_transparent_with_color(
            indexed_img, palette, alpha, transparency_color, alpha_threshold
        )

    # Generate output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = input_path_obj.with_name(input_path_obj.stem + f'_8bit.{output_format.lower()}')

    output_path = Path(output_path)

    # Save indexed color image
    result = Image.fromarray(indexed_img, mode='P')

    # Set palette
    palette_flat = palette.flatten().tolist()
    palette_flat += [0] * (768 - len(palette_flat))
    result.putpalette(palette_flat)

    result.save(output_path, format=output_format)

    print(f"\n✓ Saved: {output_path}")
    print(f"  Colors used: {len(np.unique(indexed_img))}/{len(palette)}")

    # Save alpha mask if requested
    if save_alpha and alpha is not None:
        alpha_path = output_path.with_name(output_path.stem + '_alpha' + output_path.suffix)
        save_alpha_mask(alpha, alpha_path, format=output_format)
    elif save_alpha and alpha is None:
        print(f"  Note: No alpha channel found in input image")

    return result, palette


def save_palette_act(palette, output_path):
    """Save palette as Adobe Color Table (.ACT) format"""
    act_data = np.zeros(768, dtype=np.uint8)
    n_colors = len(palette)

    for i in range(min(n_colors, 256)):
        act_data[i*3:(i+1)*3] = palette[i]

    with open(output_path, 'wb') as f:
        f.write(act_data.tobytes())

    print(f"✓ Saved palette: {output_path}")


def parse_color(color_str):
    """Parse color from string format"""
    color_presets = {
        'pink': (255, 192, 203),
        'magenta': (255, 0, 255),
        'green': (0, 255, 0),
        'cyan': (0, 255, 255),
        'black': (0, 0, 0),
    }

    if color_str.lower() in color_presets:
        return color_presets[color_str.lower()]
    else:
        try:
            rgb = tuple(int(x.strip()) for x in color_str.split(','))
            if len(rgb) != 3 or not all(0 <= c <= 255 for c in rgb):
                raise ValueError
            return rgb
        except:
            raise ValueError(f"Invalid color: {color_str}. Use preset (pink/magenta/green/cyan/black) or R,G,B format")


def main():
    parser = argparse.ArgumentParser(
        description='Convert 24-bit PNG/BMP to 8-bit indexed color using K-means',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sprite.png                              # Convert to 256 colors
  %(prog)s sprite.png -c 15 --trans-color magenta  # 15 colors + magenta for transparency
  %(prog)s sprite.png -c 15 --save-alpha           # 15 colors + magenta + alpha mask
  %(prog)s sprite.png --trans-color magenta        # 256 colors with magenta transparency
  %(prog)s sprite.png -f BMP --trans-color magenta # BennuGD/SoR format
  %(prog)s photo.png -c 128 -d                     # 128 colors with dithering
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
                        help='Pixels per batch (default: 5000)')
    parser.add_argument('--save-alpha', action='store_true',
                        help='Save alpha mask file (auto-enables --trans-color magenta)')
    parser.add_argument('--trans-color', type=str, metavar='COLOR',
                        help='Replace transparency with color at index 255 (magenta/pink/green/cyan/black or R,G,B)')
    parser.add_argument('--alpha-threshold', type=int, default=128, metavar='N',
                        help='Alpha threshold for transparency (0-255, default: 128)')
    parser.add_argument('--save-palette', action='store_true',
                        help='Save palette as .ACT file')
    parser.add_argument('-i', '--iterations', type=int, default=30,
                        help='K-means iterations (default: 30)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Parse transparency color
    trans_color = None
    if args.trans_color:
        try:
            trans_color = parse_color(args.trans_color)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    try:
        result, palette = convert_image(
            args.input, 
            args.output, 
            n_colors=args.colors,
            dither=args.dither,
            output_format=args.format,
            batch_size=args.batch_size,
            save_alpha=args.save_alpha,
            transparency_color=trans_color,
            alpha_threshold=args.alpha_threshold
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

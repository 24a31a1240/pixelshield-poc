#!/usr/bin/env python3
"""
Step 5: Generate side-by-side comparison image.

Takes original and cloaked images, arranges them side-by-side with labels,
and saves as a single visual comparison image.
"""

import argparse
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_comparison_image(original_path, cloaked_path, output_path, label_original="Original", label_cloaked="Cloaked"):
    """
    Create a side-by-side comparison image.
    
    Args:
        original_path: Path to original image
        cloaked_path: Path to cloaked image
        output_path: Path to save comparison image
        label_original: Label for original image
        label_cloaked: Label for cloaked image
    """
    print(f"\n{'='*70}")
    print(f"Generating Comparison Image")
    print(f"{'='*70}")
    
    # Load images
    print(f"[*] Loading images...")
    orig_img = Image.open(original_path).convert('RGB')
    cloak_img = Image.open(cloaked_path).convert('RGB')
    
    print(f"[+] Original: {orig_img.size}")
    print(f"[+] Cloaked:  {cloak_img.size}")
    
    # Ensure same size
    if orig_img.size != cloak_img.size:
        print(f"[!] Resizing images to match...")
        size = orig_img.size
        cloak_img = cloak_img.resize(size, Image.Resampling.LANCZOS)
    
    # Create comparison: side by side
    width = orig_img.width + cloak_img.width + 60  # 20px padding on each side, 20px in middle
    height = max(orig_img.height, cloak_img.height) + 100  # Extra space for labels
    
    comparison = Image.new('RGB', (width, height), color=(255, 255, 255))
    
    # Paste images
    comparison.paste(orig_img, (20, 60))
    comparison.paste(cloak_img, (orig_img.width + 40, 60))
    
    # Add labels
    draw = ImageDraw.Draw(comparison)
    
    # Try to use a nice font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    # Original label
    draw.text((20, 15), label_original, fill=(0, 0, 0), font=font)
    
    # Cloaked label
    draw.text((orig_img.width + 40, 15), label_cloaked, fill=(0, 0, 0), font=font)
    
    # Bottom caption
    caption = "Visual difference should be imperceptible to human eye"
    draw.text((20, height - 30), caption, fill=(100, 100, 100), font=small_font)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comparison.save(output_path)
    print(f"\n[+] Comparison image saved: {output_path}")
    print(f"[+] Size: {comparison.size}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side comparison of original and cloaked images"
    )
    parser.add_argument('original', help='Path to original image')
    parser.add_argument('cloaked', help='Path to cloaked image')
    parser.add_argument('-o', '--output', default=None,
                       help='Output path (default: output/comparison/comparison.png)')
    parser.add_argument('--label-original', default='Original Face',
                       help='Label for original image (default: "Original Face")')
    parser.add_argument('--label-cloaked', default='Adversarially Cloaked',
                       help='Label for cloaked image (default: "Adversarially Cloaked")')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = 'output/comparison/comparison.png'
    
    create_comparison_image(
        args.original,
        args.cloaked,
        args.output,
        label_original=args.label_original,
        label_cloaked=args.label_cloaked
    )


if __name__ == '__main__':
    main()

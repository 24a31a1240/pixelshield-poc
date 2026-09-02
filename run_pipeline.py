#!/usr/bin/env python3
"""
Pipeline runner: Execute the complete PixelShield workflow end-to-end.

This script automates Steps 3-5 in a single command:
1. Load image → cloak it (Step 3)
2. Test face recognition (Step 4)
3. Generate comparison image (Step 5)

Usage:
  python run_pipeline.py images/input/my_face.jpg [--reference images/input/reference_face.jpg]
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cloak import ImageCloaker
from attack_test import FaceRecognitionTester
from compare import create_comparison_image
import torch

def run_full_pipeline(input_path, reference_path=None, epsilon=8/255, alpha=2/255,
                      num_steps=40, lpips_threshold=0.05, device=None):
    """
    Execute complete cloaking pipeline:
    1. Cloak the image
    2. Test face recognition
    3. Generate visual comparison
    """
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"PIXELSHIELD PHASE 1: COMPLETE PIPELINE")
    print(f"{'='*70}")
    print(f"Device: {device.upper()}")
    print(f"Input: {input_path}")
    if reference_path:
        print(f"Reference: {reference_path}")
    print(f"{'='*70}\n")
    
    # Verify input exists
    if not os.path.exists(input_path):
        print(f"[!] Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if reference_path and not os.path.exists(reference_path):
        print(f"[!] Error: Reference file not found: {reference_path}")
        sys.exit(1)
    
    # Step 1: CLOAK
    print(f"\n{'*'*70}")
    print(f"STEP 1: ADVERSARIAL CLOAKING")
    print(f"{'*'*70}")
    
    input_stem = Path(input_path).stem
    cloaked_path = f'output/cloaked/{input_stem}_cloaked.png'
    
    cloaker = ImageCloaker(device=device)
    cloak_stats = cloaker.cloak(
        input_path,
        cloaked_path,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        lpips_threshold=lpips_threshold
    )
    
    # Step 2: ATTACK TEST
    print(f"\n{'*'*70}")
    print(f"STEP 2: FACE RECOGNITION ATTACK TEST")
    print(f"{'*'*70}")
    
    tester = FaceRecognitionTester(device=device)
    test_stats = tester.test(input_path, cloaked_path, reference_path=reference_path)
    
    # Step 3: GENERATE COMPARISON
    print(f"\n{'*'*70}")
    print(f"STEP 3: GENERATE VISUAL COMPARISON")
    print(f"{'*'*70}")
    
    comparison_path = f'output/comparison/{input_stem}_comparison.png'
    create_comparison_image(
        input_path,
        cloaked_path,
        comparison_path,
        label_original='Original Face',
        label_cloaked='Adversarially Cloaked'
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE - RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nCloaking Statistics:")
    print(f"  • Perturbation Magnitude (L∞): {cloak_stats['perturbation_mag']:.6f}")
    print(f"  • Embedding Distance (L2):     {cloak_stats['embedding_distance']:.4f}")
    if cloak_stats['lpips_score']:
        print(f"  • LPIPS Perceptual Score:      {cloak_stats['lpips_score']:.4f}")
    
    print(f"\nAttack Test Statistics:")
    print(f"  • Original vs Cloaked Distance:    {test_stats['original_vs_cloaked_distance']:.4f}")
    print(f"  • Original vs Cloaked Similarity:  {test_stats['original_vs_cloaked_similarity']:.4f}")
    
    print(f"\nOutput Files:")
    print(f"  • Cloaked Image:      {os.path.abspath(cloaked_path)}")
    print(f"  • Comparison Image:   {os.path.abspath(comparison_path)}")
    
    print(f"\n{'='*70}")
    print(f"[✓] SUCCESS! All steps completed.")
    print(f"View the comparison image to see the results.")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="PixelShield Phase 1: Complete adversarial cloaking pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Basic usage with automatic parameters:
  python run_pipeline.py images/input/my_face.jpg
  
  # With reference face for attack testing:
  python run_pipeline.py images/input/my_face.jpg --reference images/input/reference_face.jpg
  
  # Strong cloaking (more disruption):
  python run_pipeline.py images/input/my_face.jpg --epsilon 16/255 --steps 100
  
  # Weak cloaking (less visible change):
  python run_pipeline.py images/input/my_face.jpg --epsilon 4/255 --steps 20
        """
    )
    
    parser.add_argument('input', help='Path to input face image')
    parser.add_argument('-r', '--reference', default=None,
                       help='Path to reference image (another photo of same person)')
    parser.add_argument('--epsilon', type=float, default=8/255,
                       help='Max pixel change (default: 8/255)')
    parser.add_argument('--alpha', type=float, default=2/255,
                       help='PGD step size (default: 2/255)')
    parser.add_argument('--steps', type=int, default=40,
                       help='Number of optimization steps (default: 40)')
    parser.add_argument('--lpips-threshold', type=float, default=0.05,
                       help='Max perceptual distance (default: 0.05)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    try:
        run_full_pipeline(
            args.input,
            reference_path=args.reference,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.steps,
            lpips_threshold=args.lpips_threshold,
            device=args.device
        )
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

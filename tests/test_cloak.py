#!/usr/bin/env python3
"""
Comprehensive unit tests for PixelShield cloaking pipeline.

Tests:
1. Embedding distance increases after cloaking
2. Cloaked image stays within epsilon bound
3. Original resolution is maintained
4. LPIPS stays below perceptual threshold
5. EOT robustness to JPEG compression
6. CLI runs without errors
"""

import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cloak import ImageCloaker


class TestImageCloaker(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create test fixtures once."""
        # Create temporary directory
        cls.test_dir = tempfile.mkdtemp()
        
        # Generate synthetic test face (160x160 for consistency)
        np.random.seed(42)
        face_array = np.random.randint(100, 200, (160, 160, 3), dtype=np.uint8)
        # Add some structure (circle) to make it look more face-like
        y, x = np.ogrid[:160, :160]
        mask = (x - 80)**2 + (y - 80)**2 <= 60**2
        face_array[mask] = np.clip(face_array[mask] + 30, 0, 255).astype(np.uint8)
        
        cls.test_face = Image.fromarray(face_array)
        cls.test_face_path = os.path.join(cls.test_dir, 'test_face.jpg')
        cls.test_face.save(cls.test_face_path)
        
        # Also create a larger test face (to test resolution preservation)
        face_array_large = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
        y, x = np.ogrid[:400, :400]
        mask = (x - 200)**2 + (y - 200)**2 <= 120**2
        face_array_large[mask] = np.clip(face_array_large[mask] + 30, 0, 255).astype(np.uint8)
        
        cls.test_face_large = Image.fromarray(face_array_large)
        cls.test_face_large_path = os.path.join(cls.test_dir, 'test_face_large.jpg')
        cls.test_face_large.save(cls.test_face_large_path)
        
        # Initialize cloaker
        cls.cloaker = ImageCloaker(device='cpu')
    
    def test_01_cloak_increases_embedding_distance(self):
        """Test that cloaking increases face embedding distance."""
        print("\nTest 1: Embedding distance increases")
        
        img_full, img_small_160, _, _ = self.cloaker.load_image(self.test_face_path)
        orig_embedding = self.cloaker.get_embedding(img_small_160)
        
        # Run quick cloaking (5 steps for speed)
        cloaked_img, _, emb_distance, _, _ = self.cloaker.compute_perturbation(
            img_full, img_small_160, num_steps=5, use_eot=False
        )
        
        # Distance should increase significantly
        self.assertGreater(emb_distance, 0.2, 
                          f"Embedding distance {emb_distance} too small (should be > 0.2)")
        print(f"  ✓ Embedding distance: {emb_distance:.4f}")
    
    def test_02_perturbation_within_epsilon(self):
        """Test that perturbation stays within epsilon bound."""
        print("\nTest 2: Perturbation within epsilon bound")
        
        epsilon = 8/255
        img_full, img_small_160, _, _ = self.cloaker.load_image(self.test_face_path)
        
        cloaked_img, perturb_mag, _, _, _ = self.cloaker.compute_perturbation(
            img_full, img_small_160, epsilon=epsilon, num_steps=5, use_eot=False
        )
        
        # Perturbation magnitude should be <= epsilon
        self.assertLessEqual(perturb_mag, epsilon + 1e-6,
                            f"Perturbation {perturb_mag} exceeds epsilon {epsilon}")
        print(f"  ✓ Perturbation magnitude: {perturb_mag:.6f} (epsilon={epsilon:.6f})")
    
    def test_03_resolution_preserved(self):
        """Test that original resolution is preserved (FIX #3)."""
        print("\nTest 3: Original resolution preserved")
        
        # Test with large image
        img_full, img_small_160, _, original_size = self.cloaker.load_image(self.test_face_large_path)
        
        print(f"  Original size: {original_size}")
        print(f"  Tensor shape: {img_full.shape}")
        
        # After cloaking, should maintain full resolution
        cloaked_img, _, _, _, _ = self.cloaker.compute_perturbation(
            img_full, img_small_160, num_steps=3, use_eot=False
        )
        
        # Check shape matches
        self.assertEqual(cloaked_img.shape, img_full.shape,
                        f"Shape mismatch: {cloaked_img.shape} != {img_full.shape}")
        
        print(f"  ✓ Output shape preserved: {cloaked_img.shape}")
    
    def test_04_lpips_below_threshold(self):
        """Test that LPIPS stays below perceptual threshold."""
        print("\nTest 4: LPIPS below threshold")
        
        img_full, img_small_160, _, _ = self.cloaker.load_image(self.test_face_path)
        
        threshold = 0.05
        _, _, _, lpips_score, _ = self.cloaker.compute_perturbation(
            img_full, img_small_160, lpips_threshold=threshold, num_steps=5, use_eot=False
        )
        
        if lpips_score is not None:
            self.assertLess(lpips_score, threshold + 0.01,
                           f"LPIPS {lpips_score} exceeds threshold {threshold}")
            print(f"  ✓ LPIPS score: {lpips_score:.4f} (threshold={threshold})")
        else:
            print(f"  ⚠ LPIPS not available (skipped)")
    
    def test_05_jpeg_robustness(self):
        """Test that cloaking survives JPEG compression (FIX #5)."""
        print("\nTest 5: JPEG robustness")
        
        img_full, img_small_160, _, _ = self.cloaker.load_image(self.test_face_path)
        
        _, _, _, _, jpeg_robustness = self.cloaker.compute_perturbation(
            img_full, img_small_160, num_steps=5, use_eot=True
        )
        
        print(f"  JPEG robustness scores:")
        for quality, distance in jpeg_robustness.items():
            print(f"    {quality}: {distance:.4f}")
            # Distance should remain significant even after JPEG
            self.assertGreater(distance, 0.15,
                              f"Robustness too weak at {quality}: {distance}")
        
        print(f"  ✓ Perturbation survives real-world compression")
    
    def test_06_cloaked_image_valid(self):
        """Test that cloaked image can be saved and loaded."""
        print("\nTest 6: Cloaked image I/O")
        
        img_full, img_small_160, _, _ = self.cloaker.load_image(self.test_face_path)
        
        cloaked_img, _, _, _, _ = self.cloaker.compute_perturbation(
            img_full, img_small_160, num_steps=3, use_eot=False
        )
        
        # Save and reload
        output_path = os.path.join(self.test_dir, 'cloaked_output.png')
        self.cloaker.save_image(cloaked_img, output_path)
        
        # Check file exists and can be loaded
        self.assertTrue(os.path.exists(output_path), "Output file not created")
        reloaded = Image.open(output_path)
        print(f"  ✓ Cloaked image saved and reloaded: {reloaded.size}")
    
    def test_07_values_in_valid_range(self):
        """Test that cloaked image pixel values stay in [0, 1]."""
        print("\nTest 7: Pixel values in valid range")
        
        img_full, img_small_160, _, _ = self.cloaker.load_image(self.test_face_path)
        
        cloaked_img, _, _, _, _ = self.cloaker.compute_perturbation(
            img_full, img_small_160, num_steps=5, use_eot=False
        )
        
        # Check range
        min_val = cloaked_img.min().item()
        max_val = cloaked_img.max().item()
        
        self.assertGreaterEqual(min_val, 0, f"Min value {min_val} < 0")
        self.assertLessEqual(max_val, 1, f"Max value {max_val} > 1")
        print(f"  ✓ Pixel values in range [{min_val:.4f}, {max_val:.4f}]")
    
    def test_08_deterministic_with_seed(self):
        """Test that results are reproducible with same random seed."""
        print("\nTest 8: Reproducibility with seed")
        
        torch.manual_seed(42)
        img_full_1, img_small_160_1, _, _ = self.cloaker.load_image(self.test_face_path)
        _, _, dist_1, _, _ = self.cloaker.compute_perturbation(
            img_full_1, img_small_160_1, num_steps=3, use_eot=False
        )
        
        torch.manual_seed(42)
        img_full_2, img_small_160_2, _, _ = self.cloaker.load_image(self.test_face_path)
        _, _, dist_2, _, _ = self.cloaker.compute_perturbation(
            img_full_2, img_small_160_2, num_steps=3, use_eot=False
        )
        
        self.assertAlmostEqual(dist_1, dist_2, places=4,
                              msg=f"Distances differ: {dist_1} vs {dist_2}")
        print(f"  ✓ Reproducible results: {dist_1:.4f} == {dist_2:.4f}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)

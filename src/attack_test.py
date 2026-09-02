#!/usr/bin/env python3
"""
Step 4: Test script - run face recognition/embedding similarity test on original vs. cloaked.

This simulates an attacker trying to use the face for recognition or swapping.
We measure: does the face still match a reference photo?
- Original: should have HIGH similarity
- Cloaked: should have LOW similarity (attack disrupted)
"""

import argparse
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
from pathlib import Path


class FaceRecognitionTester:
    """Test how well face recognition works on original vs. cloaked images."""
    
    def __init__(self, device='cpu'):
        """Initialize the face embedding model."""
        self.device = device
        print(f"[*] Initializing FaceNet model on {device.upper()}...")
        
        self.model = InceptionResnetV1(pretrained='vggface2')
        self.model = self.model.to(device).eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print("[+] Model loaded successfully.")
    
    def load_and_preprocess(self, image_path):
        """Load image and preprocess for FaceNet."""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((160, 160), Image.Resampling.LANCZOS)
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        return img_tensor, img
    
    def get_embedding(self, img_tensor):
        """Extract face embedding."""
        with torch.no_grad():
            x = self.normalize(img_tensor)
            embedding = self.model(x)
        return embedding.detach()
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        return torch.nn.functional.cosine_similarity(vec1, vec2).item()
    
    def euclidean_distance(self, vec1, vec2):
        """Compute L2 Euclidean distance between two vectors."""
        return torch.norm(vec1 - vec2, p=2).item()
    
    def test(self, original_path, cloaked_path, reference_path=None):
        """
        Test: Compare original and cloaked images against a reference face.
        
        If reference_path is provided:
          - Load reference image (another photo of the same person)
          - Compute similarity scores for original and cloaked
          - Show that original matches reference, but cloaked doesn't
        
        Otherwise, just show the embedding distance between original and cloaked.
        """
        print(f"\n{'='*70}")
        print(f"Face Recognition Attack Test")
        print(f"{'='*70}")
        
        # Load images
        print(f"\n[*] Loading images...")
        orig_tensor, orig_img = self.load_and_preprocess(original_path)
        print(f"[+] Original: {Path(original_path).name} ({orig_img.size})")
        
        cloak_tensor, cloak_img = self.load_and_preprocess(cloaked_path)
        print(f"[+] Cloaked:  {Path(cloaked_path).name} ({cloak_img.size})")
        
        # Get embeddings
        print(f"\n[*] Computing embeddings...")
        orig_emb = self.get_embedding(orig_tensor)
        cloak_emb = self.get_embedding(cloak_tensor)
        
        # Compare original vs. cloaked
        orig_cloak_distance = self.euclidean_distance(orig_emb, cloak_emb)
        orig_cloak_similarity = self.cosine_similarity(orig_emb, cloak_emb)
        
        print(f"\n{'='*70}")
        print(f"ORIGINAL vs. CLOAKED")
        print(f"{'='*70}")
        print(f"L2 Distance:       {orig_cloak_distance:.4f}")
        print(f"Cosine Similarity: {orig_cloak_similarity:.4f}")
        print(f"  (0.0 = completely different, 1.0 = identical)")
        
        # If reference provided, test how well each matches the reference
        if reference_path:
            print(f"\n[*] Loading reference face...")
            ref_tensor, ref_img = self.load_and_preprocess(reference_path)
            print(f"[+] Reference: {Path(reference_path).name} ({ref_img.size})")
            
            ref_emb = self.get_embedding(ref_tensor)
            
            # Compare original to reference
            orig_ref_distance = self.euclidean_distance(orig_emb, ref_emb)
            orig_ref_similarity = self.cosine_similarity(orig_emb, ref_emb)
            
            # Compare cloaked to reference
            cloak_ref_distance = self.euclidean_distance(cloak_emb, ref_emb)
            cloak_ref_similarity = self.cosine_similarity(cloak_emb, ref_emb)
            
            print(f"\n{'='*70}")
            print(f"ATTACK TEST: Matching to Reference Face")
            print(f"{'='*70}")
            print(f"\nORIGINAL vs. REFERENCE:")
            print(f"  L2 Distance:       {orig_ref_distance:.4f}")
            print(f"  Cosine Similarity: {orig_ref_similarity:.4f}")
            print(f"\nCLOAKED vs. REFERENCE:")
            print(f"  L2 Distance:       {cloak_ref_distance:.4f}")
            print(f"  Cosine Similarity: {cloak_ref_similarity:.4f}")
            
            # Compute degradation
            sim_drop = orig_ref_similarity - cloak_ref_similarity
            dist_increase = cloak_ref_distance - orig_ref_distance
            
            print(f"\n{'='*70}")
            print(f"ATTACK EFFECTIVENESS")
            print(f"{'='*70}")
            print(f"Cosine Similarity Drop:  {sim_drop:.4f} ({sim_drop/max(orig_ref_similarity, 0.001)*100:.1f}%)")
            print(f"L2 Distance Increase:    {dist_increase:.4f}")
            
            if sim_drop > 0.2:  # Heuristic threshold
                print(f"\n[✓] SUCCESS: Cloaking significantly degraded face recognition!")
            else:
                print(f"\n[!] Limited effect: Face still somewhat recognizable.")
        
        print(f"\n{'='*70}")
        print(f"[+] Test complete.")
        print(f"{'='*70}\n")
        
        return {
            'original_vs_cloaked_distance': orig_cloak_distance,
            'original_vs_cloaked_similarity': orig_cloak_similarity
        }


def main():
    parser = argparse.ArgumentParser(
        description="Test face recognition robustness: original vs. cloaked image"
    )
    parser.add_argument('original', help='Path to original image')
    parser.add_argument('cloaked', help='Path to cloaked image')
    parser.add_argument('-r', '--reference', default=None,
                       help='Path to reference image (another photo of same person) for matching test')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"[*] PyTorch version: {torch.__version__}")
    print(f"[*] CUDA available: {torch.cuda.is_available()}")
    print(f"[*] Device: {device.upper()}")
    
    tester = FaceRecognitionTester(device=device)
    tester.test(args.original, args.cloaked, reference_path=args.reference)


if __name__ == '__main__':
    main()

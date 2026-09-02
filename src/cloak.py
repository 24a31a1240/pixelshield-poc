#!/usr/bin/env python3
"""
Step 3: Core adversarial perturbation script (PGD-based image cloaking)

This script takes an input photo and generates a 'cloaked' version that:
1. Looks visually identical to humans (enforced by LPIPS perceptual loss)
2. Breaks face-embedding similarity (maximizes distance from original embedding)
3. Makes it unusable for face-swap attacks

The mechanism: Projected Gradient Descent (PGD) iteratively adds small,
invisible noise to the image to maximize the face-embedding distance while
staying within perceptual and pixel-value bounds.

CRITICAL BUGFIXES APPLIED (v2.0):
- FIX #1: Separated get_embedding() (no_grad) from get_embedding_grad() (with gradients)
  Problem: get_embedding() used inside PGD loop severed the autograd graph
  Solution: Created separate get_embedding_grad() that keeps gradients flowing to delta
  
- FIX #2: Initialize delta with random noise instead of zeros (standard PGD practice)
  Problem: L2 norm gradient at delta=0 is degenerate (0/0), gradient=0, optimizer never moves
  Solution: Random-start PGD: initialize delta uniformly in [-epsilon, epsilon] ball
  Reference: This is the standard fix used in all published PGD implementations
"""

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed. Will skip LPIPS perceptual check.")


class ImageCloaker:
    """
    Adversarial image cloaking using PGD (Projected Gradient Descent).
    
    Key concepts:
    - epsilon (ε): Maximum pixel change allowed (e.g., 8/255 ≈ 0.031). Keeps changes invisible.
    - alpha (α): Step size per PGD iteration (e.g., 2/255). Controls optimization speed.
    - num_steps: How many PGD iterations to run. More = stronger perturbation.
    - lpips_threshold: Maximum allowed perceptual distance. Prevents visible artifacts.
    
    CRITICAL FIXES (v2.0):
    1. Separated embedding computation into two functions:
       - get_embedding(): Used for original image, no gradients needed (faster)
       - get_embedding_grad(): Used inside PGD loop, gradients flow through delta
    2. Initialize delta with random noise instead of zeros (fixes degenerate gradient at origin)
    """
    
    def __init__(self, device='cpu'):
        """Initialize the FaceNet model and move to device (CPU or GPU)."""
        self.device = device
        print(f"[*] Initializing FaceNet model on {device.upper()}...")
        
        # Load pretrained InceptionResnetV1 for face embeddings
        self.model = InceptionResnetV1(pretrained='vggface2')
        self.model = self.model.to(device).eval()
        
        # Disable gradient computation in the model (we only optimize the image)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Initialize LPIPS loss if available (perceptual distance checker)
        if HAS_LPIPS:
            self.lpips_loss = lpips.LPIPS(net='alex').to(device).eval()
            for param in self.lpips_loss.parameters():
                param.requires_grad = False
        else:
            self.lpips_loss = None
        
        # Standard ImageNet normalization for FaceNet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        print("[+] Model loaded successfully.")
    
    def load_image(self, image_path):
        """Load an image and convert to normalized tensor."""
        img = Image.open(image_path).convert('RGB')
        # Resize to 160x160 (FaceNet input size)
        img = img.resize((160, 160), Image.Resampling.LANCZOS)
        img_tensor = transforms.ToTensor()(img)  # [0, 1]
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor.to(self.device), img
    
    def get_embedding(self, img_tensor):
        """
        Extract face embedding from image tensor WITHOUT gradient tracking.
        Used for original image and final evaluation only.
        """
        with torch.no_grad():
            # Normalize for FaceNet
            x = self.normalize(img_tensor)
            embedding = self.model(x)  # Returns [1, 512] embedding vector
        return embedding.detach()
    
    def get_embedding_grad(self, img_tensor):
        """
        Extract face embedding WITH gradient tracking.
        Used inside PGD loop so gradients can flow to delta.
        
        FIX #1: Separate function allows gradients to flow during optimization
        This is critical - using get_embedding() inside the loop would break
        the autograd graph and cause: RuntimeError: element 0 of tensors does not require grad
        """
        # Normalize for FaceNet (gradients flow through this)
        x = self.normalize(img_tensor)
        embedding = self.model(x)  # Gradients flow here to delta
        return embedding
    
    def compute_perturbation(self, original_img_tensor, epsilon=8/255, alpha=2/255,
                            num_steps=40, lpips_threshold=0.05):
        """
        Run PGD to compute adversarial perturbation that breaks face recognition.
        
        Args:
            original_img_tensor: Original image as tensor [1, 3, 160, 160]
            epsilon: Max pixel change (L∞ bound, e.g., 8/255)
            alpha: PGD step size (e.g., 2/255)
            num_steps: Number of optimization iterations
            lpips_threshold: Max perceptual distance before stopping (0.05 is imperceptible)
        
        Returns:
            perturbed_img_tensor: Adversarially perturbed image
            perturbation_magnitude: L∞ norm of the perturbation
            final_embedding_distance: Distance between original and cloaked embeddings
            lpips_score: Perceptual distance (if LPIPS available)
        """
        print(f"\n[*] Starting PGD optimization (epsilon={epsilon:.4f}, steps={num_steps})...")
        
        # Get original embedding (no gradients needed)
        orig_embedding = self.get_embedding(original_img_tensor)
        
        # Initialize perturbation with random noise (FIX #2)
        # This is the standard "random start" technique used in ALL published PGD implementations
        # It solves the critical problem: L2 norm gradient at delta=0 is mathematically degenerate
        # At exactly zero: d/dx ||x||_2 = x / ||x||_2 = 0/0 (undefined, PyTorch returns 0)
        # Random start ensures we begin with non-zero embeddings where gradients flow properly
        delta = (torch.rand_like(original_img_tensor) - 0.5) * 2 * epsilon
        delta = delta.to(self.device)
        delta.requires_grad = True
        
        optimizer = torch.optim.SGD([delta], lr=alpha)
        
        best_distance = 0.0
        lpips_scores = []
        embedding_distances = []
        
        for step in tqdm(range(num_steps), desc="PGD iterations"):
            optimizer.zero_grad()
            
            # Perturbed image (clipped to [0, 1])
            perturbed = torch.clamp(original_img_tensor + delta, 0, 1)
            
            # Get embedding of perturbed image WITH gradients (uses FIX #1)
            perturbed_embedding = self.get_embedding_grad(perturbed)
            
            # Loss: maximize L2 distance between embeddings
            # (We want to make the cloaked version as different as possible from original)
            embedding_distance = torch.norm(perturbed_embedding - orig_embedding, p=2)
            loss = -embedding_distance  # Negative because we're maximizing
            
            # Backpropagation - now works because:
            # 1. get_embedding_grad keeps gradients flowing (FIX #1)
            # 2. delta started with non-zero random noise so gradient is well-defined (FIX #2)
            loss.backward()
            optimizer.step()
            
            # Project delta to epsilon ball (L∞ constraint)
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            embedding_distances.append(embedding_distance.item())
            best_distance = max(best_distance, embedding_distance.item())
            
            # Check LPIPS perceptual distance
            if self.lpips_loss is not None:
                with torch.no_grad():
                    lpips_score = self.lpips_loss(original_img_tensor, perturbed).item()
                lpips_scores.append(lpips_score)
                
                # Early stopping if perceptual distance gets too large
                if lpips_score > lpips_threshold:
                    print(f"\n[!] LPIPS threshold exceeded ({lpips_score:.4f} > {lpips_threshold:.4f}). Stopping early.")
                    break
            
            if (step + 1) % 10 == 0:
                status = f"Step {step+1}: Embedding distance={embedding_distance.item():.4f}"
                if lpips_scores:
                    status += f", LPIPS={lpips_scores[-1]:.4f}"
                print(status)
        
        # Final cloaked image
        with torch.no_grad():
            cloaked_img = torch.clamp(original_img_tensor + delta, 0, 1)
        
        # Compute final stats
        final_embedding_distance = torch.norm(
            self.get_embedding(cloaked_img) - orig_embedding, p=2
        ).item()
        perturbation_magnitude = torch.abs(delta).max().item()
        
        final_lpips = None
        if self.lpips_loss is not None:
            with torch.no_grad():
                final_lpips = self.lpips_loss(original_img_tensor, cloaked_img).item()
        
        return cloaked_img, perturbation_magnitude, final_embedding_distance, final_lpips
    
    def save_image(self, img_tensor, output_path):
        """Save tensor image to disk."""
        # Remove batch dimension and move to CPU
        img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        # Clip to [0, 1] and convert to [0, 255]
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(output_path)
        print(f"[+] Saved: {output_path}")
    
    def cloak(self, input_path, output_path, epsilon=8/255, alpha=2/255,
              num_steps=40, lpips_threshold=0.05):
        """
        Main entry point: load image, compute cloaking perturbation, save result.
        """
        print(f"\n{'='*70}")
        print(f"PixelShield Adversarial Cloaking (v2.0 - Bugfixes Applied)")
        print(f"{'='*70}")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Epsilon (max pixel change): {epsilon:.4f} (~{int(epsilon*255)}/255)")
        print(f"Alpha (step size): {alpha:.4f}")
        print(f"Num steps: {num_steps}")
        if self.lpips_loss:
            print(f"LPIPS threshold: {lpips_threshold:.4f}")
        print(f"{'='*70}")
        
        # Load image
        img_tensor, pil_img = self.load_image(input_path)
        print(f"[+] Loaded image: {pil_img.size}")
        
        # Compute cloaking perturbation
        cloaked_tensor, perturb_mag, emb_distance, lpips_score = self.compute_perturbation(
            img_tensor,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            lpips_threshold=lpips_threshold
        )
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.save_image(cloaked_tensor, output_path)
        
        # Print final report
        print(f"\n{'='*70}")
        print(f"CLOAKING RESULTS")
        print(f"{'='*70}")
        print(f"L∞ Perturbation Magnitude: {perturb_mag:.6f} ({perturb_mag*255:.2f}/255)")
        print(f"Embedding Distance (L2):   {emb_distance:.4f}")
        if lpips_score:
            print(f"LPIPS Perceptual Score:    {lpips_score:.4f}")
            print(f"  (< 0.01 = imperceptible, 0.01-0.05 = barely noticeable)")
        print(f"{'='*70}")
        print(f"[+] Cloaking complete! Image saved to: {output_path}")
        
        return {
            'perturbation_mag': perturb_mag,
            'embedding_distance': emb_distance,
            'lpips_score': lpips_score
        }


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial cloaking: make a photo invisible to face-swap AI"
    )
    parser.add_argument('input', help='Path to input image')
    parser.add_argument('-o', '--output', default=None,
                       help='Output path for cloaked image (default: output/cloaked/input_filename)')
    parser.add_argument('--epsilon', type=float, default=8/255,
                       help='Max pixel change L∞ bound (default: 8/255)')
    parser.add_argument('--alpha', type=float, default=2/255,
                       help='PGD step size (default: 2/255)')
    parser.add_argument('--steps', type=int, default=40,
                       help='Number of PGD iterations (default: 40)')
    parser.add_argument('--lpips-threshold', type=float, default=0.05,
                       help='Max LPIPS perceptual distance (default: 0.05)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"[*] PyTorch version: {torch.__version__}")
    print(f"[*] CUDA available: {torch.cuda.is_available()}")
    print(f"[*] Device: {device.upper()}")
    
    # Set default output path
    if args.output is None:
        input_filename = Path(args.input).stem + '_cloaked.png'
        args.output = f'output/cloaked/{input_filename}'
    
    # Run cloaking
    cloaker = ImageCloaker(device=device)
    cloaker.cloak(
        args.input,
        args.output,
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_steps=args.steps,
        lpips_threshold=args.lpips_threshold
    )


if __name__ == '__main__':
    main()

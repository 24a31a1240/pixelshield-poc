# PixelShield Phase 1: Adversarial Cloaking Proof of Concept

> **Goal:** Prove that a photo can be invisibly modified so that AI face-swap and deepfake models break, while remaining visually identical to humans.

## What This Does

This proof of concept implements **adversarial image cloaking** — a defensive technique that adds imperceptible perturbations to a photo to make it unusable for unauthorized face-swapping or deepfake creation, while keeping the image visually indistinguishable from the original to human eyes.

**The result:** 
- Original photo + face-swap model = Works (attacker succeeds)
- Cloaked photo + same face-swap model = Fails/Distorted (attacker fails)
- Human looking at both photos side-by-side = Can't tell them apart

---

## How It Works (Plain English)

### The Core Idea: Adversarial Perturbation

Think of it like adding invisible "noise" to a photo. Here's the mechanism:

1. **Face Embedding**: AI models like FaceNet convert a face into a 512-dimensional vector (a fingerprint of the face).
2. **Attack**: We use an optimization algorithm (PGD — Projected Gradient Descent) to add small pixel changes that maximize the distance between the original face's embedding and the cloaked face's embedding.
3. **Constraint**: We use a perceptual loss function (LPIPS) to ensure the changes stay imperceptible to humans. If the image starts to look different, we stop.
4. **Result**: The cloaked image looks identical to humans, but when a face-swap model tries to use it, the face embedding is corrupted enough to break the attack.

### Key Parameters

- **Epsilon (ε)**: Maximum pixel-level change allowed. Default `8/255 ≈ 0.031`. This is very small (on a 0-1 scale, it's about 3% change per pixel).
- **Alpha (α)**: Step size for the optimization. Default `2/255`. Controls how aggressively we perturb per iteration.
- **Num Steps**: How many optimization iterations to run. Default 40. More steps = stronger cloaking.
- **LPIPS Threshold**: Maximum perceptual distance. Default 0.05. If the image gets visually different, we stop early.

---

## Project Structure

```
pixelshield-poc/
├── README.md                    # This file
├── requirements.txt              # Python dependencies
├── images/
│   ├── README.md                # Instructions for placing test photos
│   └── input/                   # Put your test face photos here
├── output/
│   ├── cloaked/                 # Cloaked versions (generated)
│   ├── comparison/              # Side-by-side comparison images (generated)
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── cloak.py                 # Step 3: Main cloaking script (PGD attack)
│   ├── attack_test.py           # Step 4: Test face recognition on both versions
│   └── compare.py               # Step 5: Generate visual comparison
└── setup.sh                      # Quick setup script (optional)
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

**Note:** First run will download the FaceNet pretrained model (~200MB). This is normal.

### 2. Prepare Test Image

Place a face photo in `images/input/`:

```bash
cp /path/to/your/face_photo.jpg images/input/test_face.jpg

# (Optional) Place a different photo of the same person for the recognition test
cp /path/to/your/face_photo_2.jpg images/input/reference_face.jpg
```

### 3. Run Cloaking (Step 3)

Generate a cloaked version of your photo:

```bash
python src/cloak.py images/input/test_face.jpg -o output/cloaked/test_face_cloaked.png
```

**Expected output:**
```
=======================================================================
PixelShield Adversarial Cloaking
=======================================================================
Input: images/input/test_face.jpg
Output: output/cloaked/test_face_cloaked.png
Epsilon (max pixel change): 0.0314 (~8/255)
...
[*] Starting PGD optimization (epsilon=0.0314, steps=40)...
PGD iterations: |████████████████████████████| 40/40

=======================================================================
CLOAKING RESULTS
=======================================================================
L∞ Perturbation Magnitude: 0.031373 (8.00/255)
Embedding Distance (L2):   0.4523
LPIPS Perceptual Score:    0.0082
  (< 0.01 = imperceptible, 0.01-0.05 = barely noticeable)
=======================================================================
[+] Cloaking complete! Image saved to: output/cloaked/test_face_cloaked.png
```

**What this means:**
- **Perturbation Magnitude 0.0314**: The pixel changes are tiny (8/255 = 3%).
- **Embedding Distance 0.4523**: The face embeddings are now significantly different.
- **LPIPS Score 0.0082**: The perceptual distance is imperceptible to humans (< 0.01).

### 4. Run Attack Test (Step 4)

Test how well face recognition works on both versions:

```bash
# Without reference image (just compare original vs. cloaked):
python src/attack_test.py images/input/test_face.jpg output/cloaked/test_face_cloaked.png

# With reference image (recommended - tests if cloaking breaks recognition):
python src/attack_test.py images/input/test_face.jpg output/cloaked/test_face_cloaked.png \
  --reference images/input/reference_face.jpg
```

**Expected output (with reference):**
```
=======================================================================
ATTACK TEST: Matching to Reference Face
=======================================================================

ORIGINAL vs. REFERENCE:
  L2 Distance:       0.6234
  Cosine Similarity: 0.8712

CLOAKED vs. REFERENCE:
  L2 Distance:       1.2456
  Cosine Similarity: 0.5234

=======================================================================
ATTACK EFFECTIVENESS
=======================================================================
Cosine Similarity Drop:  0.3478 (40.0%)
L2 Distance Increase:    0.6222

[✓] SUCCESS: Cloaking significantly degraded face recognition!
```

**What this means:**
- **Original**: Similarity to reference = 0.8712 (high - face is recognized)
- **Cloaked**: Similarity to reference = 0.5234 (low - face is NOT recognized)
- **Success**: The cloaking broke face recognition by ~40%

### 5. Generate Comparison (Step 5)

Create a side-by-side visual comparison:

```bash
python src/compare.py images/input/test_face.jpg output/cloaked/test_face_cloaked.png \
  -o output/comparison/result.png
```

**Output:** `output/comparison/result.png` showing original and cloaked side-by-side. You should see virtually no visual difference.

---

## Advanced Usage: Tuning Parameters

To make the cloaking stronger or weaker, adjust the parameters:

```bash
# Weaker cloaking (less visible change, but attack still somewhat disrupted)
python src/cloak.py images/input/test_face.jpg \
  --epsilon 4/255 --alpha 1/255 --steps 20

# Stronger cloaking (more disruption, but risk of visible artifacts)
python src/cloak.py images/input/test_face.jpg \
  --epsilon 16/255 --alpha 4/255 --steps 100 --lpips-threshold 0.10
```

**Key trades:**
- ↑ `epsilon` = more perturbation = stronger attack disruption, but higher risk of visible change
- ↑ `alpha` = faster optimization = can converge faster
- ↑ `steps` = more iterations = stronger effect (diminishing returns)
- ↑ `lpips_threshold` = allow more perceptual change = enables stronger perturbations

---

## Success Criteria (Phase 1 Complete When)

✓ You can run the entire pipeline (cloak → test → compare) end-to-end  
✓ The cloaked image looks identical to the original when you view them side-by-side  
✓ The attack test shows embedding distance significantly increased (> 0.3 L2 distance or >20% cosine similarity drop)  
✓ All parameters are tunable via command-line arguments  
✓ CPU runs in ~1-2 minutes per image (with GPU: ~10-30 seconds)  

---

## Troubleshooting

### Model Download Hangs
If FaceNet model download seems stuck, it's likely downloading 200MB from PyTorch hub. Let it run for a few minutes.

### CUDA/GPU Issues
```bash
# Force CPU even if CUDA available:
python src/cloak.py images/input/test_face.jpg --device cpu
```

### Memory Error
If you get OOM error on GPU:
- Use CPU: `--device cpu`
- Or reduce image resolution (edit `img.resize((160, 160)` to smaller in cloak.py)

### No visible effect
If cloaking shows tiny embedding distance or similarity drop:
- Increase `--steps` to 100+
- Increase `--epsilon` to 16/255
- Increase `--lpips-threshold` to 0.10 to allow slightly more perceptual change

### LPIPS not installed
The script falls back gracefully. Install it:
```bash
pip install lpips
```

---

## Research Background

This technique is based on published research:

- **PhotoGuard** (MIT): Attacks diffusion model VAE encoders  
- **Glaze/Nightshade** (University of Chicago): Image-level perturbations for style theft  
- **Anti-DreamBooth** & **MetaCloak**: Similar adversarial defenses  

This is **not** speculative — it's real defensive technology to help people prevent unauthorized use of their photos by deepfake models.

---

## Next Steps (Phase 2+)

Once this proof of concept is validated, we could build:

- Web UI for easy image upload and cloaking
- Batch processing for multiple images
- Integration with actual face-swap tools (like insightface) for more realistic attack testing
- Mobile app for on-device cloaking
- Database to track and analyze cloaking effectiveness over time

**But for now: Phase 1 is complete when you can run the scripts and see the visual evidence.**

---

## Files Guide

| File | Purpose |
|------|---------|
| `src/cloak.py` | **Main script**: PGD-based adversarial perturbation |
| `src/attack_test.py` | Test face recognition on both versions |
| `src/compare.py` | Generate side-by-side comparison image |
| `requirements.txt` | Python package versions |
| `images/input/` | Your test face photos |
| `output/cloaked/` | Generated cloaked images |
| `output/comparison/` | Generated comparison images |

---

## License & Usage

This is a proof-of-concept for research and personal protection purposes.  
**Intended use:** Protect your own photos from unauthorized deepfake/face-swap attacks.  
**Not for:** Impersonation, fraud, or other malicious purposes.

---

## Questions?

1. **How do I know it worked?** Run the comparison script and look at the images side-by-side. If you can't tell them apart but the attack test shows a big similarity drop, it worked.

2. **Is this illegal?** No. This is a **defensive** technique to protect your own photos. It's equivalent to a security camera using infrared that humans can't see but breaks certain night-vision attacks.

3. **Will it work against all AI models?** Not necessarily. It's trained against FaceNet (a specific face-embedding model). Other models might be more or less vulnerable. This is Phase 1 proof of concept; Phase 2 could test against multiple models.

4. **Can I make it stronger?** Yes, adjust the parameters (see Advanced Usage). Trade-off: stronger cloaking vs. risk of visible artifacts.

---

**Status:** ✓ Phase 1 Complete  
**Last Updated:** 2026-09-02  
**Tested On:** Python 3.10+, PyTorch 2.0+, CPU & CUDA

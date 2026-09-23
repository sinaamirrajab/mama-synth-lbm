# MAMA-SYNTH LBM 🧬✨

**Latent Bridge Matching for pre- to post-contrast breast DCE-MRI synthesis.**

**Pre- to Post-Contrast Synthesis of Breast DCE-MRI using Latent Bridge Matching**

Sina Amirrajab<sup>1</sup>, Zohaib Salahuddin<sup>1</sup>, Henry C. Woodruff<sup>1,2</sup>, Philippe Lambin<sup>1,2</sup>

<sup>1</sup> The D-Lab, Department of Precision Medicine, GROW – Research Institute for Oncology and Reproduction, Maastricht University, Maastricht, the Netherlands<br>
<sup>2</sup> Department of Radiology and Nuclear Medicine, GROW – Research Institute for Oncology and Reproduction, Maastricht University Medical Center+, Maastricht, the Netherlands

[![arXiv](https://img.shields.io/badge/arXiv-2608.10000-b31b1b.svg)](https://arxiv.org/abs/2608.10000)

This repository contains the minimal training and inference code for the MICCAI 2026 paper [Pre- to Post-Contrast Synthesis of Breast DCE-MRI using Latent Bridge Matching](https://arxiv.org/abs/2608.10000).

LBM starts from the observed pre-contrast MRI latent instead of random noise, then progressively transports it toward a peak-enhanced post-contrast latent. The goal is practical virtual contrast enhancement: preserve patient anatomy, model localized enhancement, and keep the release small enough to actually clone.

📄 **Paper:** [arXiv:2608.10000](https://arxiv.org/abs/2608.10000) · [PDF](resources/MICCAI_2026___MAMA_SYNTH.pdf)

## Method 🧠

![Latent Bridge Matching method overview](resources/method.png)

The model encodes the pre-contrast source image `x0` and peak-enhanced target image `x1` with a Stable Diffusion VAE. During training, it samples noisy bridge states between the paired source and target latents, then trains a latent UNet to predict the remaining correction toward the target latent. At inference time, only the pre-contrast image is available; the source latent is refined over a decreasing bridge-time schedule and decoded back to image space.

## Qualitative Results 🔍

![Qualitative MAMA-SYNTH LBM results](resources/qualitative_representative_n4_tumor_zoom.png)

The paper evaluates LBM on 91 DUKE validation cases from the MAMA-SYNTH setting. Tumor-mask conditioning improves the source-only LBM across the reported validation metrics, and using predicted masks gives a reviewer-facing comparison for a more realistic inference setup.

| Model | MSE ↓ | LPIPS ↓ | Tumor SSIM ↑ | FRD ↓ |
|---|---:|---:|---:|---:|
| LBM, source only | 1.023 ± 1.169 | 0.119 ± 0.035 | 0.355 ± 0.232 | 7.523 |
| LBM, source + tumor mask | 0.940 ± 1.085 | 0.114 ± 0.034 | 0.429 ± 0.185 | 4.716 |
| LBM, source + predicted mask | 0.985 ± 1.128 | 0.115 ± 0.034 | 0.356 ± 0.229 | 5.107 |
| LDM, source + tumor mask | 1.122 ± 1.248 | 0.136 ± 0.037 | 0.322 ± 0.174 | 4.786 |

## What Is Included 📦

- Minimal latent-manifest training code for the LBM paper model.
- A submission-style inference runtime for single-slice `.mha`, `.nii`, or `.nii.gz` inputs.
- Paper PDF, method figure, qualitative visualization, and citation metadata.

## Download Model Weights 🔗

The code repository stays lightweight, while the trained resources are distributed separately:

| Asset | Filename | Link |
|---|---|---|
| Trained model resources | `mama-synth-lbm-model-resources-v0.1.0.tar` | [Google Drive](https://drive.google.com/file/d/1wc4D6ZxSRThMtbIW5ASTziWf38pa2jSE/view?usp=sharing) |

Expected SHA256 checksum:

```text
eac4037d24e16a2675f89f7c445cd0292296891aa43fa63d456dfbf89adad982  mama-synth-lbm-model-resources-v0.1.0.tar
```

After downloading the model archive into a clone of this repository:

```bash
tar -xf mama-synth-lbm-model-resources-v0.1.0.tar -C .
```

This installs the runtime assets into `resources/` and restores the matching `submission_config.json`.

## Installation ⚙️

Use Python 3.10 or newer. Install PyTorch for your CUDA version first, then:

```bash
pip install -r requirements.txt
pip install -e .
```

## Train From A Latent Manifest 🚀

Training expects a `latent_manifest.csv` whose rows point to `.pt` latent payloads with at least:

```text
image_0_latent
peak_latent
segmentation_latent
patient_id
split
latent_path
```

Run the paper-style source plus tumor-mask LBM:

```bash
bash scripts/train_lbm_latents.sh /path/to/latent_manifest.csv runs/lbm_source_seg
```

By default the script uses DUKE and ISPY2 patient-id prefixes. To train on all rows:

```bash
INCLUDE_DATASETS="" bash scripts/train_lbm_latents.sh /path/to/latent_manifest.csv runs/lbm_all
```

The trainer writes checkpoints as:

```text
runs/lbm_source_seg/
  args.json
  checkpoint_rollout_metrics.csv
  checkpoint-0001000/
    unet/
    scheduler/
    training_state.json
```

## Inference With Released Weights 🧪

After unpacking `mama-synth-lbm-model-resources-v0.1.0.tar`, run local inference on one image file:

```bash
bash scripts/infer_slice.sh /path/to/patient.mha /tmp/mama_lbm_output
```

Or run on an already staged input directory:

```bash
MAMA_INPUT_DIR=/path/to/input \
MAMA_OUTPUT_DIR=/path/to/output \
python inference.py
```

The expected staged input layout is:

```text
input/
  images/
    pre-contrast-dce-mri-slice-breast/
      patient.mha
```

Predictions are written to:

```text
output/images/synthetic-contrast-dce-mri-slice-breast/output.mha
```

## Citation 📚

```bibtex
@article{amirrajab2026mamasynthlbm,
  title   = {Pre- to Post-Contrast Synthesis of Breast DCE-MRI using Latent Bridge Matching},
  author  = {Amirrajab, Sina and Salahuddin, Zohaib and Woodruff, Henry C. and Lambin, Philippe},
  journal = {arXiv preprint arXiv:2608.10000},
  year    = {2026},
  eprint  = {2608.10000},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV},
  url     = {https://arxiv.org/abs/2608.10000}
}
```

## License 🎓

This code and accompanying paper assets are released for **academic and non-commercial research use only**. They are not licensed for clinical use, commercial use, redistribution as a commercial product, or medical decision-making. See [LICENSE](LICENSE).

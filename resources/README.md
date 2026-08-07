# Resources

This directory contains the small paper assets committed with the repository:

```text
MICCAI_2026___MAMA_SYNTH.pdf
method.png
qualitative_representative_n4_tumor_zoom.pdf
qualitative_representative_n4_tumor_zoom.png
```

Large model resources are intentionally not committed. Download the released model bundle from:

```text
MODEL_DOWNLOAD_URL
```

After unpacking, the inference resource layout should also include:

```text
resources/
  checkpoint/
    args.json
    latent_scaling.json
    checkpoint-0006000/
      scheduler/
      unet/
      training_state.json
  vae/
  text_encoder/
  tokenizer/
  predicted_peak_upper/
    peak_upper_hist_mlp.pt
    feature_metadata.json
  nnunet/
    run_nnunet_inference_single_image_ensemble_best.py
    nnUNet_results/
```

The runtime is offline by default. It should not download Hugging Face or nnU-Net assets during inference.

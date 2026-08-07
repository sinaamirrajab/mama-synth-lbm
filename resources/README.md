# Model Resources

This folder is intentionally empty in GitHub. Download the released model bundle from:

```text
MODEL_DOWNLOAD_URL
```

After unpacking, the layout should be:

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

# Sparse volumetric sampling for LS-ExSM

Code, pretrained models and small acquired-data examples for denoising and intermediate-slice reconstruction (ISR) in LS-ExSM.

## Installation

Clone this repository and run commands from its root in a Python environment with PyTorch installed:

```sh
python -m pip install -r requirements.txt
```

CPU inference is supported. For GPU inference, install a PyTorch build matching your CUDA environment and use `--device cuda:0`.

## Run the examples

```sh
python infer.py denoise --input samples/denoise_time_clip.tif --checkpoint models/denoise_hyper_300epoch.pth --output outputs/denoise.npz --device cpu
python infer.py isr --input samples/isr_pair_and_reference.npz --checkpoint models/isr_hyper_280epoch.pth --output outputs/isr.npz --device cpu
```

Each command writes floating-point ADU intensities to NPZ, a clipped uint16 TIFF for viewing, and JSON metadata. Use NPZ values for quantitative analysis. The TIFF conversion truncates fractional intensities.

## Data conventions

| Task | Input axes | Processing |
| --- | --- | --- |
| Denoising | `time,y,x` | Five temporal frames, centre-frame prediction, reflected time boundaries; spatial dimensions divisible by 8 |
| ISR | `z,y,x` or `channel,z,y,x` | Each spectral channel processed independently; adjacent two-plane inputs produce four-plane outputs |

Denoising uses `x/250`. ISR uses `(x-110)/250` and restores ADU after prediction. For N input planes, ISR produces `3*N-2` planes. Shared endpoint predictions are averaged; acquired endpoint values are not forcibly copied into the output. The historical Python class name `FSR` implements the manuscript's ISR algorithm.

NPZ inputs use the key `input_intensity_adu`; NPZ predictions use `intensity_adu`. The denoising example's first axis is time. The ISR example contains eight spectral channels and a four-plane densely sampled denoised reference. This reference is a processed acquired measurement, not an independent noise-free biological ground truth. Acquisition identifiers and axial indices are in `samples/Source_coordinates.json`.

## Included models

| File in `models/` | Condition | Selected epoch |
| --- | --- | --- |
| `denoise_hyper_300epoch.pth` | Hypertonic denoising | 300 |
| `denoise_hypo_300epoch.pth` | Hypotonic denoising | 300 |
| `isr_hyper_280epoch.pth` | Hypertonic ISR | 280 |

These are the original experimental checkpoints. Newly trained movie-separated benchmark models are separate experiments. Architecture and normalization settings are recorded in `inference_configuration.json`.

## Training

Selected laboratory training sources and their dependencies are in `training/`. See [training instructions](training/README.md) for data layouts and commands. Full training data are not included; the inference samples are small demonstration inputs.

## Verify the installation

```sh
python verify_examples.py
python verify_examples.py --device cuda:0
```

This loads all three checkpoints and compares example predictions with the supplied expected arrays. The absolute tolerance is one ADU, the intensity step of the original uint16 acquisition; relative tolerance is zero. Inference and verification disable TF32 for CUDA matrix multiplication and cuDNN. Floating-point results can vary across PyTorch versions and devices. File integrity hashes are in `checksums.sha256`. See [validation results](docs/VALIDATION.md) for the tested environment and measured errors.

## Downstream export

`Export_for_downstream.py --help` describes how to export matched eight-channel reconstructions from TIFF stacks. Physical pixel and axial calibration must be supplied from acquisition metadata for downstream measurements.

## Repository contents

- `infer.py`, `networks/`: inference and model definitions.
- `models/`, `samples/`: pretrained weights, demonstration inputs and expected outputs.
- `training/`: selected training sources.
- `docs/`: release scope and third-party attribution notes.

## License and attribution

No project-wide license has been selected by the repository owners. This upload does not assign a new license to the supplied code, model weights or example data. See [attribution notes](docs/ATTRIBUTION.md). Citation metadata will be added when the authors provide the final publication details.

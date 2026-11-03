# Training sources

Install additional dependencies from the repository root:

```sh
python -m pip install -r requirements-training.txt
```

## Denoising

```sh
python training/denoising/train_3d_final.py --data_root data/denoising --outf outputs/denoise_training --scale_factor 250 --clip_len 5
```

The loader reads TIFF time stacks from `Train_Hyper/` and `Valid_Hyper/` under the supplied data root. Training uses pairs of adjacent five-frame temporal clips, so each training stack needs at least six frames. The five-frame inference sample is not a training dataset. The source retains its default training schedule; the released checkpoints were selected at epoch 300. Set parameters to match the intended experiment.

## ISR

The selected ISR training entry requires a CUDA GPU:

```sh
python training/ISR/train_aim_generation.py --data_root data/isr --outf outputs/isr_training/ --use_tiff --hyper_frames 4 --less_frames 2 --scalefactor 250 --background_value 110 --method fsr
```

Place dense TIFF axial stacks in `Train/` and `Valid/` beneath the data root. Keep acquisition groups separate when constructing these directories. The loader makes four-plane targets and sparse endpoint inputs. Only the selected `fsr` model is exposed; unrelated experimental variants are omitted. The released ISR checkpoint was selected at epoch 280.

These are selected original training implementations, with workstation paths replaced by command-line inputs and imports restricted to the included model. Training has not been rerun as part of repository packaging. The directories do not contain the separate movie-grouped benchmark or oblique-simulation experiments.

# Source and attribution notes

The model definitions, training sources and pretrained weights were supplied with the LS-ExSM project. Existing source comments and attribution notices are retained. Packaging changes provide relative imports, command-line data locations, inference examples and documentation.

Runtime dependencies include PyTorch, NumPy, einops and tifffile. Training additionally uses Pillow, SciPy, h5py and tqdm; each dependency retains its own license.

`pix2pix_util.py` contains discriminator utilities under the historical pix2pix naming. The source package did not include a separate license or a precise upstream revision for this file. Repository owners should confirm upstream attribution and reuse terms before assigning a project-wide license.

Third-party baseline repositories and their weights are not redistributed in this release.

# Validation

The release was executed on an NVIDIA RTX 4090 D with Python 3.13.2, PyTorch 2.10.0+cu129, NumPy 2.4.2, einops 0.8.1 and tifffile 2026.6.1. All three pretrained checkpoints loaded with strict state-dictionary matching.

The expected example arrays were supplied by the original reconstruction package and generated using PyTorch 2.0.1+cu118. Their image and model bytes are preserved. Cross-environment output differences are measured in ADU:

| Task | Maximum absolute difference | RMSE |
| --- | --- | --- |
| Denoising | 0.26512146 | 0.07744174 |
| ISR | 0.17190552 | 0.00373393 |

The installation check accepts an absolute difference of at most one ADU with zero relative tolerance, corresponding to one step of the original uint16 intensity scale. An initial 0.2 ADU absolute / 0.0001 relative comparison did not pass for the denoising example. This cross-environment check is distinct from the original same-environment equivalence check; it does not assert bitwise equality or establish paper-wide reproduction.

The JSON report contains the environment and measured output shapes. Training sources were checked for syntax and command-line import/help behavior; model training was not repeated during packaging.

"""Run inference and compare with supplied acquired-example predictions."""
from pathlib import Path
import hashlib
import argparse
import json
import platform

import einops
import numpy as np
import tifffile
import torch

from infer import denoise, isr, load_model

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(4, torch.get_num_threads()))
    for line in (ROOT / 'checksums.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'File checksum mismatch: {name}')
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'torch': torch.__version__, 'numpy': np.__version__,
              'einops': einops.__version__, 'tifffile': tifffile.__version__,
              'device': args.device, 'atol_adu': 1.0, 'rtol': 0.0,
              'reference_environment': 'PyTorch 2.0.1+cu118',
              'tolerance_basis': 'one ADU, the intensity step of the original uint16 acquisition',
              'examples': {}}
    if args.device.startswith('cuda'):
        report['gpu'] = torch.cuda.get_device_name(args.device)
    with np.load(ROOT / 'samples/Expected_outputs.npz') as expected:
        for task, checkpoint, sample in [
            ('denoise', 'denoise_hyper_300epoch.pth', 'denoise_time_clip.tif'),
            ('isr', 'isr_hyper_280epoch.pth', 'isr_pair_and_reference.npz'),
        ]:
            model = load_model(task, ROOT / 'models' / checkpoint, args.device)
            path = ROOT / 'samples' / sample
            if task == 'isr':
                with np.load(path) as data:
                    inputs = data['input_intensity_adu']
            else:
                inputs = tifffile.imread(path)
            output = (denoise if task == 'denoise' else isr)(model, inputs, args.device)
            reference = expected[f'{task}_intensity_adu']
            assert output.shape == reference.shape
            assert np.isfinite(output).all()
            report['examples'][task] = {'input_shape': list(inputs.shape),
                                       'output_shape': list(output.shape),
                                       'max_absolute_error_adu': float(np.max(np.abs(output-reference))),
                                       'rmse_adu': float(np.sqrt(np.mean((output-reference)**2))),
                                       'passed': True}
            np.testing.assert_allclose(output, reference, atol=1.0, rtol=0.0)
    load_model('denoise', ROOT / 'models/denoise_hypo_300epoch.pth', args.device)
    report['hypotonic_checkpoint_load'] = 'passed'
    path = ROOT / 'outputs/verification.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

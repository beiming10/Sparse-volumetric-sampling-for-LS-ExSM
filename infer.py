"""Portable inference for the existing LS-ExSM denoising and ISR checkpoints."""
from pathlib import Path
import argparse, json
import numpy as np
import torch
import tifffile
from networks.denoise import UNet_Attention_Hybrid
from networks.architecture.FSR import FSR

ROOT = Path(__file__).resolve().parent

def load_model(task, checkpoint, device):
    model = UNet_Attention_Hybrid(1, 1) if task == 'denoise' else FSR()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for key in ['model', 'state_dict', 'model_state_dict']:
        if isinstance(ck, dict) and key in ck:
            ck = ck[key]
            break
    state = {k.removeprefix('module.'): v for k, v in ck.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

def denoise(model, stack, device, batch_size=1):
    """Input time,y,x; five reflected time frames, return each centre frame."""
    x = np.asarray(stack, dtype=np.float32)
    if x.ndim != 3 or len(x) < 2:
        raise ValueError('Denoising requires a time,y,x stack with at least two frames')
    h, w = x.shape[-2:]
    if h % 8 or w % 8:
        raise ValueError('Denoising height and width must be divisible by 8')
    padded = np.pad(x / 250, ((2,2),(0,0),(0,0)), mode='reflect')
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            clips = np.stack([padded[i:i+5] for i in range(start, min(start+batch_size, len(x)))])
            pred = model(torch.from_numpy(clips[:,None]).to(device))[:,0,2]
            out.extend(pred.cpu().numpy() * 250)
    return np.asarray(out, dtype=np.float32)

def isr(model, stack, device, batch_size=1):
    """Input z,y,x or channel,z,y,x; process spectral channels independently."""
    x = np.asarray(stack, dtype=np.float32)
    single = x.ndim == 3
    if single: x = x[None]
    if x.ndim != 4 or x.shape[1] < 2:
        raise ValueError('ISR requires z,y,x or channel,z,y,x and at least two planes')
    c, n, h, w = x.shape
    result = np.empty((c, 3*n-2, h, w), dtype=np.float32)
    with torch.inference_mode():
        for ch in range(c):
            scaled = (x[ch]-110) / 250
            sequences = []
            for start in range(0, n-1, batch_size):
                pair = np.stack([scaled[i:i+2] for i in range(start, min(start+batch_size, n-1))])
                sequences.extend(model(torch.from_numpy(pair).to(device)).cpu().numpy())
            result[ch,:4] = sequences[0]
            for i, prediction in enumerate(sequences[1:], 1):
                result[ch,3*i] = (result[ch,3*i] + prediction[0]) / 2
                result[ch,3*i+1:3*i+4] = prediction[1:]
    result = result*250+110
    return result[0] if single else result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('task', choices=['denoise','isr'])
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=1)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('--batch-size must be a positive integer')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.input.suffix == '.npz':
        with np.load(args.input) as z: data = z['input_intensity_adu']
    else: data = tifffile.imread(args.input)
    model = load_model(args.task, args.checkpoint, args.device)
    result = (denoise if args.task == 'denoise' else isr)(model, data, args.device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Float output retains pre-quantization values for quantitative evaluation.
    np.savez_compressed(args.output, intensity_adu=result)
    tifffile.imwrite(args.output.with_suffix('.tif'), np.clip(result,0,65535).astype(np.uint16), photometric='minisblack')
    args.output.with_suffix('.json').write_text(json.dumps(dict(task=args.task, axes='time,y,x' if args.task=='denoise' else ('z,y,x' if result.ndim==3 else 'channel,z,y,x'), shape=list(result.shape), checkpoint=str(args.checkpoint), normalization='x/250' if args.task=='denoise' else '(x-110)/250', floating_point_output='NPZ intensity_adu', tiff_output='clipped [0,65535], truncated uint16'), indent=2))
    print(json.dumps(dict(output=str(args.output), shape=list(result.shape))))

if __name__ == '__main__': main()

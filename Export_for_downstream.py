"""Export matched eight-channel ISR intensities for independent downstream analysis."""
from pathlib import Path
import argparse,csv,json,re
import numpy as np,torch,tifffile
from infer import load_model,isr
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-folder',type=Path,required=True);p.add_argument('--volume',type=int,required=True);p.add_argument('--output-folder',type=Path,required=True);p.add_argument('--input-kind',choices=['dense-reference','acquired-sparse'],default='dense-reference');p.add_argument('--checkpoint',type=Path,default=ROOT/'models/isr_hyper_280epoch.pth');p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu');args=p.parse_args()
 if args.output_folder.resolve()==args.input_folder.resolve():raise ValueError('Use a separate output directory')
 files={}
 for path in args.input_folder.glob('*.tif'):
  m=re.search(r'_spectrum(\d+)_vol(\d+)\.tif$',path.name,re.I)
  if m and int(m[2])==args.volume:
   if int(m[1]) in files:raise ValueError('Multiple input files for the same channel')
   files[int(m[1])]=path
 if set(files)!=set(range(1,9)):raise ValueError('Eight channel files are required for the selected volume')
 reference=np.stack([tifffile.imread(files[c]) for c in range(1,9)])
 if args.input_kind=='dense-reference':
  n=(reference.shape[1]-1)//3*3+1;reference=reference[:,:n];indices=np.arange(0,n,3);inputs=reference[:,indices];coordinates=np.arange(n,dtype=float)
 else:
  inputs=reference;indices=np.arange(reference.shape[1]);coordinates=np.arange(3*len(indices)-2,dtype=float)/3
 model=load_model('isr',args.checkpoint,args.device);prediction=isr(model,inputs,args.device)
 args.output_folder.mkdir(parents=True,exist_ok=True)
 sources=[]
 for c in range(1,9):
  tifffile.imwrite(args.output_folder/f'Input_spectrum{c}_vol{args.volume:03d}.tif',inputs[c-1],photometric='minisblack',compression='deflate')
  tifffile.imwrite(args.output_folder/f'ISR_spectrum{c}_vol{args.volume:03d}.tif',prediction[c-1].astype(np.float32),photometric='minisblack',compression='deflate')
  if args.input_kind=='dense-reference':tifffile.imwrite(args.output_folder/f'Reference_spectrum{c}_vol{args.volume:03d}.tif',reference[c-1],photometric='minisblack',compression='deflate')
  sources.append(dict(spectrum=c,source=str(files[c]),wavelength_nm=None))
 with (args.output_folder/'Axial_positions.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['output_plane','reference_coordinate','input_position','output_value_role']);writer.writerows((z,float(coord),z%3==0,'model endpoint prediction' if z%3==0 else 'model intermediate prediction') for z,coord in enumerate(coordinates))
 metadata=dict(input_kind=args.input_kind,volume=args.volume,reference_definition='existing densely sampled denoised intensities' if args.input_kind=='dense-reference' else 'acquired sparse intensity inputs',sources=sources,input_reference_indices=indices.tolist(),output_axes='z,y,x per spectral channel',output_shape=list(prediction.shape),intensity_unit='ADU',normalization='ISR (x-110)/250, restored to ADU',prediction_dtype='float32',endpoint_handling='original model predicts endpoints; shared endpoint predictions are averaged',physical_pixel_pitch_um=None,physical_z_step_um=None,checkpoint=str(args.checkpoint))
 (args.output_folder/'Source_and_axes.json').write_text(json.dumps(metadata,indent=2))
 print(json.dumps(dict(output_folder=str(args.output_folder),channels=8,output_planes=prediction.shape[1],bytes=sum(p.stat().st_size for p in args.output_folder.iterdir() if p.is_file()))),flush=True)
if __name__=='__main__':main()

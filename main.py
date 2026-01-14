import argparse
import yaml
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
from tqdm import tqdm
from functools import partial

# DPS imports
from guided_diffusion.measurements import register_operator, LinearOperator, get_noise, get_operator
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.img_utils import clear_color

# Ultrasound utilities
from ultrasound_utils import (
    load_psf, load_ultrasound_image, rf2bmode, 
    CalcPerf, calc_CNR, normalize_image
)


@register_operator(name='ultrasound_psf')
class UltrasoundPSFOperator(LinearOperator):
    def __init__(self, psf_path, device):
        self.device = device
        psf = load_psf(psf_path, normalize=True, verbose=False)
        h, w = psf.shape
        self.pad = (h//2, w//2)
        self.kernel = torch.from_numpy(psf).float().view(1,1,h,w).to(device)
    
    def forward(self, x, **kw):
        out = []
        for c in range(x.shape[1]):
            xp = F.pad(x[:,c:c+1], (self.pad[1],)*2+(self.pad[0],)*2, mode='reflect')
            out.append(F.conv2d(xp, self.kernel, padding=0))
        return torch.cat(out, dim=1)
    
    def transpose(self, x, **kw):
        return self.forward(x)


def find_datasets(base_path):
    """Find all ultrasound datasets"""
    datasets = []
    
    # Simulated data
    simu_path = Path(base_path) / 'simu'
    if simu_path.exists():
        for folder in ['1', '2', '3', '4']:
            folder_path = simu_path / folder
            if folder_path.exists():
                datasets.append({
                    'type': 'simu',
                    'name': f'simu_{folder}',
                    'path': folder_path,
                    'has_gt': True
                })
    
    # In vivo data
    vivo_path = Path(base_path) / 'vivo'
    if vivo_path.exists():
        for folder in vivo_path.iterdir():
            if folder.is_dir():
                datasets.append({
                    'type': 'vivo',
                    'name': f'vivo_{folder.name}',
                    'path': folder,
                    'has_gt': False
                })
    
    return datasets


def load_image(dataset_path, image_type='bmode'):
    """Load ultrasound image (try .png first, then .mat)"""
    for ext in ['.png', '.mat']:
        img_path = dataset_path / f'{image_type}{ext}'
        if img_path.exists():
            if ext == '.png':
                img = np.array(Image.open(img_path).convert('L'))
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            else:
                img = load_ultrasound_image(str(img_path), verbose=False)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return img
    return None


def compute_cnr_auto(bmode_img):
    """
    Compute CNR with automatic ROI selection
    Signal ROI: center region
    Background ROI: corner regions
    """
    h, w = bmode_img.shape
    
    # Signal ROI (center third)
    signal_roi = bmode_img[h//3:2*h//3, w//3:2*w//3]
    
    # Background ROI (four corners)
    corner_size = min(h, w) // 4
    background_roi = np.concatenate([
        bmode_img[:corner_size, :corner_size].flatten(),
        bmode_img[:corner_size, -corner_size:].flatten(),
        bmode_img[-corner_size:, :corner_size].flatten(),
        bmode_img[-corner_size:, -corner_size:].flatten()
    ])
    
    result = calc_CNR(signal_roi.flatten(), background_roi)
    return result


def process_dataset(dataset, model, sampler, noiser, cond, device, save_dir):
    print(f"\n{'='*60}")
    print(f"Processing: {dataset['name']}")
    print(f"{'='*60}")
    
    results = {
        'name': dataset['name'],
        'type': dataset['type']
    }
    
    # Create output directory
    out_dir = Path(save_dir) / dataset['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PSF
    psf_path = dataset['path'] / 'psf_est.mat'
    if not psf_path.exists():
        psf_path = dataset['path'] / 'psf_GT.mat'
    
    if not psf_path.exists():
        print(f"[WARNNING] No PSF found, skipping...")
        return None
    
    print(f"[INFO] PSF: {psf_path.name}")
    
    # Update operator with this PSF
    operator = UltrasoundPSFOperator(str(psf_path), device)
    cond.operator = operator
    
    # Load observed image
    print(f"[INFO] Loading image...")
    img = load_image(dataset['path'], 'bmode')
    if img is None:
        img = load_image(dataset['path'], 'rf')
        if img is not None:
            print(f"[INFO] Converting RF to B-mode...")
            img = rf2bmode(img) / 255.0
    
    if img is None:
        print(f"[WARNING] No image found, skipping...")
        return None
    
    # Resize to 256x256 (image from the paper)
    img = np.array(Image.fromarray((img*255).astype(np.uint8)).resize((256, 256)))
    img = img / 255.0
    
    # Convert to torch tensor
    x = torch.from_numpy(img).float().view(1,1,256,256).to(device)
    x = x * 2 - 1  # Normalize to [-1, 1]
    x = x.repeat(1, 3, 1, 1)  # RGB
    
    # Forward model & add noise
    y = noiser(operator.forward(x))
    
    # Save input
    y_np = clear_color(y)
    Image.fromarray((y_np*255).astype(np.uint8)).save(out_dir / 'input.png')
    
    # Run DPS reconstruction
    print(f"[INFO] Running DPS reconstruction...")
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=cond.conditioning)
    sample = sample_fn(
        x_start=torch.randn_like(x).requires_grad_(),
        measurement=y,
        record=False,
        save_root=str(out_dir)
    )
    
    # Save reconstruction
    recon_np = clear_color(sample)
    Image.fromarray((recon_np*255).astype(np.uint8)).save(out_dir / 'reconstruction.png')
    
    # Convert to grayscale for metrics (if RGB)
    if recon_np.ndim == 3:
        recon_gray = np.mean(recon_np, axis=2)
    else:
        recon_gray = recon_np
    
    # Compute metrics
    if dataset['has_gt']:
        print(f"[INFO] Computing metrics with ground truth...")
        
        # Load ground truth
        gt = load_image(dataset['path'], 'bmode_GT')
        if gt is None:
            gt_rf = load_image(dataset['path'], 'rf_GT')
            if gt_rf is not None:
                gt = rf2bmode(gt_rf) / 255.0
        
        if gt is not None:
            gt = np.array(Image.fromarray((gt*255).astype(np.uint8)).resize((256, 256))) / 255.0
            Image.fromarray((gt*255).astype(np.uint8)).save(out_dir / 'ground_truth.png')
            
            # CalcPerf metrics (use grayscale)
            metrics = CalcPerf(gt * 255, recon_gray * 255)
            results.update({
                'MSE': metrics['MSE'],
                'PSNR': metrics['PSNR'],
                'SSIM': None,
                'RMSE': metrics['RMSE'],
                'NRMSE': metrics['NRMSE'],
                'MAPE': metrics['MAPE'],
                'Rvalue': metrics['Rvalue']
            })
            
            # Add SSIM
            try:
                from skimage.metrics import structural_similarity as ssim
                results['SSIM'] = ssim(gt, recon_gray, data_range=1.0)
            except ImportError:
                pass
            
            print(f"PSNR: {metrics['PSNR']:.2f} dB")
            print(f"MSE: {metrics['MSE']:.6f}")
            print(f"SSIM: {results['SSIM']:.4f}" if results['SSIM'] else "")
            
            # Also compute CNR on reconstruction
            recon_bmode = (recon_gray * 255).astype(np.uint8)
            cnr_result = compute_cnr_auto(recon_bmode)
            results['CNR_recon'] = cnr_result['CNR']
    
    else:
        print(f"  📊 Computing CNR (in vivo data)...")
        
        # Compute CNR for in vivo data (use grayscale)
        recon_bmode = (recon_gray * 255).astype(np.uint8)
        cnr_result = compute_cnr_auto(recon_bmode)
        results.update({
            'CNR': cnr_result['CNR'],
            'CNR_max': cnr_result['CNR_max'],
            'mu_signal': cnr_result['mu_1'],
            'mu_background': cnr_result['mu_2'],
            'var_signal': cnr_result['var_1'],
            'var_background': cnr_result['var_2']
        })
        print(f"CNR: {cnr_result['CNR']:.2f} dB")
    
    # Save metrics to text file
    with open(out_dir / 'metrics.txt', 'w') as f:
        f.write(f"Dataset: {dataset['name']}\n")
        f.write(f"Type: {dataset['type']}\n")
        f.write(f"{'-'*40}\n")
        for key, val in results.items():
            if val is not None and key not in ['name', 'type']:
                f.write(f'{key}: {val}\n')
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Process all ultrasound data with DPS')
    parser.add_argument('--data_root', default='./data', help='Root directory containing simu/ and vivo/')
    parser.add_argument('--model_config', required=True)
    parser.add_argument('--diffusion_config', required=True)
    parser.add_argument('--task_config', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', default='./results_all')
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configs
    print("\n[INFO] Loading configurations...")
    model_cfg = yaml.safe_load(open(args.model_config))
    diff_cfg = yaml.safe_load(open(args.diffusion_config))
    task_cfg = yaml.safe_load(open(args.task_config))
    
    # Setup model
    print("[INFO] Loading model...")
    model = create_model(**model_cfg).to(device).eval()
    sampler = create_sampler(**diff_cfg)
    noiser = get_noise(**task_cfg['measurement']['noise'])
    
    
    from scipy.io import savemat
    dummy_psf = np.ones((31, 31), dtype=np.float64)
    dummy_psf = dummy_psf / dummy_psf.sum()
    dummy_psf_path = Path(args.save_dir) / 'dummy_psf.mat'
    dummy_psf_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(dummy_psf_path), {'psf': dummy_psf})
    
    operator = UltrasoundPSFOperator(str(dummy_psf_path), device)
    cond = get_conditioning_method(
        task_cfg['conditioning']['method'], 
        operator, 
        noiser, 
        **task_cfg['conditioning']['params']
    )
    
    # Find all datasets
    print(f"\nSearching for datasets in {args.data_root}...")
    datasets = find_datasets(args.data_root)
    
    if not datasets:
        print(f"No datasets found in {args.data_root}")
        return
    
    print(f"\nFound {len(datasets)} datasets:")
    for ds in datasets:
        print(f"  • {ds['name']} ({ds['type']}) - GT: {ds['has_gt']}")
    
    # Process all datasets
    print(f"\n{'='*60}")
    print("Starting processing...")
    print(f"{'='*60}")
    
    all_results = []
    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}]", end=" ")
        result = process_dataset(dataset, model, sampler, noiser, cond, device, args.save_dir)
        if result:
            all_results.append(result)
    
    # Save summary
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = Path(args.save_dir) / 'summary.csv'
        df.to_csv(csv_path, index=False)
        
        # Print statistics
        if 'PSNR' in df.columns:
            simu_df = df[df['type'] == 'simu']
            if not simu_df.empty:
                print(f"\n Simulated data statistics:")
                print(f"  Average PSNR: {simu_df['PSNR'].mean():.2f} ± {simu_df['PSNR'].std():.2f} dB")
                print(f"  Average MSE: {simu_df['MSE'].mean():.6f} ± {simu_df['MSE'].std():.6f}")
                if 'SSIM' in simu_df.columns:
                    print(f"  Average SSIM: {simu_df['SSIM'].mean():.4f} ± {simu_df['SSIM'].std():.4f}")
        
        if 'CNR' in df.columns:
            vivo_df = df[df['type'] == 'vivo']
            if not vivo_df.empty:
                print(f"\nIn vivo data statistics:")
                print(f"  Average CNR: {vivo_df['CNR'].mean():.2f} ± {vivo_df['CNR'].std():.2f} dB")
    else:
        print("\n[WARNNING] No results to save")


if __name__ == '__main__':
    main()
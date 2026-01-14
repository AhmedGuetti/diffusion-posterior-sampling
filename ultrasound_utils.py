"""
Ultrasound Processing Utilities
================================
Python equivalents of MATLAB functions for ultrasound image processing.

Functions:
    - rf2bmode: Convert RF image to B-mode
    - calc_CNR: Calculate Contrast-to-Noise Ratio
    - CalcPerf: Calculate performance metrics (MSE, PSNR, RMSE, etc.)
    - load_psf: Load PSF from .mat file
    - load_ultrasound_data: Load ultrasound images from various formats

Author: Adapted from MATLAB code by Renaud Morin (morin@irit.fr)
"""

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import hilbert
from PIL import Image
import os


# ============================================================
# PSF LOADING
# ============================================================

def load_psf(psf_path, normalize=True, verbose=True):
    """
    Load PSF (Point Spread Function) from a .mat file.
    
    Parameters
    ----------
    psf_path : str
        Path to the .mat file containing the PSF
    normalize : bool, optional
        If True, normalize PSF to sum to 1 (default: True)
    verbose : bool, optional
        If True, print PSF information (default: True)
    
    Returns
    -------
    psf : ndarray
        The PSF as a 2D numpy array
    
    Example
    -------
    >>> psf = load_psf('data/simu/1/psf_est.mat')
    >>> print(psf.shape)
    (31, 31)
    """
    if not os.path.exists(psf_path):
        raise FileNotFoundError(f"PSF file not found: {psf_path}")
    
    # Load .mat file
    mat_data = loadmat(psf_path)
    
    # Common variable names for PSF in .mat files
    possible_names = ['psf', 'PSF', 'psf_GT', 'psf_est', 'h', 'kernel', 'H']
    
    psf = None
    var_name = None
    
    for name in possible_names:
        if name in mat_data:
            psf = mat_data[name]
            var_name = name
            break
    
    # If not found with common names, get first non-meta variable
    if psf is None:
        for key, value in mat_data.items():
            if not key.startswith('__'):
                psf = value
                var_name = key
                break
    
    if psf is None:
        raise ValueError(f"Could not find PSF data in {psf_path}")
    
    # Convert to float64 numpy array
    psf = np.array(psf, dtype=np.float64)
    
    # Handle 3D arrays (take first slice)
    if psf.ndim == 3:
        psf = psf[:, :, 0]
    
    # Squeeze any singleton dimensions
    psf = np.squeeze(psf)
    
    if verbose:
        print(f"Loaded PSF from: {psf_path}")
        print(f"  Variable name: {var_name}")
        print(f"  Shape: {psf.shape}")
        print(f"  Min: {psf.min():.6f}, Max: {psf.max():.6f}")
    
    # Normalize
    if normalize:
        psf = psf / np.sum(psf)
        if verbose:
            print(f"  Normalized: sum = {np.sum(psf):.6f}")
    
    return psf


def load_ultrasound_image(image_path, verbose=True):
    """
    Load ultrasound image from .mat or image file (.png, .jpg, etc.)
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    verbose : bool, optional
        If True, print image information
    
    Returns
    -------
    image : ndarray
        The image as a 2D numpy array (float64, range depends on original)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if image_path.endswith('.mat'):
        mat_data = loadmat(image_path)
        
        # Common variable names for images
        possible_names = ['rf', 'RF', 'bmode', 'Bmode', 'image', 'img', 'data']
        
        image = None
        var_name = None
        
        for name in possible_names:
            if name in mat_data:
                image = mat_data[name]
                var_name = name
                break
        
        if image is None:
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.ndim >= 2:
                        image = value
                        var_name = key
                        break
        
        if image is None:
            raise ValueError(f"Could not find image data in {image_path}")
        
        image = np.array(image, dtype=np.float64)
        
    else:
        # Load as image file
        img = Image.open(image_path)
        
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        image = np.array(img, dtype=np.float64)
        var_name = "image"
    
    # Squeeze singleton dimensions
    image = np.squeeze(image)
    
    if verbose:
        print(f"Loaded image from: {image_path}")
        print(f"  Shape: {image.shape}")
        print(f"  Dtype: {image.dtype}")
        print(f"  Range: [{image.min():.2f}, {image.max():.2f}]")
    
    return image


# ============================================================
# RF TO B-MODE CONVERSION
# ============================================================

def rf2bmode(rf, increase=0):
    """
    Convert RF (Radio Frequency) image to B-mode image.
    
    Equivalent to MATLAB rf2bmode.m by Renaud Morin.
    
    Process:
    1. Apply Hilbert transform to get analytic signal
    2. Take absolute value (envelope detection)
    3. Apply log compression (20 * log10)
    4. Normalize to [0, 255] range
    
    Parameters
    ----------
    rf : ndarray
        RF image (2D or 3D array)
    increase : float, optional
        Increase factor to adjust contrast (added before log)
        Default: 0
    
    Returns
    -------
    bmode : ndarray
        B-mode image, values in range [0, 255]
    
    Example
    -------
    >>> rf = load_ultrasound_image('data/simu/1/rf.mat')
    >>> bmode = rf2bmode(rf)
    >>> plt.imshow(bmode, cmap='gray')
    """
    rf = np.array(rf, dtype=np.float64)
    
    # Handle 3D arrays (multiple frames)
    if rf.ndim == 3:
        bmode = np.zeros_like(rf)
        for i in range(rf.shape[2]):
            bmode[:, :, i] = rf2bmode(rf[:, :, i], increase)
        return bmode
    
    # Apply Hilbert transform along first axis (axial direction)
    # hilbert() returns the analytic signal
    analytic_signal = hilbert(rf, axis=0)
    
    # Envelope detection (absolute value of analytic signal)
    envelope = np.abs(analytic_signal) # type: ignore
    
    # Add increase factor and apply log compression
    # Using natural log like MATLAB, then multiply by 20
    # Note: MATLAB uses log() which is natural log, equivalent to np.log()
    bmode_temp = 20 * np.log(envelope + increase + 1e-10)  # Small epsilon to avoid log(0)
    
    # Normalize: subtract minimum
    bmode_temp = bmode_temp - np.min(bmode_temp)
    
    # Scale to [0, 255]
    max_val = np.max(bmode_temp)
    if max_val > 0:
        bmode = 255.0 * bmode_temp / max_val
    else:
        bmode = bmode_temp
    
    return bmode


# ============================================================
# CNR CALCULATION
# ============================================================

def calc_CNR(R1, R2):
    """
    Calculate Contrast-to-Noise Ratio (CNR) between two regions.
    
    Equivalent to MATLAB calc_CNR.m by Renaud Morin.
    
    CNR (in dB) = 20 * log10( |μ1 - μ2| / sqrt(σ1² + σ2²) )
    
    Parameters
    ----------
    R1 : ndarray
        First region of interest (e.g., lesion/target)
    R2 : ndarray
        Second region of interest (e.g., background)
    
    Returns
    -------
    dict with keys:
        'CNR' : float
            Contrast-to-noise ratio in dB
        'CNR_max' : float
            Maximum CNR (|μ1| / sqrt(σ2²))
        'mu_1' : float
            Mean of region 1
        'mu_2' : float
            Mean of region 2
        'var_1' : float
            Variance of region 1
        'var_2' : float
            Variance of region 2
    
    Example
    -------
    >>> # Select regions from image
    >>> R1 = image[50:100, 50:100]  # Signal region
    >>> R2 = image[150:200, 150:200]  # Background region
    >>> results = calc_CNR(R1, R2)
    >>> print(f"CNR = {results['CNR']:.2f} dB")
    """
    R1 = np.array(R1, dtype=np.float64).flatten()
    R2 = np.array(R2, dtype=np.float64).flatten()
    
    # Mean of regions
    mu_1 = np.mean(R1)
    mu_2 = np.mean(R2)
    
    # Variance of regions (using ddof=0 for population variance like MATLAB)
    var_1 = np.var(R1, ddof=0)
    var_2 = np.var(R2, ddof=0)
    
    # Compute CNR
    denominator = np.sqrt(var_1 + var_2)
    
    if denominator > 0:
        cnr_linear = np.abs(mu_1 - mu_2) / denominator
        CNR = 20 * np.log10(cnr_linear + 1e-10)  # Avoid log(0)
    else:
        CNR = 0.0
    
    # CNR max
    if var_2 > 0:
        CNR_max = np.abs(mu_1) / np.sqrt(var_2)
    else:
        CNR_max = 0.0
    
    return {
        'CNR': CNR,
        'CNR_max': CNR_max,
        'mu_1': mu_1,
        'mu_2': mu_2,
        'var_1': var_1,
        'var_2': var_2
    }


def calc_CNR_from_masks(image, mask_signal, mask_background):
    """
    Calculate CNR using binary masks to define regions.
    
    Parameters
    ----------
    image : ndarray
        The image to analyze
    mask_signal : ndarray
        Binary mask for signal region (same shape as image)
    mask_background : ndarray
        Binary mask for background region (same shape as image)
    
    Returns
    -------
    dict
        Same as calc_CNR()
    """
    R1 = image[mask_signal > 0]
    R2 = image[mask_background > 0]
    
    return calc_CNR(R1, R2)


# ============================================================
# PERFORMANCE METRICS (CalcPerf)
# ============================================================

def CalcPerf(Reference, Test):
    """
    Calculate performance metrics between reference and test images.
    
    Equivalent to MATLAB CalcPerf.m by Abbas Manthiri S.
    
    Parameters
    ----------
    Reference : ndarray
        Reference/ground truth image
    Test : ndarray
        Test/reconstructed image
    
    Returns
    -------
    dict with keys:
        'MSE' : float
            Mean Squared Error
        'PSNR' : float
            Peak Signal-to-Noise Ratio (dB)
        'Rvalue' : float
            R value (coefficient of determination-like metric)
        'RMSE' : float
            Root Mean Square Error
        'NRMSE' : float
            Normalized RMSE
        'MAPE' : float
            Mean Absolute Percentage Error (%)
    
    Example
    -------
    >>> gt = load_ultrasound_image('bmode_GT.png')
    >>> recon = load_ultrasound_image('bmode_reconstructed.png')
    >>> metrics = CalcPerf(gt, recon)
    >>> print(f"PSNR = {metrics['PSNR']:.2f} dB")
    >>> print(f"MSE = {metrics['MSE']:.6f}")
    """
    Reference = np.array(Reference, dtype=np.float64)
    Test = np.array(Test, dtype=np.float64)
    
    # Check dimensions
    if Reference.shape != Test.shape:
        raise ValueError(f"Input must have same dimensions. "
                        f"Reference: {Reference.shape}, Test: {Test.shape}")
    
    # Flatten for calculations
    ref_flat = Reference.flatten()
    test_flat = Test.flatten()
    
    # MSE - Mean Squared Error
    MSE = np.mean((Reference - Test) ** 2)
    
    # PSNR - Peak Signal-to-Noise Ratio
    # Determine max intensity based on data range
    max_ref = np.max(Reference)
    if max_ref > 1:
        maxI = 255  # Assume 8-bit range
    else:
        maxI = 1    # Assume normalized [0, 1] range
    
    if MSE > 0:
        PSNR = 10 * np.log10(maxI**2 / MSE)
    else:
        PSNR = float('inf')
    
    # R Value (similar to coefficient of determination)
    ss_res = np.sum((Test - Reference) ** 2)
    ss_tot = np.sum(Reference ** 2)
    
    if ss_tot > 0:
        Rvalue = 1 - np.abs(ss_res / ss_tot)
    else:
        Rvalue = 0.0
    
    # RMSE - Root Mean Square Error
    RMSE = np.sqrt(MSE)
    
    # NRMSE - Normalized RMSE
    data_range = np.max(Reference) - np.min(Reference)
    if data_range > 0:
        NRMSE = RMSE / data_range
    else:
        NRMSE = 0.0
    
    # MAPE - Mean Absolute Percentage Error
    # Avoid division by zero
    ref_nonzero = Reference.copy()
    ref_nonzero[ref_nonzero == 0] = 1e-10
    MAPE = np.mean(np.abs((Test - Reference) / ref_nonzero)) * 100
    
    return {
        'MSE': MSE,
        'PSNR': PSNR,
        'Rvalue': Rvalue,
        'RMSE': RMSE,
        'NRMSE': NRMSE,
        'MAPE': MAPE
    }


def CalcPerf_extended(Reference, Test):
    """
    Extended performance metrics including SSIM.
    
    Parameters
    ----------
    Reference : ndarray
        Reference/ground truth image
    Test : ndarray
        Test/reconstructed image
    
    Returns
    -------
    dict
        All metrics from CalcPerf() plus:
        'SSIM' : float
            Structural Similarity Index
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Get basic metrics
    metrics = CalcPerf(Reference, Test)
    
    # Normalize images for SSIM
    ref_norm = (Reference - Reference.min()) / (Reference.max() - Reference.min() + 1e-10)
    test_norm = (Test - Test.min()) / (Test.max() - Test.min() + 1e-10)
    
    # Calculate SSIM
    metrics['SSIM'] = ssim(ref_norm, test_norm, data_range=1.0)
    
    return metrics


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_image(image, target_range='01'):
    """
    Normalize image to specified range.
    
    Parameters
    ----------
    image : ndarray
        Input image
    target_range : str
        '01' for [0, 1] range
        '-11' for [-1, 1] range
        '255' for [0, 255] range
    
    Returns
    -------
    ndarray
        Normalized image
    """
    image = np.array(image, dtype=np.float64)
    
    img_min = image.min()
    img_max = image.max()
    
    # Normalize to [0, 1] first
    if img_max - img_min > 0:
        img_norm = (image - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(image)
    
    if target_range == '01':
        return img_norm
    elif target_range == '-11':
        return img_norm * 2 - 1
    elif target_range == '255':
        return img_norm * 255
    else:
        raise ValueError(f"Unknown target_range: {target_range}")


def create_circular_roi(shape, center, radius):
    """
    Create a circular ROI mask.
    
    Parameters
    ----------
    shape : tuple
        Shape of the output mask (height, width)
    center : tuple
        Center of the circle (x, y)
    radius : float
        Radius of the circle
    
    Returns
    -------
    ndarray
        Binary mask (1 inside circle, 0 outside)
    """
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cx, cy = center
    
    mask = ((x - cx)**2 + (y - cy)**2) <= radius**2
    
    return mask.astype(np.float64)


def create_rectangular_roi(shape, top_left, bottom_right):
    """
    Create a rectangular ROI mask.
    
    Parameters
    ----------
    shape : tuple
        Shape of the output mask (height, width)
    top_left : tuple
        Top-left corner (x, y)
    bottom_right : tuple
        Bottom-right corner (x, y)
    
    Returns
    -------
    ndarray
        Binary mask
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float64)
    
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    mask[y1:y2, x1:x2] = 1
    
    return mask


# ============================================================
# MAIN - DEMONSTRATION
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Ultrasound Processing Utilities - Demo")
    print("=" * 60)
    
    # Create synthetic test data
    print("\n1. Creating synthetic test data...")
    np.random.seed(42)
    
    # Synthetic RF signal
    t = np.linspace(0, 1, 256)
    x = np.linspace(0, 1, 128)
    T, X = np.meshgrid(t, x)
    
    # RF signal with some structure
    rf_synthetic = np.sin(2 * np.pi * 10 * T) * np.exp(-((X - 0.5)**2) / 0.1)
    rf_synthetic += 0.3 * np.random.randn(*rf_synthetic.shape)
    
    print(f"   RF shape: {rf_synthetic.shape}")
    
    # 2. Test rf2bmode
    print("\n2. Testing rf2bmode...")
    bmode = rf2bmode(rf_synthetic)
    print(f"   B-mode shape: {bmode.shape}")
    print(f"   B-mode range: [{bmode.min():.1f}, {bmode.max():.1f}]")
    
    # 3. Test CalcPerf
    print("\n3. Testing CalcPerf...")
    
    # Create reference and degraded test image
    reference = np.random.rand(100, 100) * 255
    test = reference + np.random.randn(100, 100) * 10  # Add noise
    test = np.clip(test, 0, 255)
    
    metrics = CalcPerf(reference, test)
    print("   Performance metrics:")
    for key, value in metrics.items():
        if key == 'PSNR':
            print(f"      {key}: {value:.2f} dB")
        else:
            print(f"      {key}: {value:.6f}")
    
    # 4. Test calc_CNR
    print("\n4. Testing calc_CNR...")
    
    # Create two regions with different statistics
    R1 = np.random.randn(50, 50) * 10 + 100  # Signal region (mean=100)
    R2 = np.random.randn(50, 50) * 15 + 50   # Background region (mean=50)
    
    cnr_results = calc_CNR(R1, R2)
    print("   CNR results:")
    print(f"      CNR: {cnr_results['CNR']:.2f} dB")
    print(f"      μ_signal: {cnr_results['mu_1']:.2f}")
    print(f"      μ_background: {cnr_results['mu_2']:.2f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
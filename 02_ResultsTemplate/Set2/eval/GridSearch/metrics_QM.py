import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
from itertools import repeat

# ==============================================================================
# The `rgb2ycbcrn` and `qm` functions from the previous answer go here.
# They are unchanged.
# ... (insert previous functions here) ...
def rgb2ycbcrn(rgb_image, n=8):
    """
    Python translation of rgb2ycbcrn.m.
    Converts an RGB image to YCbCr using the BT.709 standard for a given bit depth.
    """
    M = np.array([
        [0.212600, 0.715200, 0.072200],
        [-0.114572, -0.385428, 0.500000],
        [0.500000, -0.454153, -0.045847]
    ])
    original_shape = rgb_image.shape
    reshaped_rgb = rgb_image.reshape(-1, 3)
    ycbcr = reshaped_rgb @ M.T
    ycbcr = ycbcr.reshape(original_shape)
    ycbcr[:, :, 0] = (219 * ycbcr[:, :, 0] + 16) * (2 ** (n - 8))
    ycbcr[:, :, 1:3] = (224 * ycbcr[:, :, 1:3] + 128) * (2 ** (n - 8))
    if n == 8:
        ycbcr = ycbcr.astype(np.uint8)
    elif n in [10, 16]:
        ycbcr = np.round(ycbcr).astype(np.uint16)
    else:
        raise ValueError("Invalid bit depth. Supported values are 8, 10, 16.")
    return ycbcr

def qm(ref, rec, n_rgb, n_yuv):
    """
    Python translation of QM.m.
    Calculates image quality metrics (PSNR, SSIM) between a reference and reconstructed image.
    """
    ref_double = ref.astype(np.float64) / (2**n_rgb - 1)
    rec_double = rec.astype(np.float64) / (2**n_rgb - 1)
    ref_ycbcr = rgb2ycbcrn(ref_double, n_yuv)
    rec_ycbcr = rgb2ycbcrn(rec_double, n_yuv)
    Y1, U1, V1 = ref_ycbcr[:,:,0], ref_ycbcr[:,:,1], ref_ycbcr[:,:,2]
    Y2, U2, V2 = rec_ycbcr[:,:,0], rec_ycbcr[:,:,1], rec_ycbcr[:,:,2]
    max_val = 2**n_yuv - 1
    def calculate_psnr(img1, img2, max_val):
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0: return float('inf')
        return 10 * np.log10(max_val**2 / mse)
    y_psnr = calculate_psnr(Y1, Y2, max_val)
    u_psnr = calculate_psnr(U1, U2, max_val)
    v_psnr = calculate_psnr(V1, V2, max_val)
    yuv_psnr = (6 * y_psnr + u_psnr + v_psnr) / 8
    y1_double = Y1.astype(np.float64) / max_val
    y2_double = Y2.astype(np.float64) / max_val
    y_ssim = ssim(y1_double, y2_double, data_range=1.0)
    return y_psnr, u_psnr, v_psnr, yuv_psnr, y_ssim
# ==============================================================================


def process_view(view_coords, input1_path, input2_path):
    """
    Worker function: processes a single view to calculate quality metrics.
    This function will be run in parallel on different CPU cores.
    """
    t, s = view_coords
    filename = f'{t:03d}_{s:03d}.ppm'
    
    try:
        img1_path = os.path.join(input1_path, filename)
        img2_path = os.path.join(input2_path, filename)
        
        img1 = np.array(Image.open(img1_path))
        img2 = np.array(Image.open(img2_path))

        n_rgb_bits = 16 if img1.dtype == np.uint16 else 8
        
        # Calculate metrics using the qm function
        y_psnr, _, _, psnr_yuv, ssim_y = qm(img2, img1, n_rgb=n_rgb_bits, n_yuv=10)
        
        return (y_psnr, psnr_yuv, ssim_y)

    except FileNotFoundError:
        # Return None or default values if a file is missing
        print(f"Warning: Could not process view {filename}. File not found.")
        return None

def main_parallel():
    """
    Main script logic, modified to run the image processing loop in parallel.
    """
    input2_base_path = '/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/set2'
    
    # Dynamically find conf files to determine rates, viewsh, and viewsv
    import re
    import glob
    current_dir = os.path.dirname(os.path.abspath(__file__))
    heuristic = os.path.basename(current_dir)
    dataset_dir = os.path.dirname(os.path.dirname(current_dir))
    conf_dir = os.path.join(dataset_dir, heuristic)
    conf_files = glob.glob(os.path.join(conf_dir, '*_encode.conf'))
    
    rates = []
    viewsh = None
    viewsv = None
    
    for cf in conf_files:
        basename = os.path.basename(cf)
        m = re.search(r'_([\d\.]+)_encode\.conf', basename)
        if m:
            rates.append(float(m.group(1)))
            
    if conf_files:
        with open(conf_files[0], 'r') as f:
            conf_text = f.read()
            match_nv = re.search(r'-nv\s+(\d+)', conf_text)
            match_nh = re.search(r'-nh\s+(\d+)', conf_text)
            if match_nv:
                viewsv = int(match_nv.group(1))
            if match_nh:
                viewsh = int(match_nh.group(1))
                
    if viewsv is None or viewsh is None:
        img_files = glob.glob(os.path.join(input2_base_path, '*_*.ppm'))
        if not img_files:
            img_files = glob.glob(os.path.join(input2_base_path, '*_*.png'))
        
        max_t = -1
        max_s = -1
        for img in img_files:
            basename = os.path.basename(img)
            m_name = re.match(r'(\d+)_(\d+)\.\w+', basename)
            if m_name:
                t = int(m_name.group(1))
                s = int(m_name.group(2))
                if t > max_t: max_t = t
                if s > max_s: max_s = s
        
        if max_t >= 0 and max_s >= 0:
            viewsv = max_t + 1
            viewsh = max_s + 1
        else:
            viewsv = 13
            viewsh = 13
                
    rates = sorted(list(set(rates))) if rates else [0.1]
    
    total_views = viewsh * viewsv
    results_data = []


    # Dynamically determine image dimensions
    first_img_path = os.path.join(input2_base_path, '000_000.ppm')
    try:
        with Image.open(first_img_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Warning: Could not open {first_img_path}. Falling back to 1936x1288.")
        img_width, img_height = 1936, 1288

    for rate in rates:
        rate_str = "{:.3g}".format(rate).replace("+", "")
        input1_base_path = f'../../GridSearch/{rate}/'
        
        # --- File size calculation (same as before) ---
        try:
            comp_file_path = glob.glob(os.path.join(input1_base_path, '*.comp'))[0]
            filesize_bits = os.path.getsize(comp_file_path) * 8
            true_rate = filesize_bits / (total_views * img_width * img_height)
        except IndexError:
            print(f"Error: Compressed file for rate {rate} not found in {input1_base_path}.")
            continue
            
        print(f"Processing rate: {rate} (True rate: {true_rate:.6f} bpp)")

        # --- Parallel Processing Setup ---
        # 1. Create a list of all view coordinates to process
        view_coordinates = [(t, s) for s in range(viewsh) for t in range(viewsv)]
        
        # Use a context manager to handle the process pool
        # This will automatically use all available CPU cores
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # 2. Map the process_view function to each set of coordinates.
            # `repeat` is used to efficiently pass the same path arguments to every call.
            results = executor.map(process_view, view_coordinates, repeat(input1_base_path), repeat(input2_base_path))

        # 3. Filter out any failed tasks (e.g., file not found) and aggregate results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print(f"No views were successfully processed for rate {rate}.")
            continue

        # Unzip the list of tuples into separate lists for each metric
        all_psnr_y, all_psnr_yuv, all_ssim = zip(*valid_results)
        
        total_psnr_y = sum(all_psnr_y)
        total_psnr_yuv = sum(all_psnr_yuv)
        total_ssim = sum(all_ssim)

        # --- Calculate mean values ---
        num_processed_views = len(valid_results)
        mean_psnr_yuv = total_psnr_yuv / num_processed_views
        mean_psnr_y = total_psnr_y / num_processed_views
        mean_ssim = total_ssim / num_processed_views
        
        # Store results for this rate
        current_result = {
            'Rate': true_rate,
            'Mean_PSNR_Y': mean_psnr_y,
            'Mean_PSNR_YUV': mean_psnr_yuv,
            'Mean_SSIM': mean_ssim,
            'Target_Rate': rate
        }
        results_data.append(current_result)
        
        print(f"Results for target rate {rate}:")
        print(f"  - Mean PSNR Y: {mean_psnr_y:.4f}")
        print(f"  - Mean PSNR YUV: {mean_psnr_yuv:.4f}")
        print(f"  - Mean SSIM: {mean_ssim:.4f}")

    # --- Write results to CSV (same as before) ---
    if results_data:
        df = pd.DataFrame(results_data)
        df = df[['Rate', 'Mean_PSNR_Y', 'Mean_PSNR_YUV', 'Mean_SSIM', 'Target_Rate']]
        df.to_csv('metrics_results.csv', index=False)
        print("\nResults successfully saved to metrics_results.csv")
    else:
        print("\nNo results were generated.")


# The __name__ == "__main__" guard is ESSENTIAL for multiprocessing
if __name__ == "__main__":
    main_parallel()
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict

def read_ppm(file_path: Path) -> Tuple[np.ndarray, int]:
    """Reads a PPM file with robust header parsing."""
    with open(file_path, 'rb') as f:
        # Helper to get the next token, skipping comments
        def get_next_token(f):
            while True:
                line = f.readline()
                if not line: return None
                # Strip comments
                line = line.split(b'#')[0]
                # Split into tokens
                tokens = line.split()
                for token in tokens:
                    yield token

        token_gen = get_next_token(f)
        
        # 1. Magic number
        magic = next(token_gen).decode('ascii')
        if magic not in ['P5', 'P6']:
            raise ValueError(f"Unsupported magic number: {magic} in {file_path}")
            
        # 2. Width, Height, MaxVal
        w = int(next(token_gen))
        h = int(next(token_gen))
        max_val = int(next(token_gen))

        # 3. Read raw data after the header
        # The header ends after the max_val. We need to find the exact byte position.
        # We know the header is finished after the whitespace following max_val.
        data = np.fromfile(f, dtype=np.uint8)
        
        # 4. Handle bit depth and channels
        channels = 1 if magic == 'P5' else 3
        
        if max_val >= 256:
            # 16-bit: Data is 2 bytes per channel, big-endian
            data = data.view(np.uint16)
            # The C++ code uses byteswapping for 16-bit
            data = data.byteswap()
            
        return data.reshape((h, w, channels)), max_val

def read_collection(data_root: Path, pattern: str) -> Tuple[np.ndarray, int]:
    """
    Scans a directory for PPM files matching a pattern, mapping their U and V 
    coordinates to a compact 5D Lightfield tensor (V, U, H, W, C).
    """
    regex = re.compile(pattern)
    view_list: Dict[Tuple[int, int], Path] = {}

    for entry in data_root.iterdir():
        if not entry.is_file():
            continue
            
        match = regex.match(entry.name)
        if match:
            u = int(match.group('U'))
            v = int(match.group('V'))
            view_list[(v, u)] = entry

    if not view_list:
        raise RuntimeError(f"No views found in {data_root} matching pattern: {pattern}")

    # Build sorted unique lists of original coords to map to compact indices
    uniq_v = sorted(list(set(k[0] for k in view_list.keys())))
    uniq_u = sorted(list(set(k[1] for k in view_list.keys())))

    v_map = {val: idx for idx, val in enumerate(uniq_v)}
    u_map = {val: idx for idx, val in enumerate(uniq_u)}

    # Get shape from the first entry
    first_entry = next(iter(view_list.values()))
    first_view, scale = read_ppm(first_entry)
    h, w, c = first_view.shape

    # Initialize empty Lightfield tensor
    lightfield_shape = (len(uniq_v), len(uniq_u), h, w, c)
    lightfield = np.zeros(lightfield_shape, dtype=first_view.dtype)

    # Populate tensor
    for (v_orig, u_orig), path in view_list.items():
        v_idx = v_map[v_orig]
        u_idx = u_map[u_orig]
        view_data, _ = read_ppm(path)
        lightfield[v_idx, u_idx] = view_data

    return lightfield, scale

def compute_block_psnr(gt_lf: np.ndarray, input_lf: np.ndarray, block_size: Tuple[int, int], max_val: int) -> np.ndarray:
    """Computes PSNR per spatial block by aggregating MSE over the angular and channel dimensions."""
    v, u, h, w, c = gt_lf.shape
    bh, bw = block_size
    
    num_blocks_h = h // bh
    num_blocks_w = w // bw
    
    psnr_map = np.zeros((num_blocks_h, num_blocks_w), dtype=np.float32)
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            h_start, h_end = i * bh, (i + 1) * bh
            w_start, w_end = j * bw, (j + 1) * bw
            
            gt_block = gt_lf[:, :, h_start:h_end, w_start:w_end, :].astype(np.float64)
            in_block = input_lf[:, :, h_start:h_end, w_start:w_end, :].astype(np.float64)
            
            mse = np.mean((gt_block - in_block) ** 2)
            
            if mse == 0:
                psnr_map[i, j] = float('inf')
            else:
                psnr_map[i, j] = 10 * np.log10((max_val ** 2) / mse)
                
    return psnr_map

def main():
    parser = argparse.ArgumentParser(description="Compute block-wise PSNR on Lightfield Tensors")
    parser.add_argument('--gt', type=Path, required=True, help="Path to ground truth folder")
    parser.add_argument('--input', type=Path, required=True, help="Path to input folder")
    parser.add_argument('-b', '--block_size', type=int, nargs=2, required=True, help="Block size: height width")
    parser.add_argument('--o', type=Path, required=True, help="Output image file path (.png)")
    parser.add_argument('--bits', type=int, default=None, help="Assumed bit depth (e.g., 10). If omitted, uses the PPM header max_val.")
    parser.add_argument('--pattern', type=str, default=r"^(?P<U>\d+)_(?P<V>\d+)\.ppm$", help="Regex pattern with U and V named groups")

    args = parser.parse_args()

    # Create the tensors
    gt_lf, gt_max_val = read_collection(args.gt, args.pattern)
    in_lf, in_max_val = read_collection(args.input, args.pattern)

    if gt_lf.shape != in_lf.shape:
        raise ValueError(f"Tensor shape mismatch: GT {gt_lf.shape} vs Input {in_lf.shape}")

    # Determine maximum signal value for PSNR
    max_value = (1 << args.bits) - 1 if args.bits is not None else gt_max_val

    # Compute PSNR matrix
    psnr_map = compute_block_psnr(gt_lf, in_lf, tuple(args.block_size), max_value)
    
    # Cap infinity values to the max finite value for heatmap rendering
    if np.any(np.isinf(psnr_map)):
        finite_max = np.max(psnr_map[np.isfinite(psnr_map)]) if np.any(np.isfinite(psnr_map)) else 100.0
        psnr_map[np.isinf(psnr_map)] = finite_max

    # Render and save the output matrix
    plt.imshow(psnr_map, cmap='jet', interpolation='nearest')
    plt.colorbar(label='PSNR (dB)')
    plt.title(f'Block-wise PSNR ({args.block_size[0]}x{args.block_size[1]})')
    plt.tight_layout()
    plt.savefig(args.o)
    print(f"Successfully generated block PSNR heatmap at: {args.o}")

if __name__ == "__main__":
    main()
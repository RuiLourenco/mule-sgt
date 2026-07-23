import math
import numpy as np

def get_valid_position(adjustment_d, lf_shape, block_shape, block_start, is_horizontal):
    view_coordinate = 1 if is_horizontal else 0
    spatial_coordinate = 3 if is_horizontal else 2
    
    lf_extra_size = int(abs(round(adjustment_d * (lf_shape[view_coordinate] - 1))))
    lf_shape_spatial = lf_shape[spatial_coordinate] - lf_extra_size
    
    if adjustment_d == 0:
        true_alpha = 0.0
    else:
        true_alpha = (adjustment_d / abs(adjustment_d)) * (lf_extra_size / (lf_shape[view_coordinate] - 1))
        
    padding_coordinates = []
    
    for l_ in range(lf_shape[view_coordinate]):
        if block_start[view_coordinate] > l_:
            continue
        if block_start[view_coordinate] + block_shape[view_coordinate] - 1 < l_:
            continue
            
        n_start = math.floor(l_ * true_alpha)
        n_end = lf_shape_spatial + math.floor(l_ * true_alpha)
        
        if true_alpha < 0:
            n_start -= (lf_shape[view_coordinate] - 1) * true_alpha
            n_end -= (lf_shape[view_coordinate] - 1) * true_alpha
            
        if n_start - block_start[spatial_coordinate] < block_shape[spatial_coordinate]:
            blk_n_start = max(n_start - block_start[spatial_coordinate], 0)
            blk_n_end = min(n_end - block_start[spatial_coordinate], block_shape[spatial_coordinate])
            
            if blk_n_end - blk_n_start >= 1:
                indexes = list(range(int(blk_n_start), int(blk_n_end)))
                indexes = [i + block_shape[spatial_coordinate] * l_ for i in indexes]
                padding_coordinates.extend(indexes)
                
    return padding_coordinates

lf_shape = [9, 9, 512, 512]
block_shape = [9, 9, 16, 16]
block_start = [0, 0, 0, 0] # Top-left block
adjustment_d = 0.1
print("Horizontal (Top Left Block):")
h_idx = get_valid_position(adjustment_d, lf_shape.copy(), block_shape.copy(), block_start.copy(), True)
print(f"Num valid: {len(h_idx)}")

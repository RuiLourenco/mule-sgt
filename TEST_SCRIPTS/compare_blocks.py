import re
import sys

def parse_file(filename, is_encoder):
    blocks = {}
    current_block_pos = None
    current_block_data = {'lines': []}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if line.startswith("Block Position:"):
            if current_block_pos is not None:
                blocks[current_block_pos] = current_block_data
            current_block_pos = line
            current_block_data = {'lines': []}
            continue
        
        if current_block_pos is not None:
            if is_encoder and (line.startswith("origBlock.data mean:") or 
                               line.startswith("invBlockTensor mean:") or 
                               line.startswith("MSE (Original vs Inverse):") or 
                               line.startswith("PSNR (Original vs Inverse):") or 
                               line.startswith("MSE (Mathematical With Integer Truncation):") or 
                               line.startswith("Max Diff is at") or 
                               line.startswith("Number of coefficients with diff > 100:") or
                               line.startswith("Max Diff between math recoveredQuant") or
                               line.startswith("MSE (Standalone Decode):") or
                               line.startswith("PSNR (Standalone Decode):")):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    current_block_data[parts[0].strip()] = parts[1].strip()
            elif line.startswith("----------------------------------------"):
                pass
            else:
                current_block_data['lines'].append(line)
        
    if current_block_pos is not None:
        blocks[current_block_pos] = current_block_data
        
    return blocks

enc_blocks = parse_file("results/Set2/set2_0.1_testPolarityFix.comp_encoder_matrices_Y.txt", True)
dec_blocks = parse_file("results/Set2/set2_0.1_testPolarityFix.comp_decoder_matrices_Y.txt", False)

divergent_blocks = []
for pos, dec_data in dec_blocks.items():
    if pos in enc_blocks:
        enc_data = enc_blocks[pos]
        enc_lines = '\n'.join(enc_data['lines'])
        dec_lines = '\n'.join(dec_data['lines'])
        if enc_lines != dec_lines:
            divergent_blocks.append(pos)
    else:
        divergent_blocks.append(f"{pos} (Missing in Encoder)")

problematic_blocks = []
for pos, enc_data in enc_blocks.items():
    problem = False
    reasons = []
    
    diff_gt_100_str = enc_data.get('Number of coefficients with diff > 100', '0')
    try:
        if int(diff_gt_100_str) > 0:
            problem = True
            reasons.append(f"coeffs with diff > 100 = {diff_gt_100_str}")
    except:
        pass
        
    max_diff_str = enc_data.get('Max Diff between math recoveredQuant and tempBlock.getFlatBlock()', '0')
    try:
        if int(max_diff_str) > 5:
            problem = True
            reasons.append(f"Max Diff recoveredQuant vs tempBlock = {max_diff_str}")
    except:
        pass
        
    mse_orig_str = enc_data.get('MSE (Original vs Inverse)', '0')
    try:
        if float(mse_orig_str) > 10.0:
            problem = True
            reasons.append(f"MSE (Original vs Inverse) = {mse_orig_str}")
    except:
        pass

    if problem:
        problematic_blocks.append(f"{pos}: " + ", ".join(reasons))

with open('comparison_output.txt', 'w') as f:
    f.write("DIVERGENT BLOCKS (Encoder vs Decoder):\n")
    if not divergent_blocks:
        f.write("None found.\n")
    for b in divergent_blocks:
        f.write(f"- {b}\n")
        
    f.write("\nPROBLEMATIC ENCODER BLOCKS:\n")
    if not problematic_blocks:
        f.write("None found.\n")
    for b in problematic_blocks:
        f.write(f"- {b}\n")

print("Analysis complete. Check comparison_output.txt")

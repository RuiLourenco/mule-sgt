import torch
import numpy as np
import matplotlib.pyplot as plt


# Load the eigenvalues from both runs
encoder_evals_script = torch.jit.load('/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-structure-tensor/63bb69c07be2d-5-7fcb5636ce00_transform_eig_d_signNorm.pt')
decoder_evals_script = torch.jit.load('/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-structure-tensor/63bb69f17fff0-5-7fdcddd4c0c0_transform_eig_d_signNorm.pt')

encoder_evals = encoder_evals_script.data
decoder_evals = decoder_evals_script.data
print("Encoder tensor shape:", encoder_evals.shape)
print("Decoder tensor shape:", decoder_evals.shape)
# Compute the absolute difference between the two tensors
abs_diff = torch.abs(encoder_evals) - torch.abs(decoder_evals)
diff = encoder_evals - decoder_evals
max_per_column, max_row_indices = torch.max(abs_diff, dim=0)

for i in range(0, 17*64):
    if i < abs_diff.shape[1] and max_per_column[i] > 1e-3:
        col_data = encoder_evals[:, i].cpu().numpy()
        col_data_flipped = np.flip(col_data)
        col_data_reshaped = col_data.reshape(17, 64)
        col_data_flipped_reshaped = col_data_flipped.reshape(17, 64)
        plt.imsave(f'encoder_column_{i}.png', col_data_reshaped, cmap='jet')
        plt.imsave(f'encoder_flipped_column_{i}.png', col_data_flipped_reshaped, cmap='jet')
        max_row = max_row_indices[i].item()
        print(f"Column {i} has a significant error: {max_per_column[i].item()} at row {max_row}")

       

diff_np = diff.cpu().numpy()
diff_np = np.clip(diff_np,-1e-8, 1e-8)
plt.imsave('diff.png', diff_np, cmap='jet')
# Normalize abs_diff so that 1e-5 is considered the maximum value
abs_diff_np = abs_diff.cpu().numpy()
max_val = 1e-8
abs_diff_norm = np.clip(abs_diff_np / max_val, 0, 1)
plt.imsave('abs_diff.png', abs_diff_norm, cmap='jet')
# Normalize encoder_evals and decoder_evals to range [-0.01, 0.01

encoder_evals_norm = np.clip(encoder_evals, -0.1,0.1)
decoder_evals_norm = np.clip(decoder_evals,-0.1,0.1)

plt.imsave('encoder.png', encoder_evals_norm.cpu().numpy(), cmap='jet')
plt.imsave('decoder.png', decoder_evals_norm.cpu().numpy(), cmap='jet')

# Find the index of the maximum difference
max_diff = torch.max(abs_diff)
max_idx = torch.argmax(abs_diff)


# Print the coordinates and the value of the maximum difference
coords = list(np.unravel_index(max_idx.cpu().numpy() if max_idx.is_cuda else max_idx.numpy(), abs_diff.shape))
col_data = encoder_evals[:, coords[1]].cpu().numpy()
col_data_reshaped = col_data.reshape(17, 64)
plt.imsave('encoder_column.png', col_data_reshaped, cmap='jet')

col_data = decoder_evals[:, coords[1]].cpu().numpy()
col_data_reshaped = col_data.reshape(17, 64)
plt.imsave('decoder_column.png', col_data_reshaped, cmap='jet')


print(f"Maximum difference: {max_diff.item()} at coordinates {coords}")
row_idx = coords[0]
print("First 12 items of the row with maximum difference:")
print(encoder_evals[0:6,194].tolist())
print(decoder_evals[0:6,194].tolist())
print(abs_diff[0:6,194].tolist())
print(f"Encoder eigenvalue at index: {encoder_evals.flatten()[max_idx].item()}")
print(f"Decoder eigenvalue at index: {decoder_evals.flatten()[max_idx].item()}")
# It's highly likely the eigenvalues themselves WILL be identical.
# The problem is within one of these lists.
print("Are the eigenvalue lists bit-for-bit identical?", torch.equal(encoder_evals, decoder_evals))

# Let's check for repeated values. Sort them first.
sorted_evals = torch.sort(encoder_evals).values

# Compute the difference between adjacent values
diffs = torch.diff(sorted_evals)

# Find the minimum difference. Is it close to zero?
min_diff = torch.min(diffs)
print(f"Minimum difference between sorted eigenvalues: {min_diff.item()}")

# If min_diff is very small (e.g., < 1e-6 or smaller), you have repeated or near-repeated eigenvalues.
if min_diff < 1e-6:
    print("\n*** WARNING: Repeated or near-repeated eigenvalues detected! ***")
    print("This is almost certainly the cause of the eigenvector instability.")
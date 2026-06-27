#include "LightField/LightField.h"
#include "LightField/Block4D_.h"

#include <string.h>
#include <stdlib.h>
#include "IO/io.h"


/******************************************************************** */
/*                         Gradient Calculations                      */    
/******************************************************************** */

// Function to create 1D Gaussian kernel
torch::Tensor create_gaussian_kernel(int size, float sigma) {
    torch::Tensor kernel = torch::zeros({size});
    int center = size / 2;
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float x = i - center;
        float value = std::exp(-(x * x) / (2 * sigma * sigma));
        kernel[i] = value;
        sum += value;
    }
    
    // Normalize the kernel
    kernel /= sum;
    return kernel;
}

// Function to create 1D derivative of Gaussian kernel
torch::Tensor create_gaussian_derivative_kernel(int size, float sigma) {
    torch::Tensor kernel = torch::zeros({size});
    int center = size / 2;
    
    for (int i = 0; i < size; i++) {
        float x = i - center;
        // Derivative of Gaussian: -x/(sigma^2) * exp(-x^2/(2*sigma^2))
        float value = -x * std::exp(-(x * x) / (2 * sigma * sigma)) / (sigma * sigma);
        kernel[i] = value;
    }
    
    // Normalize the kernel to ensure it sums to zero and has appropriate magnitude
    float pos_sum = 0.0f;
    float neg_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = kernel[i].item<float>();
        if (val > 0) pos_sum += val;
        else neg_sum -= val;
    }
    
    float scale = std::max(pos_sum, neg_sum);
    if (scale > 0) kernel /= scale;
    
    return kernel;
}



// Alternative implementation using separable convolutions for better efficiency
torch::Tensor compute_first_order_derivatives_separable(const torch::Tensor& input, int kernel_size = 5, float sigma = 1.0) {
    // Check if input is 4D
    if (input.dim() != 4) {
        throw std::runtime_error("Input tensor must be 4D");
    }
    
    // Get dimensions
    auto dims = input.sizes();
    int64_t dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim3 = dims[3];
    
    // Create Gaussian and derivative kernels
    torch::Tensor gaussian_kernel = create_gaussian_kernel(kernel_size, sigma);
    torch::Tensor derivative_kernel = create_gaussian_derivative_kernel(kernel_size, sigma);
    
    // Initialize the 5D output tensor
    torch::Tensor output = torch::zeros({dim0, dim1, dim2, dim3, 4}, at::kDouble);
    
    // Work with a CPU tensor for consistent indexing
    bool was_cuda = input.is_cuda();
    torch::Tensor cpu_input = was_cuda ? input.to(torch::kCPU) : input;
    
    // For each dimension
    for (int dim = 0; dim < 4; dim++) {
        // Create a copy to work with
        torch::Tensor temp_result = cpu_input.clone().to(at::kDouble);
        // Apply explicit separable convolution
        int pad_size = kernel_size / 2;
        
        // Smoothing and filtering along all dimensions
        for (int other_dim = 0; other_dim < 4; other_dim++) {
            at::Tensor kernel;
            if (other_dim == dim) {
                kernel = derivative_kernel;
            }else{
                kernel = gaussian_kernel;
            }
            
            
            // Create temporary result tensor
            std::array<int64_t,4> permutation = {0,1,2,3};
            permutation[3] = other_dim;
            permutation[other_dim] = 3;
            
            // Apply convolution based on dimension 
            // The first step should be to reshape the Tensor into a 2D image where only the dimension of interest is kept
            temp_result = temp_result.permute(permutation);
            std::vector<int64_t> sizePermuted(temp_result.sizes().begin(), temp_result.sizes().end());
            temp_result = temp_result.reshape({-1, dims[other_dim]});
            kernel = kernel.unsqueeze(0).unsqueeze(0);
            kernel = kernel.expand({temp_result.size(0),1,kernel.size(2)}).to(at::kDouble);
            std::cout<<"kernel size = "<<kernel.sizes()<<std::endl;
            temp_result = torch::nn::functional::conv1d(temp_result, kernel, torch::nn::functional::Conv1dFuncOptions().padding(torch::kSame).groups(temp_result.size(0)));
            std::cout<<"temp_result size = "<<temp_result.sizes()<<std::endl;
            std::cout<<sizePermuted<<std::endl;

            temp_result = temp_result.reshape(sizePermuted);
            std::cout<<"temp_result size = "<<temp_result.sizes()<<std::endl;
            temp_result = temp_result.permute(permutation); 
            std::cout<<"FINAL temp_result size = "<<temp_result.sizes()<<std::endl;

        }
        
        // Apply derivative filter along the current dimension
        torch::Tensor derivative = temp_result;
        
        output.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),dim}) = temp_result;
    }
    // Move back to GPU if needed
    if (was_cuda) {
        output = output.to(torch::kCUDA);
    }
    
    return output;
}

torch::Tensor compute_first_order_derivatives_in_place_style(const torch::Tensor& input, int kernel_size = 5, float sigma = 1.0) {
    // Check if input is 4D
    if (input.dim() != 4) {
        throw std::runtime_error("Input tensor must be 4D");
    }

    // Get dimensions
    auto dims = input.sizes();
    int64_t dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim3 = dims[3];

    // Create Gaussian and derivative kernels
    torch::Tensor gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(input.device()).to(at::kDouble);
    torch::Tensor derivative_kernel = create_gaussian_derivative_kernel(kernel_size, sigma).to(input.device()).to(at::kDouble);

    // Initialize the 5D output tensor
    torch::Tensor output = torch::zeros({dim0, dim1, dim2, dim3, 4}, input.options().dtype(at::kDouble));

    // Allocate two buffers for processing to avoid repeated allocations
    torch::Tensor buffer1 = torch::empty_like(input, input.options().dtype(at::kDouble));
    torch::Tensor buffer2 = torch::empty_like(input, input.options().dtype(at::kDouble));

    // For each dimension (u, v, x, y)
    for (int dim_to_derive = 0; dim_to_derive < 4; ++dim_to_derive) {
        // Start with the original data in buffer1
        buffer1.copy_(input.to(at::kDouble));

        // Apply separable convolution along all dimensions
        for (int other_dim = 0; other_dim < 4; ++other_dim) {
            at::Tensor& kernel = (other_dim == dim_to_derive) ? derivative_kernel : gaussian_kernel;
            
            // Permute the dimension to be convolved to the last position
            std::vector<int64_t> permutation;
            for(int i = 0; i < 4; ++i) if(i != other_dim) permutation.push_back(i);
            permutation.push_back(other_dim);

            torch::Tensor current_src = buffer1.permute(permutation);
            
            // Reshape for 1D convolution
            std::vector<int64_t> original_shape(current_src.sizes().begin(), current_src.sizes().end());
            int64_t last_dim_size = original_shape.back();
            current_src = current_src.reshape({-1, last_dim_size});

            // Prepare kernel for conv1d
            torch::Tensor conv_kernel = kernel.view({1, 1, -1});

            // Apply 1D convolution
            torch::Tensor conv_output = torch::nn::functional::conv1d(
                current_src.unsqueeze(1), 
                conv_kernel, 
                torch::nn::functional::Conv1dFuncOptions().padding(kernel_size / 2)
            ).squeeze(1);

            // Reshape back to original permuted shape
            conv_output = conv_output.view(original_shape);

            // Inverse permutation to get back to original dimension order and store in buffer2
            std::vector<int64_t> inv_permutation(4);
            for(size_t i = 0; i < 4; ++i) inv_permutation[permutation[i]] = i;
            buffer2.copy_(conv_output.permute(inv_permutation));
            
            // Swap buffers for the next iteration
            std::swap(buffer1, buffer2);
        }
        
        // After all convolutions for this derivative, buffer1 holds the result
        output.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), dim_to_derive}, buffer1);
    }

    return output;
}

// Helper function to perform a clean 1D convolution
void apply_conv_along_dim(const torch::Tensor& src, torch::Tensor& dst, int conv_dim, const torch::Tensor& kernel) {
    const int kernel_size = kernel.size(2);

    std::vector<int64_t> permutation;
    for(int i = 0; i < 4; ++i) if(i != conv_dim) permutation.push_back(i);
    permutation.push_back(conv_dim);

    torch::Tensor permuted_src = src.permute(permutation);
    auto permuted_shape = permuted_src.sizes();
    permuted_src = permuted_src.reshape({-1, 1, permuted_shape.back()});

    torch::Tensor conv_result = torch::nn::functional::conv1d(
        permuted_src, kernel,
        torch::nn::functional::Conv1dFuncOptions().padding(kernel_size / 2)
    );

    conv_result = conv_result.reshape(permuted_shape);

    std::vector<int64_t> inv_permutation(4);
    for(size_t i = 0; i < permutation.size(); ++i) {
        inv_permutation[permutation[i]] = i;
    }
    dst = conv_result.permute(inv_permutation);
}

/**
 * @brief Computes derivatives using a highly memory-optimized single-buffer approach.
 * This is the recommended version for your 120 GB system.
 */
torch::Tensor compute_first_order_derivatives_lean(const torch::Tensor& input, int kernel_size = 5, float sigma = 1.0) {
    TORCH_CHECK(input.dim() == 4, "Input tensor must be 4D");
    auto options = input.options(); // Assumes float32 from the calling function
    const auto dims = input.sizes();

    torch::Tensor gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(options).view({1, 1, kernel_size});
    torch::Tensor derivative_kernel = create_gaussian_derivative_kernel(kernel_size, sigma).to(options).view({1, 1, kernel_size});
    std::vector<torch::Tensor> kernels = {gaussian_kernel, gaussian_kernel, gaussian_kernel, gaussian_kernel};

    // --- Peak Memory Allocation ---
    // 1. The final output tensor (e.g., ~66.2 GB)
    torch::Tensor output = torch::zeros({dims[0], dims[1], dims[2], dims[3], 4}, options);
    // 2. A SINGLE intermediate buffer (e.g., ~16.55 GB)
    torch::Tensor buffer = torch::empty_like(input, options);

    for (int dim_to_derive = 0; dim_to_derive < 4; ++dim_to_derive) {
        kernels[dim_to_derive] = derivative_kernel;
        auto dest_slice = output.slice(/*dim=*/4, /*start=*/dim_to_derive, /*end=*/dim_to_derive + 1).squeeze(4);

        // Ping-pong between the single buffer and the final destination slice
        apply_conv_along_dim(input,       buffer,     0, kernels[0]);
        apply_conv_along_dim(buffer,      dest_slice, 1, kernels[1]);
        apply_conv_along_dim(dest_slice,  buffer,     2, kernels[2]);
        apply_conv_along_dim(buffer,      dest_slice, 3, kernels[3]);

        kernels[dim_to_derive] = gaussian_kernel; // Reset for next iteration
        std::cout << "  - Gradient for dimension " << dim_to_derive << " computed." << std::endl;
    }

    return output;
}

/**
 * @brief Core logic to compute derivatives on a single, small chunk of data.
 * The memory allocated inside this function is proportional to the chunk_size,
 * keeping the temporary memory footprint very low.
 */
torch::Tensor process_derivative_chunk(const torch::Tensor& input_chunk, int kernel_size, float sigma) {
    auto options = input_chunk.options();
    const auto dims = input_chunk.sizes();

    torch::Tensor gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).to(options).view({1, 1, kernel_size});
    torch::Tensor derivative_kernel = create_gaussian_derivative_kernel(kernel_size, sigma).to(options).view({1, 1, kernel_size});

    // Buffers are allocated per-chunk, so their size is small.
    torch::Tensor output_chunk = torch::zeros({dims[0], dims[1], dims[2], dims[3], 4}, options);
    torch::Tensor buffer1 = torch::empty_like(input_chunk, options);
    torch::Tensor buffer2 = torch::empty_like(input_chunk, options);

    for (int dim_to_derive = 0; dim_to_derive < 4; ++dim_to_derive) {
        torch::Tensor* current_src = &buffer1;
        torch::Tensor* current_dst = &buffer2;
        current_src->copy_(input_chunk);

        for (int conv_dim = 0; conv_dim < 4; ++conv_dim) {
            const auto& kernel = (conv_dim == dim_to_derive) ? derivative_kernel : gaussian_kernel;

            // Permute the dimension to convolve to the last position
            std::vector<int64_t> permutation;
            for(int i = 0; i < 4; ++i) if(i != conv_dim) permutation.push_back(i);
            permutation.push_back(conv_dim);

            torch::Tensor permuted_src = current_src->permute(permutation);
            std::vector<int64_t> permuted_shape(permuted_src.sizes().begin(), permuted_src.sizes().end());
            
            // Reshape for conv1d
            permuted_src = permuted_src.reshape({-1, 1, permuted_shape.back()});

            // Apply standard 1D convolution
            torch::Tensor conv_result = torch::nn::functional::conv1d(
                permuted_src, kernel,
                torch::nn::functional::Conv1dFuncOptions().padding(kernel_size / 2)
            );

            // Reshape back to 4D
            conv_result = conv_result.reshape(permuted_shape);

            // Invert permutation and store in the destination buffer
            std::vector<int64_t> inv_permutation(4);
            for(size_t i = 0; i < permutation.size(); ++i) {
                inv_permutation[permutation[i]] = i;
            }
            *current_dst = conv_result.permute(inv_permutation);
            
            // Swap buffers for the next iteration
            std::swap(current_src, current_dst);
        }
        output_chunk.index_put_({torch::indexing::Slice(), "...", dim_to_derive}, *current_src);
    }
    return output_chunk;
}

/**
 * @brief Manages the chunking process. It allocates the full final gradient tensor,
 * then iterates through the input data slice-by-slice, processing each one with
 * a low memory footprint and filling the final tensor.
 */
torch::Tensor compute_first_order_derivatives_chunked(const torch::Tensor& y_channel, int kernel_size = 5, float sigma = 1.0, int chunk_size = 1) {
    TORCH_CHECK(y_channel.dim() == 4, "Input tensor must be 4D");
    const auto full_dims = y_channel.sizes();
    int64_t total_slices = full_dims[0];

    // Allocate the full, final output tensor. This is one of the large, persistent allocations.
    torch::Tensor final_output = torch::zeros({full_dims[0], full_dims[1], full_dims[2], full_dims[3], 4}, y_channel.options());

    std::cout << "Processing " << total_slices << " slices in chunks of " << chunk_size << "..." << std::endl;

    for (int64_t i = 0; i < total_slices; i += chunk_size) {
        int64_t start_slice = i;
        int64_t end_slice = std::min(i + chunk_size, total_slices);
        
        std::cout << "  - Processing slices [" << start_slice << " to " << end_slice - 1 << "]" << std::endl;

        // Get a VIEW of the current chunk from the Y-channel. This is memory-free.
        auto input_chunk_view = y_channel.slice(/*dim=*/0, /*start=*/start_slice, /*end=*/end_slice);

        // Process this small chunk. All temporary buffers are now tiny.
        auto gradient_chunk = process_derivative_chunk(input_chunk_view.contiguous(), kernel_size, sigma);

        // Place the computed chunk into the correct location in the final output tensor.
        final_output.slice(/*dim=*/0, /*start=*/start_slice, /*end=*/end_slice) = gradient_chunk;
    }
    return final_output;
}
/**
 * @brief Manages chunking WITH OVERLAP to ensure correct gradient calculation.
 * This is the mathematically correct approach for out-of-core convolution.
 */
torch::Tensor compute_first_order_derivatives_chunked_with_overlap(const torch::Tensor& y_channel, int kernel_size = 5, float sigma = 1.0, int chunk_size = 1) {
    TORCH_CHECK(y_channel.dim() == 4, "Input tensor must be 4D");
    const auto full_dims = y_channel.sizes();
    int64_t total_slices = full_dims[0];
    
    // The "halo" size needed on each side of a chunk
    const int64_t pad_size = kernel_size / 2;

    torch::Tensor final_output = torch::zeros({full_dims[0], full_dims[1], full_dims[2], full_dims[3], 4}, y_channel.options());

    std::cout << "Processing " << total_slices << " slices in chunks of " << chunk_size << " with overlap of " << pad_size << "..." << std::endl;

    for (int64_t i = 0; i < total_slices; i += chunk_size) {
        // Define the region for the CURRENT chunk's final output
        int64_t chunk_start = i;
        int64_t chunk_end = std::min(i + chunk_size, total_slices);
        
        // Define the PADDED region we need to LOAD from the input tensor
        int64_t padded_start = std::max((int64_t)0, chunk_start - pad_size);
        int64_t padded_end = std::min(total_slices, chunk_end + pad_size);

        std::cout << "  - Processing chunk " << chunk_start << ":" << chunk_end-1
                  << " (loading data from " << padded_start << ":" << padded_end-1 << ")" << std::endl;

        // Get a VIEW of the input data WITH PADDING. This is still cheap.
        auto input_chunk_with_padding = y_channel.slice(/*dim=*/0, /*start=*/padded_start, /*end=*/padded_end);

        // Process the entire padded chunk. Buffers are sized to this larger chunk.
        auto gradient_chunk_with_padding = process_derivative_chunk(input_chunk_with_padding.contiguous(), kernel_size, sigma);

        // Define the region to CROP from the processed chunk.
        // This corresponds to the valid, non-halo area.
        int64_t crop_start = chunk_start - padded_start;
        int64_t crop_end = crop_start + (chunk_end - chunk_start);

        // Get a VIEW of the valid gradient data
        auto valid_gradient_chunk = gradient_chunk_with_padding.slice(/*dim=*/0, /*start=*/crop_start, /*end=*/crop_end);

        // Place the valid, computed chunk into the correct location in the final output tensor.
        final_output.slice(/*dim=*/0, /*start=*/chunk_start, /*end=*/chunk_end) = valid_gradient_chunk;
    }
    return final_output;
}
/**
 * @brief The core engine. Computes gradients for a specified sub-volume and writes them
 * into a pre-allocated output_buffer. The sub-volume is defined by a slice range
 * along a specified split dimension.
 *
 * @param y_channel The full 4D source tensor for the Y-channel.
 * @param output_buffer A pre-allocated tensor where the results will be written.
 * @param split_dim The dimension along which to slice and process (e.g., 2 for the 'x' spatial dim).
 * @param range_start The starting index (inclusive) along split_dim to process.
 * @param range_end The ending index (exclusive) along split_dim to process.
 * @param kernel_size The size of the convolution kernels.
 * @param sigma The standard deviation of the Gaussian.
 * @param chunk_size A tunable parameter for performance vs. memory, applied along split_dim.
 */
void compute_gradient_subvolume_into_buffer(
    const torch::Tensor& y_channel,
    torch::Tensor& output_buffer,
    int split_dim,
    int64_t range_start,
    int64_t range_end,
    int kernel_size,
    float sigma,
    int chunk_size)
{
    const auto full_dims = y_channel.sizes();
    int64_t total_slices_in_dim = full_dims[split_dim];
    const int64_t pad_size = kernel_size / 2;

    TORCH_CHECK(output_buffer.size(split_dim) == (range_end - range_start),
                "Output buffer has incorrect size for the given range along the split dimension.");

    std::cout << "Processing dimension " << split_dim << ", slices " << range_start << " to " << range_end - 1
              << " in chunks of " << chunk_size << "..." << std::endl;

    for (int64_t i = range_start; i < range_end; i += chunk_size) {
        // Define the region for the CURRENT chunk's final output, relative to the full tensor
        int64_t chunk_start_full = i;
        int64_t chunk_end_full = std::min(i + chunk_size, range_end);
        
        // Define the PADDED region we need to LOAD from the input tensor
        int64_t padded_start = std::max((int64_t)0, chunk_start_full - pad_size);
        int64_t padded_end = std::min(total_slices_in_dim, chunk_end_full + pad_size);

        std::cout << "  - Chunk " << chunk_start_full << ":" << chunk_end_full - 1
                  << " (loading data from " << padded_start << ":" << padded_end - 1 << ")" << std::endl;

        // Get a VIEW of the input data WITH PADDING along the specified split_dim.
        auto input_chunk_with_padding = y_channel.slice(/*dim=*/split_dim, /*start=*/padded_start, /*end=*/padded_end);
        auto gradient_chunk_with_padding = process_derivative_chunk(input_chunk_with_padding.contiguous(), kernel_size, sigma);

        // Define the region to CROP from the processed chunk.
        int64_t crop_start = chunk_start_full - padded_start;
        int64_t crop_end = crop_start + (chunk_end_full - chunk_start_full);
        auto valid_gradient_chunk = gradient_chunk_with_padding.slice(/*dim=*/split_dim, /*start=*/crop_start, /*end=*/crop_end);

        // Calculate the write position relative to the START of the output_buffer
        int64_t buffer_write_start = chunk_start_full - range_start;
        int64_t buffer_write_end = chunk_end_full - range_start;

        // Place the valid, computed chunk into the correct location in the output buffer.
        output_buffer.slice(/*dim=*/split_dim, /*start=*/buffer_write_start, /*end=*/buffer_write_end) = valid_gradient_chunk;
    }
}
/**
 * @brief Calculates a split point that is a multiple of block_size and is as
 * close as possible to the halfway point of the dimension.
 *
 * @param total_slices_in_dim The total size of the dimension being split.
 * @param block_size The size of the compression algorithm's blocks along this dimension.
 * @return A split point that is guaranteed to be a multiple of block_size.
 */
int64_t calculate_block_aligned_split_point(int64_t total_slices_in_dim, int block_size) {
    if (block_size <= 0) {
        // Fallback to a simple half-split if block size is invalid
        return total_slices_in_dim / 2;
    }

    int64_t ideal_split = total_slices_in_dim / 2;

    // Find the last block boundary before or at the ideal split point.
    // Integer division handles this naturally.
    int64_t num_blocks_to_midpoint = ideal_split / block_size;
    int64_t aligned_split_point = num_blocks_to_midpoint * block_size;

    // If the ideal split is very small, we might get a split point of 0.
    // This is correct: the "first half" will be empty, and the "second half" will be everything.
    return aligned_split_point;
}

/**
 * @brief Computes the gradients for the first half of a specified dimension.
 * This function allocates and returns a new tensor for the first half gradients.
 *
 * @param split_dim The dimension to split along (e.g., 2).
 * @return A tensor containing the gradients for the first half.
 */
torch::Tensor compute_first_half_gradients(
    const torch::Tensor& y_channel,
    int split_dim,
    int block_size,
    int kernel_size,
    float sigma,
    int chunk_size)
{
    const auto full_dims = y_channel.sizes();
    int64_t total_slices_in_dim = full_dims[split_dim];
        // Use the new helper to find the correct split point
    int64_t split_point = calculate_block_aligned_split_point(total_slices_in_dim, block_size);

    // Allocate a new tensor for the first half
    auto first_half_dims = full_dims.vec();
    first_half_dims[split_dim] = split_point;
    auto gradient_dims = first_half_dims;
    gradient_dims.push_back(4); // Add the gradient dimension
    torch::Tensor first_half_gradients = torch::zeros(gradient_dims, y_channel.options());

    if (split_point > 0) {
        compute_gradient_subvolume_into_buffer(y_channel, first_half_gradients, split_dim, 0, split_point, kernel_size, sigma, chunk_size);
    }

    return first_half_gradients;
}
/**
 * @brief Computes the gradients for the second half of a specified dimension,
 * overwriting the provided buffer in-place.
 *
 * @param split_dim The dimension to split along (e.g., 2).
 * @param gradient_buffer This tensor's contents will be REPLACED with the second half gradients.
 */
void compute_second_half_gradients_inplace(
    const torch::Tensor& y_channel,
    torch::Tensor& gradient_buffer,
    int split_dim,
    int block_size,
    int kernel_size,
    float sigma,
    int chunk_size)
{
    const auto full_dims = y_channel.sizes();
    int64_t total_slices_in_dim = full_dims[split_dim];
    // Calculate the same split point to know where to start
    int64_t split_point = calculate_block_aligned_split_point(total_slices_in_dim, block_size);
    int64_t second_half_start = split_point;
    int64_t second_half_end = total_slices_in_dim;

    // Check if the provided buffer is the right size for the second half
    int64_t second_half_size = second_half_end - second_half_start;
    TORCH_CHECK(gradient_buffer.size(split_dim) == second_half_size, "Provided buffer is not the correct size for the second half.");

    // Call the core engine to fill the buffer
    compute_gradient_subvolume_into_buffer(y_channel, gradient_buffer, split_dim, second_half_start, second_half_end, kernel_size, sigma, chunk_size);
}



/*******************************************************************************/
/*                        LightField class methods                             */
/*******************************************************************************/
void LightField::computeTopHalfGradients(){
    const int KERNEL_SIZE = 5;
    const int CHUNK_SIZE = 128; // Your tunable parameter, applied along dim 2
    const int SPLIT_DIM = 2;   // The crucial change: we are splitting spatially!
    torch::Tensor y_channel_slice = this->data.index({
        torch::indexing::Slice(), "...", 0
        }).to(torch::kFloat32);
    std::cout << "\n--- Computing gradients for FIRST half of dimension " << SPLIT_DIM << " ---" << std::endl;
    this->gradients = compute_first_half_gradients(y_channel_slice, SPLIT_DIM, 128, KERNEL_SIZE, 1.0, CHUNK_SIZE);
    this -> secondHalfGradientsComputed = false;
    const auto full_dims = y_channel_slice.sizes();
    this->secondHalfBias = calculate_block_aligned_split_point(full_dims[SPLIT_DIM],128);
    if (this->gradients.size(SPLIT_DIM) > 0) {
        write_tensor(this->gradients.index({this->gradients.size(0)/2, this->gradients.size(1)/2, torch::indexing::Slice(), torch::indexing::Slice(), 2}), "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-pre-slant-st-fixed/results/Greek/firstHalfGradients.png");
    }

}

void LightField::computeBottomHalfGradients(){
    const int KERNEL_SIZE = 5;
    const int CHUNK_SIZE = 128; // Your tunable parameter, applied along dim 2
    const int SPLIT_DIM = 2;   // The crucial change: we are splitting spatially!
    const int BLOCK_SIZE = 128; // The block size used in compression along the split dimension
    torch::Tensor y_channel_slice = this->data.index({
        torch::indexing::Slice(), "...", 0
    }).to(torch::kFloat32);
    const auto full_dims = y_channel_slice.sizes();
    int64_t split_point = calculate_block_aligned_split_point(full_dims[SPLIT_DIM], BLOCK_SIZE);
    int64_t second_half_size = full_dims[SPLIT_DIM] - split_point;
    
    auto second_half_dims = full_dims.vec();
    second_half_dims[SPLIT_DIM] = second_half_size;
    auto gradient_dims = second_half_dims;
    gradient_dims.push_back(4);
    this->gradients = torch::zeros(gradient_dims, y_channel_slice.options()); 

    compute_second_half_gradients_inplace(y_channel_slice, this->gradients, SPLIT_DIM,128, KERNEL_SIZE, 1.0, CHUNK_SIZE);
    this->secondHalfGradientsComputed  = true;
    this->secondHalfBias = split_point;
    if (this->gradients.size(SPLIT_DIM) > 0) {
        write_tensor(this->gradients.index({this->gradients.size(0)/2, this->gradients.size(1)/2, torch::indexing::Slice(), torch::indexing::Slice(), 2}), "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-pre-slant-st-fixed/results/Greek/SecondHalfGradients.png");
    }

    std::cout<<secondHalfBias<<std::endl;
}
void LightField::computeGradients(){
    std::cout<<"Computing Gradients"<<std::endl;
    this->gradients =  compute_first_order_derivatives_separable(this->data.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}),5,1.0);
    if (this->gradients.size(2) > 0) {
        write_tensor(this->gradients.index({this->gradients.size(0)/2, this->gradients.size(1)/2, torch::indexing::Slice(), torch::indexing::Slice(), 2}), "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-pre-slant-st-fixed/results/Greek/fullGradients.png");
    }

    std::cout<<"Gradients Computed!"<<std::endl;
}
LightField::LightField(std::string root_path,std::string pattern) {
    OpenLightFieldPPM_(root_path,pattern,'r',{0,0});
    this ->secondHalfBias = this->data.size(2);
   
}
/**
 * Constructs a LightField object with a specified size.
 * Initializes the data tensor and view cache based on the provided dimensions.
 *
 * @param size An array containing the dimensions of the light field:
 *             - size[0]: Number of vertical views
 *             - size[1]: Number of horizontal views
 *             - size[2]: Number of view columns
 *             - size[3]: Number of view lines
 *             - size[4]: Number of Colour Channels
 */


LightField :: LightField(std::array<int64_t,5> size){
    this->data = at::zeros({size[0],size[1],size[2],size[3],size[4]},at::kInt);
    this ->secondHalfBias = this->data.size(2);

}

void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, std::array<int64_t,2> firstView, std::array<int64_t,2> viewSize) {
        this->data = io::read_collection(rootPath, pattern,this->mPGMScale).to(torch::kInt16);
        std::cout<<"First View: "<<firstView[0]<<" "<<firstView[1]<<" View Size: "<<viewSize[0]<<" "<<viewSize[1]<<std::endl;
        this->data = this->data.index({at::indexing::Slice({firstView[0],firstView[0]+viewSize[0]}),at::indexing::Slice({firstView[1],firstView[1]+viewSize[1]}),at::indexing::Slice(),at::indexing::Slice()});
        // computeGradients();
}
void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, char readOrWriteLightField, std::array<int64_t,2> firstView, std::array<int64_t,2> stride) {
    if(readOrWriteLightField == 'r'){
        this->data = io::read_collection(rootPath, pattern,this->mPGMScale).to(torch::kInt16);
        //computeGradients();
    }else{
        if(readOrWriteLightField == 'w'){
            io::write_collection(rootPath,this->data,firstView,stride);
        }
    }
}



Block4D_ LightField::ReadBlock4DfromLightField_(std::array<int64_t,4>size,std::array<int64_t,4>position, int64_t channel){
    
    Block4D_ block(size,position,this);
    std::array<int64_t,4> actualSize;
    for(int n = 0; n<4; n++){
        actualSize[n] = std::min(data.size(n) - position[n],size[n]);
    }
        //std::cout<<actualSize[0]<<" "<<actualSize[1]<<" "<<actualSize[2]<<" "<<actualSize[3]<<" "<<std::endl;


    block.data.index({at::indexing::Slice(0,actualSize[0]),
                                             at::indexing::Slice(0,actualSize[1]),
                                             at::indexing::Slice(0,actualSize[2]),
                                             at::indexing::Slice(0,actualSize[3])}) 
                         = this->data.index({at::indexing::Slice(position[0],position[0]+actualSize[0]),
                                             at::indexing::Slice(position[1],position[1]+actualSize[1]),
                                             at::indexing::Slice(position[2],position[2]+actualSize[2]),
                                             at::indexing::Slice(position[3],position[3]+actualSize[3]),
                                             channel});
    if (this->preSlantTan != 0) {
        
        at::Tensor validPosition_h = Block4D_::get_valid_position(this->preSlantTan,{this->data.size(0),this->data.size(1),this->data.size(2),this->data.size(3)},size,{position[0],position[1],position[2],position[3]}, true);
        at::Tensor validPosition_v = Block4D_::get_valid_position(this->preSlantTan,{this->data.size(0),this->data.size(1),this->data.size(2),this->data.size(3)},size,{position[0],position[1],position[2],position[3]}, false);
        block.validPositions = ValidPositions{validPosition_h,validPosition_v};
        if(validPosition_h.size(0) == size[1] * size[3] && validPosition_v.size(0) == size[0] * size[2]){
                std::cout<<"All positions are valid for subblock copy. "<<std::endl;
                block.includesInvalidCorners = false;
        }
        else{
            std::cout<<"Some positions are invalid for subblock copy. "<<std::endl;
            std::cout<<"Valid Positions Horizontal: "<<validPosition_h.sizes()<<"/"<<size[1] * size[3]<<std::endl;
            std::cout<<"Valid Positions Vertical: "<<validPosition_v.sizes()<<"/"<<size[0] * size[2]<<std::endl;
            block.includesInvalidCorners = true;
        }
    }
    else{
        block.validPositions = ValidPositions{at::empty({0}),at::empty({0})};
        block.includesInvalidCorners = false;
    }
    std::cout<<"Includes Invalid Corners: "<<block.includesInvalidCorners<<std::endl;
    return block;
}
void LightField :: WriteBlock4DtoLightField_(Block4D_ sourceBlock, std::array<int64_t,5> position){
    std::array<int64_t,4> length = {std::min(this->data.size(0) - position[0],sourceBlock.data.size(0)),
                                    std::min(this->data.size(1) - position[1],sourceBlock.data.size(1)),
                                    std::min(this->data.size(2) - position[2],sourceBlock.data.size(2)),
                                    std::min(this->data.size(3) - position[3],sourceBlock.data.size(3))};
    this->data.index({at::indexing::Slice(position[0],position[0]+length[0]),
                     at::indexing::Slice(position[1],position[1]+length[1]),
                     at::indexing::Slice(position[2],position[2]+length[2]),
                     at::indexing::Slice(position[3],position[3]+length[3]),
                     position[4]}) = sourceBlock.data.index({at::indexing::Slice(0,length[0]),at::indexing::Slice(0,length[1]),at::indexing::Slice(0,length[2]),at::indexing::Slice(0,length[3])});
                    


}

void LightField::slantLightField(double slope){
    if (this->data.dim() != 5) {
        throw std::runtime_error("LightField data must be 5D");
    }
    std::cout<<"Slanting Light Field of size: "<<this->data.sizes()<<" with slope: "<<slope<<std::endl;
    this->data = slantData(this->data, slope, this->mPGMScale);
    this->preSlantTan = slope;
    this ->secondHalfBias = this->data.size(2);
    std::cout<<"Second Half Bias after slanting: "<<this->secondHalfBias<<std::endl;
    //computeGradients();
}

void LightField::slantLightFieldBack(){
    if (this->data.dim() != 5) {
        throw std::runtime_error("LightField data must be 5D");
    }

    this->data = unslantData(this->data, this->preSlantTan);
    std::cout<<"sloped back size:"<<this->data.sizes()<<std::endl;

}

at::Tensor LightField::unslantData(const at::Tensor& block, double slantSlope){
    if(slantSlope == 0) return block;
    auto oldSize = block.sizes();
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble);
    int size_decrease_v = (int)(abs(round(slantSlope*(oldSize[0]-1))));
    int size_decrease_h = (int)(abs(round(slantSlope*(oldSize[1]-1))));
   
    at::Tensor new_block = torch::zeros({(int)oldSize[0],(int)oldSize[1],(int)oldSize[2]-size_decrease_v,(int)oldSize[3]-size_decrease_h,oldSize[4]},options);
    auto size = new_block.sizes();
    double true_alpha_v = slantSlope/abs(slantSlope) * (double)size_decrease_v/((double)size[0]-1);
    double true_alpha_h = slantSlope/abs(slantSlope) * (double)size_decrease_h/((double)size[1]-1);
    for (int c = 0; c < size[4]; c++){

    
        for(int l_ = 0; l_<size[0]; l_++){

            int n_start = floor(l_*true_alpha_v);
            int n_end = (size[2]) + floor(l_*true_alpha_v);

            if(slantSlope < 0){
                n_start -= (size[0]-1)*true_alpha_v;
                n_end -= (size[0]-1)*true_alpha_v;
            }

            for(int k_ = 0; k_ < size[1]; k_++){
                int m_start = floor(k_*true_alpha_h);
                int m_end = (size[3]) + floor(k_*true_alpha_h);
                if(slantSlope < 0){
                    m_start -= (size[1]-1)*true_alpha_h;
                    m_end -= (size[1]-1)*true_alpha_h;
                }
                new_block.index({l_,k_,torch::indexing::Slice(),torch::indexing::Slice(),c}) = block.index({l_,k_,torch::indexing::Slice(n_start,n_end),torch::indexing::Slice(m_start,m_end),c});              
            }
        }
    }
    return new_block;
}
 
at::Tensor LightField::slantData(const at::Tensor& block, double slantSlope, int PGMScale){
    if(slantSlope == 0) return block;
    auto size = block.sizes();
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble);
    int size_increase_v = (int)(abs(round(slantSlope*(size[0]-1))));
    int size_increase_h = (int)(abs(round(slantSlope*(size[1]-1))));
    double fill_value = (PGMScale + 1) / 2.0;
    at::Tensor new_block = torch::full({(int)size[0],(int)size[1],(int)size[2]+size_increase_v,(int)size[3]+size_increase_h,size[4]}, fill_value, options);
    double true_alpha_v = slantSlope/abs(slantSlope) * (double)size_increase_v/((double)size[0]-1);
    double true_alpha_h = slantSlope/abs(slantSlope) * (double)size_increase_h/((double)size[1]-1);
    for (int c = 0; c < size[4]; c++){

    
        for(int l_ = 0; l_<size[0]; l_++){

            int n_start = floor(l_*true_alpha_v);
            int n_end = (size[2]) + floor(l_*true_alpha_v);

            if(slantSlope < 0){
                n_start -= (size[0]-1)*true_alpha_v;
                n_end -= (size[0]-1)*true_alpha_v;
            }

            for(int k_ = 0; k_ < size[1]; k_++){
                int m_start = floor(k_*true_alpha_h);
                int m_end = (size[3]) + floor(k_*true_alpha_h);
                if(slantSlope < 0){
                    m_start -= (size[1]-1)*true_alpha_h;
                    m_end -= (size[1]-1)*true_alpha_h;
                }
                new_block.index({l_,k_,torch::indexing::Slice(n_start,n_end),torch::indexing::Slice(m_start,m_end),c}) = block.index({l_,k_,torch::indexing::Slice(),torch::indexing::Slice(),c});              
            }
        }
    }
    return new_block;
}
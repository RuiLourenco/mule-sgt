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

// Function to compute partial derivatives using Gaussian filters
torch::Tensor compute_first_order_derivatives(const torch::Tensor& input, int kernel_size = 5, float sigma = 1.0) {
    // Check if input is 4D
    if (input.dim() != 4) {
        throw std::runtime_error("Input tensor must be 4D");
    }
    
    // Get input dimensions
    auto dims = input.sizes();
    int64_t dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim3 = dims[3];
    
    // Create Gaussian and derivative kernels
    torch::Tensor gaussian_kernel = create_gaussian_kernel(kernel_size, sigma);
    torch::Tensor derivative_kernel = create_gaussian_derivative_kernel(kernel_size, sigma);
    
    // Initialize the 5D output tensor (4D input + 1D for derivatives)
    torch::Tensor output = torch::zeros({dim0, dim1, dim2, dim3, 4}, input.options());
    
    // Helper function to apply 1D convolution along specific dimension
    auto apply_1d_filter = [&](const torch::Tensor& tensor, const torch::Tensor& filter, int dim) -> torch::Tensor {
        // Create padding configuration
        int pad_size = filter.size(0) / 2;
        std::vector<int64_t> padding(2 * tensor.dim(), 0);
        padding[2 * dim] = pad_size;
        padding[2 * dim + 1] = pad_size;
        
        // Pad the tensor
        auto padded = torch::nn::functional::pad(tensor, torch::nn::functional::PadFuncOptions(padding));
        
        // Output tensor
        auto result = tensor.clone();
        
        // For each position in the output
        torch::NoGradGuard no_grad;
        
        // Use different approaches based on dimension
        if (dim == 0) {
            for (int64_t i = 0; i < dim0; i++) {
                // Apply filter
                for (int64_t k = 0; k < filter.size(0); k++) {
                    int64_t idx = i + k - pad_size;
                    if (idx >= 0 && idx < dim0) {
                        result[i] += padded[idx] * filter[k].item<float>();
                    }
                }
            }
        } else if (dim == 1) {
            for (int64_t b = 0; b < dim0; b++) {
                for (int64_t i = 0; i < dim1; i++) {
                    // Apply filter
                    for (int64_t k = 0; k < filter.size(0); k++) {
                        int64_t idx = i + k - pad_size;
                        if (idx >= 0 && idx < dim1) {
                            result[b][i] += padded[b][idx] * filter[k].item<float>();
                        }
                    }
                }
            }
        } else if (dim == 2) {
            for (int64_t b = 0; b < dim0; b++) {
                for (int64_t c = 0; c < dim1; c++) {
                    for (int64_t i = 0; i < dim2; i++) {
                        // Apply filter
                        for (int64_t k = 0; k < filter.size(0); k++) {
                            int64_t idx = i + k - pad_size;
                            if (idx >= 0 && idx < dim2) {
                                result[b][c][i] += padded[b][c][idx] * filter[k].item<float>();
                            }
                        }
                    }
                }
            }
        } else if (dim == 3) {
            for (int64_t b = 0; b < dim0; b++) {
                for (int64_t c = 0; c < dim1; c++) {
                    for (int64_t h = 0; h < dim2; h++) {
                        for (int64_t i = 0; i < dim3; i++) {
                            // Apply filter
                            for (int64_t k = 0; k < filter.size(0); k++) {
                                int64_t idx = i + k - pad_size;
                                if (idx >= 0 && idx < dim3) {
                                    result[b][c][h][i] += padded[b][c][h][idx] * filter[k].item<float>();
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    };
    
    // Compute derivatives for each dimension
    for (int dim = 0; dim < 4; dim++) {
        // Start with the input tensor
        torch::Tensor processed = input.clone();
        
        // Smooth all dimensions except the current one
        for (int other_dim = 0; other_dim < 4; other_dim++) {
            if (other_dim != dim) {
                processed = apply_1d_filter(processed, gaussian_kernel, other_dim);
            }
        }
        
        // Apply derivative kernel along the current dimension
        torch::Tensor derivative = apply_1d_filter(processed, derivative_kernel, dim);
        
        // Store in the output tensor
        for (int64_t i = 0; i < dim0; i++) {
            for (int64_t j = 0; j < dim1; j++) {
                for (int64_t k = 0; k < dim2; k++) {
                    for (int64_t l = 0; l < dim3; l++) {
                        output[i][j][k][l][dim] = derivative[i][j][k][l];
                    }
                }
            }
        }
    }
    
    return output;
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

// Function to compute partial derivatives using libtorch functionality
torch::Tensor compute_first_order_derivatives_libtorch(const torch::Tensor& input, int kernel_size = 5, float sigma = 1.0) {
    // Check if input is 4D
    if (input.dim() != 4) {
        throw std::runtime_error("Input tensor must be 4D");
    }

    // Create Gaussian and derivative kernels
    torch::Tensor gaussian_kernel = create_gaussian_kernel(kernel_size, sigma).view({1, 1, -1});
    torch::Tensor derivative_kernel = create_gaussian_derivative_kernel(kernel_size, sigma).view({1, 1, -1});

    // Apply Gaussian smoothing
    torch::Tensor smoothed = input.view({1, 1, -1, input.size(3)});
    smoothed = torch::conv1d(smoothed, gaussian_kernel, {}, 1, kernel_size / 2);
    smoothed = smoothed.view(input.sizes());

    // Initialize the output tensor
    torch::Tensor output = torch::zeros({input.size(0), input.size(1), input.size(2), input.size(3), 4}, input.options());

    // Compute derivatives for each dimension
    for (int dim = 0; dim < 4; dim++) {
        // Permute dimensions to bring the current dimension to the last
        auto permuted = smoothed.permute({dim, (dim + 1) % 4, (dim + 2) % 4, (dim + 3) % 4}).contiguous();
        permuted = permuted.view({1, 1, -1, permuted.size(3)});

        // Apply derivative kernel
        torch::Tensor derivative = torch::conv1d(permuted, derivative_kernel, {}, 1, kernel_size / 2);
        derivative = derivative.view(input.sizes());

        // Store in the output tensor
        output.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), dim}, derivative);
    }

    return output;
}


at::Tensor central_diff(const at::Tensor& tensor, int64_t axis) {
    return (tensor.roll(-1, {axis}) - tensor.roll(1, {axis})) / 2.0;
}

at::Tensor compute_partial_derivatives(const at::Tensor& tensor) {
   
    // Computes discrete partial derivatives of a 4D tensor using central differences.
    // Assumes the tensor has shape (N, C, H, W) where:
    // - N: Batch size
    // - C: Number of channels
    // - H: Height
    // - W: Width
    
    // Returns a 5D tensor of shape (5, N, C, H, W), where the first dimension corresponds
    // to derivatives along [dt, dz, dy, dx].
    
    at::Tensor dt = central_diff(tensor, 0);  // Time derivative
    at::Tensor dz = central_diff(tensor, 1);  // Channel depth derivative
    at::Tensor dy = central_diff(tensor, 2);  // Height derivative
    at::Tensor dx = central_diff(tensor, 3);  // Width derivative
    
    return torch::stack({dt, dz, dy, dx}, 4); // Stack along a new dimension
}



// Function to verify correctness with autograd
torch::Tensor verify_with_autograd(const torch::Tensor& input) {
    // Clone and set requires grad
    torch::Tensor x = input.clone().detach().requires_grad_(true);
    
    // Get dimensions
    auto dims = x.sizes();
    int64_t dim0 = dims[0], dim1 = dims[1], dim2 = dims[2], dim3 = dims[3];
    
    // Output tensor to store derivatives
    torch::Tensor output = torch::zeros({dim0, dim1, dim2, dim3, 4}, x.options());
    
    // For each element in the tensor, compute partial derivatives
    for (int64_t i = 0; i < dim0; i++) {
        for (int64_t j = 0; j < dim1; j++) {
            for (int64_t k = 0; k < dim2; k++) {
                for (int64_t l = 0; l < dim3; l++) {
                    // Get the scalar at this position
                    torch::Tensor scalar = x[i][j][k][l];
                    
                    // Zero out existing gradients
                    if (x.grad().defined()) {
                        x.grad().zero_();
                    }
                    
                    // Backpropagate
                    scalar.backward(torch::Tensor(), true);
                    
                    // Get gradient
                    torch::Tensor grad = x.grad().clone();
                    
                    // Store in output tensor
                    for (int dim = 0; dim < 4; dim++) {
                        output[i][j][k][l][dim] = grad[i][j][k][l];
                    }
                }
            }
        }
    }
    
    return output;
}


/*******************************************************************************/
/*                        LightField class methods                             */
/*******************************************************************************/

void LightField::computeGradients(){
    std::cout<<"Computing Gradients"<<std::endl;
    // this->gradients = compute_first_order_derivatives_libtorch(this->data.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}));
    // this->gradients = compute_partial_derivatives(this->data.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}));
    this->gradients = compute_first_order_derivatives_separable(this->data.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}),5,1.0);
    std::cout<<"Gradients Computed!"<<std::endl;
}
LightField::LightField(std::string root_path,std::string pattern) {
    OpenLightFieldPPM_(root_path,pattern,'r',{0,0});
   
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
}

void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, std::array<int64_t,2> firstView, std::array<int64_t,2> viewSize) {
        this->data = io::read_collection(rootPath, pattern,this->mPGMScale).to(torch::kInt16);
        std::cout<<"First View: "<<firstView[0]<<" "<<firstView[1]<<" View Size: "<<viewSize[0]<<" "<<viewSize[1]<<std::endl;
        this->data = this->data.index({at::indexing::Slice({firstView[0],firstView[0]+viewSize[0]}),at::indexing::Slice({firstView[1],firstView[1]+viewSize[1]}),at::indexing::Slice(),at::indexing::Slice()});
        //computeGradients();
}
void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, char readOrWriteLightField,std::array<int64_t,2> firstView = {0,0} ) {
    if(readOrWriteLightField == 'r'){
        this->data = io::read_collection(rootPath, pattern,this->mPGMScale).to(torch::kInt16);
        //computeGradients();
    }else{
        if(readOrWriteLightField == 'w'){
            io::write_collection(rootPath,this->data,firstView);
        }
    }
}



Block4D_ LightField::ReadBlock4DfromLightField_(std::array<int64_t,4>size,std::array<int64_t,4>position, int64_t channel){
    Block4D_ block(size,position,this);
    std::array<int64_t,4> actualSize;
    for(int n = 0; n<4; n++){
        actualSize[n] = std::min(data.size(n) - position[n],size[n]);
    }
        std::cout<<actualSize[0]<<" "<<actualSize[1]<<" "<<actualSize[2]<<" "<<actualSize[3]<<" "<<std::endl;


    block.data.index({at::indexing::Slice(0,actualSize[0]),
                                             at::indexing::Slice(0,actualSize[1]),
                                             at::indexing::Slice(0,actualSize[2]),
                                             at::indexing::Slice(0,actualSize[3])}) 
                         = this->data.index({at::indexing::Slice(position[0],position[0]+actualSize[0]),
                                             at::indexing::Slice(position[1],position[1]+actualSize[1]),
                                             at::indexing::Slice(position[2],position[2]+actualSize[2]),
                                             at::indexing::Slice(position[3],position[3]+actualSize[3]),
                                             channel});

    at::Tensor validPosition_h = Block4D_::get_valid_position(this->preSlantTan,{this->data.size(0),this->data.size(1),this->data.size(2),this->data.size(3)},size,{position[0],position[1],position[2],position[3]}, true);
    at::Tensor validPosition_v = Block4D_::get_valid_position(this->preSlantTan,{this->data.size(0),this->data.size(1),this->data.size(2),this->data.size(3)},size,{position[0],position[1],position[2],position[3]}, false);
    block.validPositions = ValidPositions{validPosition_h,validPosition_v};
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

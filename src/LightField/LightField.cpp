#include "LightField/LightField.h"
#include <string.h>
#include <stdlib.h>
#include "IO/io.h"


/******************************************************************** */
/*                         Gradient Calculations                      */    
/******************************************************************** */

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
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float x = i - center;
        float value = -x * std::exp(-(x * x) / (2 * sigma * sigma));
        kernel[i] = value;
        sum += std::abs(value);
    }
    
    // Normalize the kernel
    kernel /= sum;
    return kernel;
}

torch::Tensor compute_first_order_derivatives_efficient(const torch::Tensor& input, int kernel_size = 5, float sigma = 1.0) {
    // Check dimensions
    if (input.dim() != 4) {
        throw std::runtime_error("Input tensor must be 4D");
    }
    
    // Get dimensions
    int64_t dim0 = input.size(0);
    int64_t dim1 = input.size(1);
    int64_t dim2 = input.size(2);
    int64_t dim3 = input.size(3);
    
    // Create Gaussian and derivative kernels
    torch::Tensor gaussian_kernel = create_gaussian_kernel(kernel_size, sigma);
    torch::Tensor derivative_kernel = create_gaussian_derivative_kernel(kernel_size, sigma);
    
    // Prepare output tensor
    torch::Tensor output = torch::zeros({dim0, dim1, dim2, dim3, 4}, input.options());
    
    // For each dimension
    for (int dim = 0; dim < 4; dim++) {
        // Clone the input for processing
        torch::Tensor processed = input.clone();
        
        // Apply the appropriate reshaping and padding based on dimension
        if (dim == 0) {
            // Dimension 0 (batch dimension)
            processed = processed.permute({1, 0, 2, 3});
            processed = processed.reshape({1, dim1, dim0, dim2 * dim3});
            
            // Prepare kernels
            auto gauss_kernel = gaussian_kernel.reshape({1, 1, kernel_size});
            auto deriv_kernel = derivative_kernel.reshape({1, 1, kernel_size});
            
            // Apply convolution
            auto pad_size = kernel_size / 2;
            auto pad = torch::nn::functional::pad(processed, torch::nn::functional::PadFuncOptions({0, 0, pad_size, pad_size}));
            processed = torch::conv1d(pad, deriv_kernel, {}, 1);
            
            // Reshape back
            processed = processed.reshape({dim1, dim0, dim2, dim3});
            processed = processed.permute({1, 0, 2, 3});
        } 
        else if (dim == 1) {
            // Dimension 1
            processed = processed.reshape({dim0 * dim1, 1, dim2, dim3});
            
            // Prepare kernels for channel dimension
            auto gauss_kernel = gaussian_kernel.reshape({1, 1, kernel_size, 1});
            auto deriv_kernel = derivative_kernel.reshape({1, 1, kernel_size, 1});
            
            // Apply 2D convolution with 1D kernels
            auto pad_size = kernel_size / 2;
            auto pad = torch::nn::functional::pad(processed, torch::nn::functional::PadFuncOptions({0, 0, pad_size, pad_size}));
            processed = torch::conv2d(pad, deriv_kernel, {}, 1);
            
            // Reshape back
            processed = processed.reshape({dim0, dim1, dim2, dim3});
        }
        else if (dim == 2) {
            // Dimension 2 (height)
            processed = processed.permute({0, 3, 1, 2});
            processed = processed.reshape({dim0 * dim3, dim1, dim2, 1});
            
            // Prepare kernels
            auto gauss_kernel = gaussian_kernel.reshape({1, 1, kernel_size, 1});
            auto deriv_kernel = derivative_kernel.reshape({1, 1, kernel_size, 1});
            
            // Apply convolution
            auto pad_size = kernel_size / 2;
            auto pad = torch::nn::functional::pad(processed, torch::nn::functional::PadFuncOptions({0, 0, pad_size, pad_size}));
            processed = torch::conv2d(pad, deriv_kernel, {}, 1);
            
            // Reshape back
            processed = processed.reshape({dim0, dim3, dim1, dim2});
            processed = processed.permute({0, 2, 3, 1});
        }
        else {  // dim == 3
            // Dimension 3 (width)
            processed = processed.permute({0, 2, 1, 3});
            processed = processed.reshape({dim0 * dim2, dim1, 1, dim3});
            
            // Prepare kernels
            auto gauss_kernel = gaussian_kernel.reshape({1, 1, 1, kernel_size});
            auto deriv_kernel = derivative_kernel.reshape({1, 1, 1, kernel_size});
            
            // Apply convolution
            auto pad_size = kernel_size / 2;
            auto pad = torch::nn::functional::pad(processed, torch::nn::functional::PadFuncOptions({pad_size, pad_size, 0, 0}));
            processed = torch::conv2d(pad, deriv_kernel, {}, 1);
            
            // Reshape back
            processed = processed.reshape({dim0, dim2, dim1, dim3});
            processed = processed.permute({0, 2, 1, 3});
        }
        
        // Store result in output tensor
        output.index_put_({torch::indexing::Ellipsis, dim}, processed);
    }
    
    return output;
}


/*******************************************************************************/
/*                        LightField class methods                             */
/*******************************************************************************/

void LightField::computeGradients(){
    this->gradients = compute_first_order_derivatives_efficient(this->data);
}
LightField::LightField(std::string root_path,std::string pattern) {
    OpenLightFieldPPM_(root_path,pattern,'r');
    computeGradients();
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
    computeGradients();
}

void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, std::array<int64_t,2> firstView, std::array<int64_t,2> viewSize) {
        this->data = io::read_collection(rootPath, pattern,this->mPGMScale).to(torch::kInt16);
        std::cout<<"First View: "<<firstView[0]<<" "<<firstView[1]<<" View Size: "<<viewSize[0]<<" "<<viewSize[1]<<std::endl;
        write_tensor(this->data[0][0],"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/firstViewBeforeTrim.png",{0,1024});
        this->data = this->data.index({at::indexing::Slice({firstView[0],firstView[0]+viewSize[0]}),at::indexing::Slice({firstView[1],firstView[1]+viewSize[1]}),at::indexing::Slice(),at::indexing::Slice()});
        write_tensor(this->data[0][0],"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/firstViewAfterTrim.png",{0,1024});
        computeGradients();
}
void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, char readOrWriteLightField ) {
    if(readOrWriteLightField == 'r'){
        this->data = io::read_collection(rootPath, pattern,this->mPGMScale).to(torch::kInt16);
    }else{
        if(readOrWriteLightField == 'w'){
            io::write_collection(rootPath,this->data);
        }
    }
}



Block4D_ LightField::ReadBlock4DfromLightField_(std::array<int64_t,4>size,std::array<int64_t,4>position, int64_t channel){
    Block4D_ block(size);
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

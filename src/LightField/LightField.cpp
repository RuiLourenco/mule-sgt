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



/*******************************************************************************/
/*                        LightField class methods                             */
/*******************************************************************************/

void LightField::computeGradients(){
    std::cout<<"Computing Gradients"<<std::endl;
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
    this->data = slantData(this->data, slope);
    std::cout<<"sloped size:"<<this->data.sizes()<<std::endl;
    at::Tensor view = this->data[4][4];
    at::Tensor epi = this->data.index({torch::indexing::Slice(), 4, torch::indexing::Slice() ,120, 0});
    std::cout<<"Slanted View Size: "<<view.sizes()<<std::endl;
    write_tensor(view.index({torch::indexing::Slice(),torch::indexing::Slice(),0}), "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-pre-slant/results/Greek/slanted_light_field.png");
    write_tensor(epi, "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt-pre-slant/results/Greek/slanted_epi.png");
    std::cout<<"Slanted Light Field with slope: "<<slope<<std::endl;
    this->preSlantTan = slope;
}
 
at::Tensor LightField::slantData(const at::Tensor& block, double slantSlope){
    if(slantSlope == 0) return block;
    auto size = block.sizes();
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble);
    int size_increase = (int)(abs(round(slantSlope*(size[0]-1))));
    at::Tensor new_block = torch::zeros({(int)size[0],(int)size[1],(int)size[2]+size_increase,(int)size[3]+size_increase,size[4]},options);
    double true_alpha = slantSlope/abs(slantSlope) * (double)size_increase/((double)size[0]-1);
    for (int c = 0; c < size[4]; c++){

    
        for(int l_ = 0; l_<size[0]; l_++){

            int n_start = floor(l_*true_alpha);
            int n_end = (size[2]) + floor(l_*true_alpha);

            if(slantSlope < 0){
                n_start -= (size[0]-1)*true_alpha;
                n_end -= (size[0]-1)*true_alpha;
            }

            for(int k_ = 0; k_ < size[1]; k_++){
                int m_start = floor(k_*true_alpha);
                int m_end = (size[3]) + floor(k_*true_alpha);
                if(slantSlope < 0){
                    m_start -= (size[1]-1)*true_alpha;
                    m_end -= (size[1]-1)*true_alpha;
                }
                new_block.index({l_,k_,torch::indexing::Slice(n_start,n_end),torch::indexing::Slice(m_start,m_end),c}) = block.index({l_,k_,torch::indexing::Slice(),torch::indexing::Slice(),c});              
            }
        }
    }
    return new_block;
}
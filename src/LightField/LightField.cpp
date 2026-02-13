#include "LightField/LightField.h"
#include "LightField/Block4D.h"

#include <string.h>
#include <stdlib.h>
#include "IO/io.h"



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
}
void LightField :: OpenLightFieldPPM_(std::string rootPath, std::string pattern, char readOrWriteLightField, std::array<int64_t,2> firstView, std::array<int64_t,2> stride) {
    if(readOrWriteLightField == 'r'){
        this->data = io::read_collection(rootPath, pattern,this->mPGMScale).to(torch::kInt16);
    }else{
        if(readOrWriteLightField == 'w'){
            io::write_collection(rootPath,this->data,firstView,stride);
        }
    }
}



Block4D LightField::ReadBlock4DfromLightField_(std::array<int64_t,4>size,std::array<int64_t,4>position, int64_t channel){
    
    Block4D block(size,position,this);
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
        
        at::Tensor validPosition_h = Block4D::get_valid_position(this->preSlantTan,{this->data.size(0),this->data.size(1),this->data.size(2),this->data.size(3)},size,{position[0],position[1],position[2],position[3]}, true);
        at::Tensor validPosition_v = Block4D::get_valid_position(this->preSlantTan,{this->data.size(0),this->data.size(1),this->data.size(2),this->data.size(3)},size,{position[0],position[1],position[2],position[3]}, false);
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
void LightField :: WriteBlock4DtoLightField_(Block4D sourceBlock, std::array<int64_t,5> position){
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
    this->data = slantData(this->data, slope);
    this->preSlantTan = slope;
    this ->secondHalfBias = this->data.size(2);
    std::cout<<"Second Half Bias after slanting: "<<this->secondHalfBias<<std::endl;
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
 
at::Tensor LightField::slantData(const at::Tensor& block, double slantSlope){
    if(slantSlope == 0) return block;
    auto size = block.sizes();
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kDouble);
    int size_increase_v = (int)(abs(round(slantSlope*(size[0]-1))));
    int size_increase_h = (int)(abs(round(slantSlope*(size[1]-1))));
    at::Tensor new_block = torch::zeros({(int)size[0],(int)size[1],(int)size[2]+size_increase_v,(int)size[3]+size_increase_h,size[4]},options);
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
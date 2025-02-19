#include "LightField/LightField.h"
#include <string.h>
#include <stdlib.h>
#include "IO/io.h"

/*******************************************************************************/
/*                        LightField class methods                             */
/*******************************************************************************/
LightField::LightField(std::string root_path,std::string pattern) {
    OpenLightFieldPPM_(root_path,pattern,'r');
    this->mNumberOfHorizontalViews = data.size(1);
    this->mNumberOfVerticalViews = data.size(0);
    this->mNumberOfViewLines = data.size(3);
    this->mNumberOfViewColumns = data.size(2);
}
LightField :: LightField(std::array<int64_t,5> size){
    this->data = at::zeros({size[0],size[1],size[2],size[3],size[4]},at::kInt);
    mViewFileNamePrefix = NULL;
    mViewFileNameSuffix = NULL;
    mNumberOfHorizontalViews = size[1];
    mNumberOfVerticalViews = size[0];
    mNumberOfCacheHorizontalViews = size[1];
    mNumberOfCacheVerticalViews = size[0];
    
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
    
    this->data.index({at::indexing::Slice(position[0],position[0]+sourceBlock.data.size(0)),
                     at::indexing::Slice(position[1],position[1]+sourceBlock.data.size(1)),
                     at::indexing::Slice(position[2],position[2]+sourceBlock.data.size(2)),
                     at::indexing::Slice(position[3],position[3]+sourceBlock.data.size(3)),
                     position[4]}) = sourceBlock.data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()});
                    


}



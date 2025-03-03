#include <torch/torch.h>
#include "View.h"
#include "Block4D_.h"
#include <array>



#ifndef LIGHTFIELD_H
#define LIGHTFIELD_H


class LightField {
public:    
    LightField() = default;
    LightField(std::string rooth_path,std::string pattern);
    LightField(std::array<int64_t,5> sizes);
    void computeGradients();
    int preSlantTan = 0;
    int mPGMScale;                      /*!< scale of the pgm files*/
    at::Tensor data;    
    at::Tensor gradients;              /*!< Pytorch Tensor*/
    void OpenLightFieldPPM_(std::string rootPath, std::string pattern, std::array<int64_t,2> firstView, std::array<int64_t,2> viewSize);
    void OpenLightFieldPPM_(std::string path,std::string pattern,char readOrWriteLightField);
    Block4D_ ReadBlock4DfromLightField_(std::array<int64_t,4> size, std::array<int64_t,4> position_t,int64_t channel );
    void WriteBlock4DtoLightField_(Block4D_ sourceBlock, std::array<int64_t,5> position);
};

#endif


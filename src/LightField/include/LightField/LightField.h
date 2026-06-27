#include <torch/torch.h>
#include "View.h"
#include <array>



#ifndef LIGHTFIELD_H
#define LIGHTFIELD_H

class Block4D_;
class LightField {
public:    
    LightField() = default;
    LightField(std::string rooth_path,std::string pattern);
    LightField(std::array<int64_t,5> sizes);
    void computeGradients();
    void computeTopHalfGradients();
    void computeBottomHalfGradients();
    int preSlantTan = 0;
    int mPGMScale;                      /*!< scale of the pgm files*/
    at::Tensor data;    
    at::Tensor gradients;              /*!< Pytorch Tensor*/
    bool secondHalfGradientsComputed = false;
    int secondHalfBias = 0;
    void OpenLightFieldPPM_(std::string rootPath, std::string pattern, std::array<int64_t,2> firstView, std::array<int64_t,2> viewSize);
    void OpenLightFieldPPM_(std::string path, std::string pattern, char readOrWriteLightField, std::array<int64_t,2> firstView = {0,0}, std::array<int64_t,2> stride = {1,1});
    Block4D_ ReadBlock4DfromLightField_(std::array<int64_t,4> size, std::array<int64_t,4> position_t,int64_t channel );
    void WriteBlock4DtoLightField_(Block4D_ sourceBlock, std::array<int64_t,5> position);
    void slantLightField(double slope);
    static at::Tensor slantData(const at::Tensor& block, double slantSlope, int PGMScale = 1023);
    void slantLightFieldBack();
    static at::Tensor unslantData(const at::Tensor& block, double slantSlope);


};

#endif


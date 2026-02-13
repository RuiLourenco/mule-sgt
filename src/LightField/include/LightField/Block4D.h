#ifndef BLOCK4D__H
#define  BLOCK4D__H

#define block4DElementType int

#define PI 3.141592653589793
#include <torch/torch.h>
#include "LightField.h"
#include <array>
#include <nlohmann/json.hpp>

#define ADAPTIVE_RHO_CALC 1
#define FLAT_TRANSFORM 1


void write_tensor(torch::Tensor tensor, std::string path, std::array<double,2> valueRange = {1,1});
struct ValidPositions{
    at::Tensor valid_positions_h;
    at::Tensor valid_positions_v;
};

class Block4D 
{
private:

    at::Tensor getFlatBlockValid();
    at::Tensor getFlatBlockAll();
    at::Tensor flat24DAll(const at::Tensor& flatBlock) const;
    at::Tensor flat24DValid(const at::Tensor& flatBlock) const;
    static at::Tensor klt(at::Tensor covMat, at::Tensor& eigVals);
    at::Tensor matrixTransform(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV, const at::Tensor& eigValsH, const at::Tensor& eigValsV) ;
    static at::Tensor iMatrixTransform(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) ;
public: 

    LightField* lightField = nullptr;
    at::Tensor autoCorr(bool isHorizontal);
    at::Tensor eigenValuesH = at::empty({0});
    at::Tensor eigenValuesV = at::empty({0});

    void kltTransform(double scale);
    at::Tensor ikltTransformData(double scale, at::Tensor covH, at::Tensor covV);
    void ikltTransform(double scale,at::Tensor covH, at::Tensor covV);
    at::Tensor getTransformMatrix(const at::Tensor& cov, bool isHorizontal,at::Tensor& eigVals) const;



    std::array<int64_t,4> size;
    std::array<int64_t,4> transformSize;
    std::array<int64_t,4> lightFieldPosition;
    at::Tensor data;
    ValidPositions validPositions;
    bool includesInvalidCorners = false;
    bool sgtDomain = false;
    static void YCoCg2RGB(Block4D &R, Block4D &G, Block4D &B, Block4D const &Y, Block4D const &Co, Block4D const &Cg, int Scale);
    static void YCbCr2RGB_BT601(Block4D &R, Block4D &G, Block4D &B, Block4D const &Y, Block4D const &Cb, Block4D const &Cr, int Scale);
    static void RGB2YCbCr_BT601(Block4D &Y, Block4D &Cb, Block4D &Cr, Block4D const &R, Block4D const &G, Block4D const &B, int Scale);
    static void RGB2YCoCg(Block4D &Y, Block4D &Co, Block4D &Cg, Block4D const &R, Block4D const &G, Block4D const &B, int Scale);
    static at::Tensor normalizeCov(at::Tensor cov);
    static at::Tensor get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal);
    at::Tensor getFlatBlock();
    at::Tensor flat24D(const at::Tensor& flatBlock) const;
    
    at::Tensor flatBlockFrom4DTensor(at::Tensor coefficients);
    void emptyTransform();
    at::Tensor batchedCovMatrix(bool isHorizontal) const;
    std::array<int64_t,4> toTransformCoords(std::array<int64_t,4> coords) const;

    void ikltTransform(Block4D reconstructedBlock, double scale);

    operator at::Tensor() const;
    //operator const at::Tensor&() const;
    Block4D() = default;
    //Block4D(at::Tensor data);
    //Block4D(const at::Tensor& data);
    //Block4D(at::Tensor& data);
    Block4D(std::array<int64_t,4> size,std::array<int64_t,4> lightFieldSize, LightField* lightField);
    Block4D(const Block4D& B00, const Block4D& B01, const Block4D& B10, const Block4D& B11, bool views);
    Block4D copySubblock(std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset);
    void Shift_UVPlane(int shift, int position_t, int position_s);
    void Ones(void);
    void Zeros(void);
    

    int computePreviousInvalidNumber(double preSlantTan,int parentBlockN, int subblockN,bool isHorizontal) const;
    void copySubblockData(Block4D& destination, std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset) const;
    std::vector<int64_t> copyValidSubblockPositions(std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset, bool isHorizontal);


    

    Block4D operator + (const Block4D &B) const;
    Block4D operator * (const Block4D &B) const;
    Block4D operator - (const Block4D &B) const;
    Block4D operator + (const at::Tensor &B) const;
    Block4D operator * (const at::Tensor &B) const;
    Block4D operator - (const at::Tensor &B) const;

    friend Block4D operator + (const int a,const Block4D & B);
    friend Block4D operator - (const int a,const Block4D & B);
    friend Block4D operator * (const int a,const Block4D & B);
    Block4D operator + (const int a) const;
    Block4D operator * (const int a) const;
    Block4D operator - (const int a) const;
    Block4D operator / (const int a) const;
    Block4D operator / (const double a) const;
   
    void operator += (const Block4D &B);
    void operator *= (const Block4D &B);
    void operator -= (const Block4D &B);
    void operator = (const Block4D &B);
    void operator = (Block4D* B);
    Block4D clone() const;

    ~Block4D() = default;
    
    void CopySubblockFrom(const Block4D &B, std::array<int64_t,4> sourceOffset, std::array<int64_t,4> targetOffset); 


    void Display(void);
    double L2Norm(void);
    void Extend_U(int position_u);
    void Extend_V(int position_v);
    void Extend_S(int position_s);
    void Extend_T(int position_t);
    void clip(int minValue, int maxValue);
    void Threshold(int minMagnitude, int maxMagnitude);

	short GetPixel(int position_t, int position_s, int position_v, int position_u) {
		return(data[position_t][position_s][position_v][position_u].item<short>());
	}

	void SetPixel(block4DElementType pixel_value, int position_t, int position_s, int position_v, int position_u) {
		data[position_t][position_s][position_v][position_u] = pixel_value;
    }

	long int LinearPosition(long int position_t, long int position_s, long int position_v, long int position_u) {
		long int linear_position = position_t*data.size(3)*data.size(2)*data.size(1);
		linear_position += position_s*data.size(3)*data.size(2) + position_v*data.size(3) + position_u;
		return(linear_position);
	}
    void CoordPosition(long int index, long int& position_t, long int& position_s, long int& position_v, long int& position_u) {
        position_t = index/(data.size(3)*data.size(2)*data.size(1));
        position_s = (index - position_t*data.size(3)*data.size(2)*data.size(1))/(data.size(3)*data.size(2));
        position_v = (index - position_t*data.size(3)*data.size(2)*data.size(1) - position_s*data.size(3)*data.size(2))/data.size(3);
        position_u = index - position_t*data.size(3)*data.size(2)*data.size(1) - position_s*data.size(3)*data.size(2) - position_v*data.size(3);
    }

};

#endif
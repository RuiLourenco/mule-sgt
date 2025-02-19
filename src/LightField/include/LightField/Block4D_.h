#ifndef BLOCK4D__H
#define  BLOCK4D__H

#define block4DElementType int

#define PI 3.141592653589793
#include <torch/torch.h>
#include <array>

class Block4D_;
void write_tensor(torch::Tensor tensor, std::string path, std::array<double,2> valueRange = {1,1});
struct ValidPositions{
    at::Tensor valid_positions_h;
    at::Tensor valid_positions_v;
};

class SgtSideInfo{
    
    int angleVInt;
    int angleHInt;
    std::array<double,2> angleRange;

    static int pruneDouble(double dNumber,int factor);
    static double recoverDouble(int iNumber,int factor);
    double DecodeAngle(int AngleCode) const;
    int codeAngle(double angle) const;


    public:
        static constexpr double PRECISION_ANGLE = 0.5;  
        void print();
        SgtSideInfo(const Block4D_& block,std::array<double,2> dispRange);
        SgtSideInfo(int angleVInt,int angleHInt, std::array<double,2> dispRange);
        SgtSideInfo(std::array<double,2> dispRange);
        SgtSideInfo(double angleV,double angleH, std::array<double,2> dispRange);
        at::Tensor QPOptimization(at::Tensor P, at::Tensor q);
        at::Tensor constrainedLeastSquares(at::Tensor A, at::Tensor b);

        std::array<double,2> disparityRange;

        int getRhoPrecision() const;
        int getAnglePrecision() const;

        double getAngleCodeScale() const;
        double getAngleCodeBias() const;
        double getDisparityV() const;
        double getDisparityH() const;
        double getAngleV() const;
        double getAngleH() const;
        int getAngleVCode() const;
        int getAngleHCode() const;
        void setAngleVCode(int code);
        void setAngleHCode(int code);
        
        void setAngleV(double angle);
        void setAngleH(double angle);
        void setAngleVFromDisparity(double disparity);
        void setAngleHFromDisparity(double disparity);
        void estimateDisparity(const Block4D_& block);
        SgtSideInfo() = default;
        void estimateAngleFromMonotony(Block4D_ block);


    private:
        std::array<double,2> angleRangeFromDispRange(std::array<double,2> dispRange);

        static double genDivergence(const at::Tensor& modelCovMat, const at::Tensor& iSqrtCovMat);

};
class Block4D_ 
{
private:

    at::Tensor getFlatBlockValid();
    at::Tensor getFlatBlockAll();
    at::Tensor flat24DAll(const at::Tensor& flatBlock) const;
    at::Tensor flat24DValid(const at::Tensor& flatBlock) const;
    static at::Tensor klt(at::Tensor covMat, at::Tensor& eigVals);
    at::Tensor sgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV, const at::Tensor& eigValsH, const at::Tensor& eigValsV) ;
    static at::Tensor isgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) ;

public:

    std::array<int64_t,4> size;
    std::array<int64_t,4> transformSize;
    SgtSideInfo ssi;
     at::Tensor data;
    ValidPositions validPositions;
    bool includesInvalidCorners = false;
    bool sgtDomain = false;

    at::Tensor getSgtTransformMatrix(const at::Tensor& cov, bool isHorizontal,at::Tensor& eigVals) const;
    at::Tensor iSqrtCovMat(at::Tensor covMat, bool isHorizontal ) const;

    at::Tensor get2DBasis(at::Tensor sgtMatrix, int n) const;
    std::array<double,2> getMainFrequency(at::Tensor basisFunction,double angle,int index) const;
    at::Tensor frequencyOrderedSgt(at::Tensor flatBlock, at::Tensor sgtMatrixH, at::Tensor sgtMatrixV);
    void view4DFrequencies(at::Tensor fullOrdinalFrequencies, at::Tensor frequencies) const;
    at::Tensor getFrequencyOrdering(at::Tensor basisFrequenciesH, at::Tensor basisFrequenciesV);
    at::Tensor reOrderCoefficients(const at::Tensor& coefficients, at::Tensor ordering);
    at::Tensor getBasisFrequencies(const at:: Tensor sgtMatrix, double angle) const;
    at::Tensor squareSorting(std::array<int64_t,4> size, at::Tensor coefficients);
    at::Tensor quadTreeSorting(std::array<int64_t,4> size, at::Tensor sortedTransformCoefficients);
    void quadTreeUnsorting(at::Tensor treeSortedTransform, at::Tensor& sortedTransformCoefficients);
    at::Tensor getFullOrder(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients) const;
    at::Tensor orderCoefficientsByFrequency(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH,const at::Tensor& sgtMatrixV) const;
    at::Tensor recoverOrder(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients) const;
    at::Tensor recoverCoefficientOrder(const at::Tensor& reOrderedCoefficientBlock, const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) const;
    at::Tensor getOrdinalFrequencies(const at::Tensor& basisFrequencies) const;
    at::Tensor to4DTransform(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients) const;
    at::Tensor from4DTransform(const at::Tensor& fullOrdinalFrequencies, const at::Tensor& coefficients4D) const;
    at::Tensor reverseOrderCoefficientsByFrequency(const at::Tensor& coefficientBlock,const at::Tensor& sgtMatrixH,const at::Tensor& sgtMatrixV) const;

   
    static void YCoCg2RGB(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Co, Block4D_ const &Cg, int Scale);
    static void YCbCr2RGB_BT601(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Cb, Block4D_ const &Cr, int Scale);
    static void RGB2YCbCr_BT601(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale);
    static void RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale);
    at::Tensor iSqrtCovMat(bool isHorizontal) const;
   
    static at::Tensor get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal);
    void sgtTransform(double scale);
    at::Tensor isgtTransformData(double scale, SgtSideInfo ssi) ;
    void isgtTransform(double scale, SgtSideInfo ssi);
    void sgtTransform(double scale,std::array<double,2> dispRange);
    at::Tensor getFlatBlock();
    at::Tensor flat24D(const at::Tensor& flatBlock) const;
    at::Tensor sgtFrom2DCoefficients(at::Tensor coefficients);
    at::Tensor flatBlockFrom4DTensor(at::Tensor coefficients);
    void emptyTransform();

    

    operator at::Tensor() const;
    //operator const at::Tensor&() const;
    Block4D_() = default;
    //Block4D_(at::Tensor data);
    Block4D_(const at::Tensor& data);
    //Block4D_(at::Tensor& data);
    Block4D_(std::array<int64_t,4> size);
    Block4D_(const Block4D_& B00, const Block4D_& B01, const Block4D_& B10, const Block4D_& B11, bool views);
    void Shift_UVPlane(int shift, int position_t, int position_s);
    void Ones(void);
    void Zeros(void);
    

    Block4D_ operator + (const Block4D_ &B);
    Block4D_ operator * (const Block4D_ &B);
    Block4D_ operator - (const Block4D_ &B);
    Block4D_ operator + (const int &a);
    Block4D_ operator * (const int &a);
    Block4D_ operator - (const int &a);
    Block4D_ operator / (const int &a);
    void operator += (const Block4D_ &B);
    void operator *= (const Block4D_ &B);
    void operator -= (const Block4D_ &B);
    void operator = (const Block4D_ &B);
    void operator = (Block4D_* B);
    Block4D_ clone() const;

    ~Block4D_() = default;
    
    void CopySubblockFrom(const Block4D_ &B, std::array<int64_t,4> sourceOffset, std::array<int64_t,4> targetOffset); 


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
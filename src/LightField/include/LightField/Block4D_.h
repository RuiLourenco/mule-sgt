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
    static constexpr double FIXED_ANGULAR_RHO = 0.99999;
    static constexpr double FIXED_SPATIAL_RHO = 0.99;
    static constexpr double VARIANCE_THRESHOLD = 3000;

    
    int angleVInt;
    int angleHInt;
    int rhoSInt;
    int rhoUInt;
    int rhoTInt;
    int rhoVInt;
    std::array<double,2> angleRange;

    static int pruneDouble(double dNumber,int factor);
    static double recoverDouble(int iNumber,int factor);
    double DecodeRho(int rhoCode) const;
    double DecodeAngle(int AngleCode) const;
    int codeRho(double rho) const;
    int codeAngle(double angle) const;


    public:
        static constexpr double PRECISION_RHO = 1e-5; 
        static constexpr double MIN_RHO = 0.2;
        static constexpr double MAX_RHO = 1-1e-5;
        static constexpr double PRECISION_ANGLE = 0.5;  
        void print();
        SgtSideInfo(const Block4D_& block,std::array<double,2> dispRange);
        SgtSideInfo(int RhoSInt,int RhoTInt,int RhoUInt,int RhoVInt,int angleVInt,int angleHInt, std::array<double,2> dispRange);
        SgtSideInfo(std::array<double,2> dispRange);
        SgtSideInfo(double disparityV,double disparityH, std::array<double,2> dispRange);
        void estimateRhosFromMonotony(Block4D_ block);
        
        std::array<double,2> disparityRange;

        int getRhoPrecision() const;
        int getAnglePrecision() const;

        double getRhoCodeScale() const;
        double getRhoCodeBias() const;
        double getAngleCodeScale() const;
        double getAngleCodeBias() const;
        double getDisparityV() const;
        double getDisparityH() const;
        double getAngleV() const;
        double getAngleH() const;
        double getRhoS() const;
        double getRhoU() const;
        double getRhoT() const;
        double getRhoV() const;
        int getRhoSCode() const;
        int getRhoTCode() const;
        int getRhoUCode() const;
        int getRhoVCode() const;
        int getAngleVCode() const;
        int getAngleHCode() const;
        void setRhoSCode(int code);
        void setRhoTCode(int code);
        void setRhoUCode(int code);
        void setRhoVCode(int code);
        void setAngleVCode(int code);
        void setAngleHCode(int code);
        void setRhoS(double rhoS);
        void setRhoU(double rhoU);
        void setRhoT(double rhoT);
        void setRhoV(double rhoV);
        void setAngleV(double angle);
        void setAngleH(double angle);
        void setAngleVFromDisparity(double disparity);
        void setAngleHFromDisparity(double disparity);
        void estimateDisparity(const Block4D_& block);
        void estimateRhos(const Block4D_& block, double varianceThreshold = VARIANCE_THRESHOLD);

        SgtSideInfo() = default;
        void estimateAngleFromMonotony(Block4D_ block);


    private:
        std::array<double,2> angleRangeFromDispRange(std::array<double,2> dispRange);

        void estimateRhoAngle(const Block4D_& block);
        void estimateRhosLS(const at::Tensor& covFun, bool isHorizontal);
        static double rhoFromCov(const at::Tensor& cov);
        void setSpatialRhos(const at::Tensor& covFunH, const at::Tensor& covFunV);
        void setSpatialRhos(const double rhoU, const double rhoV);
        void setAngularRhos(const double rhoS = FIXED_ANGULAR_RHO, const double rhoT = FIXED_ANGULAR_RHO);
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
    at::Tensor covFun2MatValid(const at::Tensor& covFun, bool isHorizontal) const;
    at::Tensor covFun2MatAll(const at::Tensor& covFun) const;
    static void splitHexaDecaTree(std::array<int64_t,4> length,std::array<int64_t,4> position,std::vector<std::array<int64_t,4>> &positions);

public: 
    at::Tensor autoCorr(bool isHorizontal);
    at::Tensor corrFun(bool isHorizontal) const;
    at::Tensor eigenValuesH = at::empty({0});
    at::Tensor eigenValuesV = at::empty({0});
    at::Tensor orderH = at::empty({0});
    at::Tensor orderV = at::empty({0});
    at::Tensor getOrderH();
    at::Tensor getOrderV();
    void kltTransform(double scale);
    at::Tensor ikltTransformData(double scale, at::Tensor covH, at::Tensor covV);
    void ikltTransform(double scale,at::Tensor covH, at::Tensor covV);

    at::Tensor getSgtTransformMatrix(const at::Tensor& cov, bool isHorizontal,at::Tensor& eigVals) const;
    static at::Tensor orderSGTByMonotony( at::Tensor epiTransform,at::Tensor sgtTransform,at::Tensor& order);

    static std::vector<std::array<int64_t,4>> treeOrderedCoefficientPositions(std::array<int64_t,4> size);
    at::Tensor iSqrtCovMat(at::Tensor covMat, bool isHorizontal ) const;

    static at::Tensor squareTransform(at::Tensor transform, at::Tensor eigenValuesH, at::Tensor eigenValuesV);
    void reOrderSGTMatrices(at::Tensor& sgtMatrixH, at::Tensor& sgtMatrixV, SgtSideInfo secondModel);
    void reReOrderSGTMatrices(at::Tensor& sgtMatrixH, at::Tensor& sgtMatrixV, SgtSideInfo secondModel);
    at::Tensor triangleSorting(std::array<int64_t,4> size, at::Tensor coefficients);
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

    at::Tensor secondModelOrderedBlock2SGT(at::Tensor flatTransform, SgtSideInfo secondModel, at::Tensor sgtMatrixH, at::Tensor sgtMatrixV);
    at::Tensor blockAsSecondModelOrderedVector(at::Tensor flatBlock, SgtSideInfo secondModel, at::Tensor sgtMatrixH, at::Tensor sgtMatrixV);
    at::Tensor blockAsEigenOrderedVector(at::Tensor flatBlock, at::Tensor eigenValuesH, at::Tensor eigenValuesV);
    std::array<int64_t,4> size;
    std::array<int64_t,4> transformSize;
    SgtSideInfo ssi;
     at::Tensor data;
    ValidPositions validPositions;
    bool includesInvalidCorners = false;
    bool sgtDomain = false;
    static void YCoCg2RGB(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Co, Block4D_ const &Cg, int Scale);
    static void YCbCr2RGB_BT601(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Cb, Block4D_ const &Cr, int Scale);
    static void RGB2YCbCr_BT601(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale);
    static void RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale);
    static at::Tensor normalizeCov(at::Tensor cov);
    at::Tensor calcModelCovMatrix(SgtSideInfo ssi,bool isHorizontal) const;
    at::Tensor calcModelCovFun(SgtSideInfo ssi,bool isHorizontal) const;
    static double varianceFromCov(const at::Tensor& cov);
    at::Tensor covFun2Mat(const at::Tensor& covFun,bool isHorizontal) const;
    at::Tensor iSqrtCovMat(bool isHorizontal) const;
   
    static at::Tensor get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal);
    void sgtTransform(double scale);
    at::Tensor isgtTransformData(double scale, SgtSideInfo ssi) ;
    void isgtTransform(double scale, SgtSideInfo ssi);
    void sgtTransform(double scale,std::array<double,2> dispRange);
    at::Tensor covFun(bool isHorizontal) const;
    at::Tensor getFlatBlock();
    at::Tensor flat24D(const at::Tensor& flatBlock) const;
    at::Tensor zigZagTransformMatrix(at::Tensor transform,bool isHorizontal);
    at::Tensor triangleTransformMatrix(at::Tensor transform,bool isHorizontal);
    at::Tensor sgtFrom2DCoefficients(at::Tensor coefficients);
    at::Tensor diagonalOrder4DSampling(at::Tensor coefficients);
    at::Tensor diagonalOrder4DBlock(at::Tensor coefficients);
    at::Tensor getZigZagIndexes(std::array<int64_t,2> size);
    at::Tensor getReverseZigZagIndexes(std::array<int64_t,2> size);
    at::Tensor getOrdered2DFromZigZagCoeffs(at::Tensor coeffs);
    at::Tensor getTriangleIndexes(std::array<int64_t,2> size);
    at::Tensor flatBlockFrom4DTensor(at::Tensor coefficients);
    void emptyTransform();
    at::Tensor batchedCovMatrix(bool isHorizontal) const;

    

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
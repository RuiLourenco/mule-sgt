#ifndef BLOCK4D__H
#define  BLOCK4D__H

#define block4DElementType int

#define PI 3.141592653589793
#include <torch/torch.h>
#include <array>

struct ValidPositions{
at::Tensor valid_positions_h;
at::Tensor valid_positions_v;
};
class Block4D_ 
{
public: 
    at::Tensor data;
    ValidPositions validPositions;
    bool includesNonValidCorners = false;
    bool sgtDomain = false;
    static at::Tensor get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal);
    void sgtTransform(double scale);

    operator at::Tensor() const;
    //operator const at::Tensor&() const;
    Block4D_() = default;
    //Block4D_(at::Tensor data);
    Block4D_(const at::Tensor& data);
    //Block4D_(at::Tensor& data);
    Block4D_(std::array<int,4> size);
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
    
    void CopySubblockFrom(const Block4D_ &B, std::array<int,4> sourceOffset, std::array<int,4> targetOffset); 


    void Display(void);
    void DCT_U(int scale);
    void IDCT_U(int scale);
    void DCT_V(int scale);
    void IDCT_V(int scale);
    void DCT_S(int scale);
    void IDCT_S(int scale);
    void DCT_T(int scale);
    void IDCT_T(int scale);
    void TRANSFORM_U(double scale, double *coefficients);
    void TRANSFORM_V(double scale, double *coefficients);
    void TRANSFORM_S(double scale, double *coefficients);
    void TRANSFORM_T(double scale, double *coefficients);
    void DCT4(int scale);
    void IDCT4(int scale);
    double L2Norm(void);
    void Extend_U(int position_u);
    void Extend_V(int position_v);
    void Extend_S(int position_s);
    void Extend_T(int position_t);
    void Clip(int minValue, int maxValue);
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

};

#endif
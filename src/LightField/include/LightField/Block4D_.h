#ifndef BLOCK4D__H
#define  BLOCK4D__H

#define block4DElementType int

#define PI 3.141592653589793
#include <torch/torch.h>
#include <array>


class Block4D_ 
{
public: 
    at::Tensor data;
    operator at::Tensor() const;
    Block4D_() = default;
    Block4D_(std::array<int,4> size);
    Block4D_(const Block4D_& B00, const Block4D_& B01, const Block4D_& B10, const Block4D_& B11);
    ~Block4D_() = default;
    void fromBlock(const Block4D_ &B, int source_offset_t, int source_offset_s, int source_offset_v, int source_offset_u, int target_offset_t=0, int target_offset_s=0, int target_offset_v=0, int target_offset_u=0););
    
    void CopySubblockFrom(const Block4D_ &B, std::array<int,4> sourceOffset, std::array<int,4> targetOffset); 

    void Ones(void);
    void Zeros(void);
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
    void Shift_UVPlane(int shift, int position_t, int position_s);

	block4DElementType GetPixel(int position_t, int position_s, int position_v, int position_u) {
		return(mPixelData[LinearPosition(position_t, position_s, position_v, position_u)]);
	}

	void SetPixel(block4DElementType pixel_value, int position_t, int position_s, int position_v, int position_u) {
		mPixelData[LinearPosition(position_t, position_s, position_v, position_u)] = pixel_value;
	}

	long int LinearPosition(long int position_t, long int position_s, long int position_v, long int position_u) {
		long int linear_position = position_t*mlength_u*mlength_v*mlength_s;
		linear_position += position_s*mlength_u*mlength_v + position_v*mlength_u + position_u;
		return(linear_position);
	}

};

#endif
#include "LightField/Block4D_.h"


Block4D_::operator at::Tensor() const{
    return this->data;
}

Block4D_::Block4D_(const at::Tensor& data){
    this->data = data;
}   
Block4D_::Block4D_(std::array<int,4> size) {
    this->data = torch::zeros({size[0], size[1], size[2], size[3]}, torch::kShort);
}

/**
 * Constructs a Block4D_ object by concatenating the data from four equal sized Block4D_ objects in a specific pattern.
 *
 * @param B00 The first Block4D_ object.
 * @param B01 The second Block4D_ object.
 * @param B10 The third Block4D_ object.
 * @param B11 The fourth Block4D_ object.
 *
 * @return None
 *
 * @throws None
 */
Block4D_::Block4D_(const Block4D_& B00, const Block4D_& B01, const Block4D_& B10, const Block4D_& B11,bool views) {
    int x1,x2;
    if(views){
        x1 = 0; x2 = 1;
    }else{
        x1 = 2; x2 = 3;
    }
    assert(B00.data.sizes() == B01.data.sizes() && B01.data.sizes() == B10.data.sizes() && B10.data.sizes() == B11.data.sizes() && "All Block4D_ objects must have the same shape.");
    this->data = torch::cat({torch::cat({B00.data, B01.data}, x1),torch::cat({B10.data, B11.data}, x1)},x2);
}
/**
 * Copies a subblock from another Block4D_ object to the current Block4D_ object.
 *
 * @param B The Block4D_ object from which the subblock is copied.
 * @param sourceOffset An array of 4 integers representing the starting index of the source subblock.
 * @param targetOffset An array of 4 integers representing the starting index of the target subblock.
 *
 * @throws None
 */
void Block4D_::CopySubblockFrom(const Block4D_ &B, std::array<int,4> sourceOffset, std::array<int,4> targetOffset){
    this->data.index({at::indexing::Slice(targetOffset[0],this->data.size(0)-1),
                          at::indexing::Slice(targetOffset[1],this->data.size(1)-1),
                          at::indexing::Slice(targetOffset[2],this->data.size(2)-1),
                          at::indexing::Slice(targetOffset[3],this->data.size(3)-1)}) =
                          B.data.index({at::indexing::Slice(sourceOffset[0],this->data.size(0)-1),
                                        at::indexing::Slice(sourceOffset[1],this->data.size(1)-1),
                                        at::indexing::Slice(sourceOffset[2],this->data.size(2)-1),
                                        at::indexing::Slice(sourceOffset[3],this->data.size(3)-1)
                                        });
}


void Block4D_::Ones(){
    this->data = torch::ones(this->data.sizes(),this->data.dtype());
}
void Block4D_::Zeros(){
    this->data = torch::zeros(this->data.sizes(),this->data.dtype());
}
void Block4D_::Shift_UVPlane(int shift, int position_t, int position_s){
   //std::cout<<this->data[position_t][position_s]<<std::endl<<std::endl;
    if(shift > 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_left_shift(shift);
    }
    if(shift < 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_right_shift(-shift);
    }
    //std::cout<<this->data[position_t][position_s]<<std::endl<<std::endl;
}

Block4D_ Block4D_::operator + (const Block4D_ &B){
    return Block4D_(this->data + B.data);
}
Block4D_ Block4D_::operator * (const Block4D_ &B){
    return Block4D_(this->data * B.data);
}
Block4D_ Block4D_::operator - (const Block4D_ &B){
    return Block4D_(this->data - B.data);
}
Block4D_ Block4D_::operator / (const int &a){
    return Block4D_(this->data / a);
}
Block4D_ Block4D_::operator + (const int &a){
    return Block4D_(this->data + a);
}
Block4D_ Block4D_::operator - (const int &a){
    return Block4D_(this->data - a);
}
Block4D_ Block4D_::operator * (const int &a){
    return Block4D_(this->data * a);
}
void Block4D_::operator += (const Block4D_ &B){
    this->data = this->data + B.data;
}
void Block4D_::operator -= (const Block4D_ &B){
    this->data = this->data - B.data;
}
void Block4D_::operator *= (const Block4D_ &B){
    this->data = this->data * B.data;
}
void Block4D_::operator = (const Block4D_ &B){
    this->data = B.data;
}

Block4D_ Block4D_::clone(){
    return this->data.clone();
}



// void Block4D_::Shift_UVPlane(int shift, int position_t, int position_s) {

//     this->data[position_t][position_s] = this->data[position_t][position_s]<<shift;

// }




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
void Block4D_::CopySubblockFrom(const Block4D_& B, std::array<int,4> sourceOffset, std::array<int,4> targetOffset){
    Block4D_ deepCopy = B.clone(); 
    this->data.index({at::indexing::Slice(targetOffset[0],this->data.size(0)-1),
                          at::indexing::Slice(targetOffset[1],this->data.size(1)-1),
                          at::indexing::Slice(targetOffset[2],this->data.size(2)-1),
                          at::indexing::Slice(targetOffset[3],this->data.size(3)-1)}) =
                          deepCopy.data.index({at::indexing::Slice(sourceOffset[0],this->data.size(0)-1),
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

Block4D_ Block4D_::clone() const{
    Block4D_ newBlock = Block4D_(this->data.clone());
    return newBlock;
}

/********Static Functions******/

at::Tensor Block4D_::get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal){
  
  
  int64_t view_coordinate = 0;
  int64_t spatial_coordinate = 2;
  if(is_horizontal){
    view_coordinate = 1;
    spatial_coordinate = 3;
  }
  int lf_extra_size = (int)(abs(round(adjustment_d*(lf_shape[view_coordinate]-1))));
  double true_alpha;
  if(adjustment_d == 0){
    true_alpha = 0;
  }else{
    true_alpha = adjustment_d/abs(adjustment_d) * (double)lf_extra_size/((double)lf_shape[view_coordinate]-1);
  }
  //cout<<"      True alpha = "<<true_alpha<<endl;
  std::vector<at::Tensor> padding_coordinates;
  //cout<<"      Block Start: "<<block_start[spatial_coordinate]<<endl;
  for(int l_ = 0; l_<lf_shape[view_coordinate]; l_++){
    if(block_start[view_coordinate]>l_) {
      //cout<<"Skipping! "<<block_start[view_coordinate]<<" "<<l_<<endl;
      continue;
    }
    if(block_start[view_coordinate]+block_shape[view_coordinate] - 1 < l_){
      //cout<<"Skipping! "<<block_start[view_coordinate]+block_shape[view_coordinate] - 1 <<" "<<l_<<endl;
      continue;
    } 
    int n_start = round(l_*true_alpha);
    int n_end = (lf_shape[spatial_coordinate]) + round(l_*true_alpha);

    if(true_alpha < 0){
      n_start -= (lf_shape[view_coordinate]-1)*true_alpha;
      n_end -= (lf_shape[view_coordinate]-1)*true_alpha;
    }

    torch::TensorOptions options = torch::TensorOptions();
    at::Tensor indexes =  at::empty({0},options.dtype(at::kLong));
    if(n_start-block_start[spatial_coordinate] < block_shape[spatial_coordinate]){
      int64_t blk_n_start = std::max(n_start-block_start[spatial_coordinate],(int64_t)0);
      int64_t blk_n_end = std::min(n_end-block_start[spatial_coordinate],block_shape[spatial_coordinate]);
      //cout<<"      "<<l_<<": "<<blk_n_start<<" "<<blk_n_end<<endl;
      indexes = (at::range(blk_n_start,blk_n_end-1,1)+block_shape[spatial_coordinate]*l_).to(at::kLong);
    }
    padding_coordinates.push_back(indexes);
  }
  at::Tensor vectorized_padding = torch::cat(padding_coordinates);
  return vectorized_padding;
}

// void Block4D_::Shift_UVPlane(int shift, int position_t, int position_s) {

//     this->data[position_t][position_s] = this->data[position_t][position_s]<<shift;

// }




#include "LightField/Block4D.h"
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>
#include <opencv2/opencv.hpp>
#include "osqp.h"


#include <boost/range/irange.hpp>
#include <boost/range/join.hpp>

#include <boost/phoenix.hpp>
#include <limits>




using namespace boost;
using namespace boost::adaptors;
using int_arr_t = std::vector<std::int64_t>;
using tensor_arr_t = std::vector<at::Tensor>;
template<class Container> class to_t {};
template<class Range, class C>
auto operator|(Range&& r, to_t<C>) { return copy_range<C>(std::forward<Range>(r)); };

#define DEBUG 0
cv::Mat torchToCv(const torch::Tensor& tensor) {
    // Get tensor shape
    auto sizes = tensor.sizes();
    int height = sizes[0];
    int width = sizes[1];

    // Check if tensor is of type float64
    //cout<<"Tensor Type is "<<tensor.dtype()<<endl;
    tensor.to(at::kDouble);
    //TORCH_CHECK(tensor.dtype() == torch::kDouble, "Tensor data type must be double.");
    cv::Mat image(height, width, CV_64FC1, cv::Scalar(0));
    for( int h = 0 ; h < height; h++){
      for( int w = 0; w< width;w++){
        image.at<double>(h,w) = tensor[h][w].item<double>();
      }
    }
    return image;
}

void write_image(const cv::Mat& image,std::string path,std::array<double,2> valueRange = {1,1}){ 
    cv::Mat display_image;
    double min_val; 
    double max_val; 
    cv::Point min_loc; 
    cv::Point max_loc;
    if(valueRange[0] == valueRange[1]){ 
        minMaxLoc( image, &min_val, &max_val, &min_loc, &max_loc );
    }
    else{
        min_val = valueRange[0];
        max_val = valueRange[1];
    }
    std::cout<<path<<" Min Val: "<<min_val<<" Max Val: "<<max_val<<std::endl;
    display_image = min(max(image,min_val), max_val);
    display_image -= min_val;
    display_image /= max_val - min_val;
    display_image *= 255;
    display_image.convertTo(display_image,CV_8U);
    cv::Mat img_color;
    // Apply the colormap:
    applyColorMap(display_image, img_color, cv::COLORMAP_JET);
    // Write the image
    cv::imwrite(path, img_color);
    //cv::imwrite(path, display_image);
}
void write_tensor(torch::Tensor tensor, std::string path, std::array<double,2> valueRange){
#if DEBUG == 1
  cv::Mat mat = torchToCv(tensor);
  write_image(mat,path,valueRange);
#endif
}


#include <torch/torch.h>
#include <osqp/osqp.h>



void Block4D::Extend_T(int length_t) {
    this->data.index({at::indexing::Slice(length_t,data.size(0)),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(length_t-1,length_t),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()}).repeat({data.size(0)-length_t,1,1,1});
 
}
void Block4D::Extend_S(int length_s) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(length_s,data.size(1)),at::indexing::Slice(),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(length_s-1,length_s),at::indexing::Slice(),at::indexing::Slice()}).repeat({1,data.size(1)-length_s,1,1});
}
void Block4D::Extend_V(int length_v) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_v,data.size(2)),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_v-1,length_v),at::indexing::Slice()}).repeat({1,1,data.size(2)-length_v,1});
}
void Block4D::Extend_U(int length_u) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_u,data.size(3))}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_u-1,length_u)}).repeat({1,1,1,data.size(1)-length_u});
}
void Block4D::clip(int min, int max) {
    this->data = this->data.clamp(min,max);
}

Block4D::operator at::Tensor() const{
    return this->data;
}


Block4D::Block4D(std::array<int64_t,4> size,std::array<int64_t,4>lightFieldPosition,LightField* lightField)
    : size(size), lightFieldPosition(lightFieldPosition), lightField(lightField){
    this->data = torch::zeros({size[0], size[1], size[2], size[3]}, torch::kInt);
    this->sgtDomain = false;
    int64_t fillHLow = std::abs(lightField->preSlantTan) * lightField->data.size(1);

    int64_t fillHHigh = lightField->data.size(3) - fillHLow - size[3];
    int64_t fillVLow = std::abs(lightField->preSlantTan) * lightField->data.size(0);
    int64_t fillVHigh = lightField->data.size(2) - fillVLow - size[2];
    
    #if FLAT_TRANSFORM == 1
    this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
    #else 
    this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
    #endif
    
    if( (lightFieldPosition[3] > fillHLow && lightFieldPosition[3] < fillHHigh &&
        lightFieldPosition[2] > fillVLow && lightFieldPosition[2] < fillVHigh) || lightField->preSlantTan == 0){
            //std::cout<<"No Invalid Corners. "<<std::endl;
            this->includesInvalidCorners = false;
    }
    else{
        std::array<int64_t,4> lfSize;
        for (int i = 0; i < 4; i++){
            lfSize[i] = lightField->data.size(i);
        }
        at::Tensor validPositionH = get_valid_position(lightField->preSlantTan,lfSize,size,lightFieldPosition,true);
        at::Tensor validPositionV = get_valid_position(lightField->preSlantTan,lfSize,size,lightFieldPosition,false);
        this->validPositions = ValidPositions{validPositionH,validPositionV};
        if(validPositionH.size(0) == size[1] * size[3] && validPositionV.size(0) == size[0] * size[2]){
                //std::cout<<"All positions are valid for subblock copy. "<<std::endl;
                this->includesInvalidCorners = false;
        }
        else{
            this->includesInvalidCorners = true;
            #if FLAT_TRANSFORM == 1
            this->transformSize = {1,1,validPositionV.size(0),validPositionH.size(0)};
            #else 
            std::cerr<<"Pre-Slant Not implemented for non-flat transform!"<<std::endl;
            #endif
        }
    }


    
}
void Block4D::emptyTransform(){
    this->data = torch::zeros({this->transformSize[0], this->transformSize[1], this->transformSize[2], this->transformSize[3]}, torch::kInt);
    this->sgtDomain = true;
}



/**
 * Constructs a Block4D object by concatenating the data from four equal sized Block4D objects in a specific pattern.
 *
 * @param B00 The first Block4D object.
 * @param B01 The second Block4D object.
 * @param B10 The third Block4D object.
 * @param B11 The fourth Block4D object.
 *
 * @return None
 *
 * @throws None
 */
Block4D::Block4D(const Block4D& B00, const Block4D& B01, const Block4D& B10, const Block4D& B11,bool views) {
    int x1,x2;
    if(views){
        x1 = 0; x2 = 1;
    }else{
        x1 = 2; x2 = 3;
    }
    
    assert(B00.data.size(x1)+B10.data.size(x1) == B01.data.size(x1)+B11.data.size(x1) && "heights don't match" );
    assert(B00.data.size(x2)+B01.data.size(x2) == B10.data.size(x2)+B11.data.size(x2) && "widths don't match");
    this->data = torch::cat({torch::cat({B00.data, B10.data}, x1),torch::cat({B01.data, B11.data}, x1)},x2).to(at::kInt);
    this->size = B00.size;
    this->size[x1] = B00.size[x1]+B10.size[x1];
    this->size[x2] = B00.size[x2]+B01.size[x2];
    
#if FLAT_TRANSFORM == 1
    this->transformSize = {1,1,this->size[0]*this->size[2],this->size[1]*this->size[3]};
#else 
    this->transformSize = {this->size[0],this->size[1],this->size[2],this->size[3]};
#endif

    this->sgtDomain = B00.sgtDomain;
    this->lightFieldPosition = B00.lightFieldPosition;
    this->lightField = B00.lightField;
    if( (!B00.includesInvalidCorners && !B01.includesInvalidCorners && !B10.includesInvalidCorners && !B11.includesInvalidCorners)){
            this->includesInvalidCorners = false;
    }
    else{
        std::array<int64_t,4> lfSize;
        for (int i = 0; i < 4; i++){
            lfSize[i] = this->lightField->data.size(i);
        }
        at::Tensor validPositionH = get_valid_position(lightField->preSlantTan,lfSize,this->size,this->lightFieldPosition,true);
        at::Tensor validPositionV = get_valid_position(lightField->preSlantTan,lfSize,this->size,this->lightFieldPosition,false);
        this->validPositions = ValidPositions{validPositionH,validPositionV};
        if(validPositionH.size(0) == size[1] * size[3] && validPositionV.size(0) == size[0] * size[2]){
                this->includesInvalidCorners = false;
        }
        else{


            this->includesInvalidCorners = true;
            #if FLAT_TRANSFORM == 1
            this->transformSize = {1,1,validPositionV.size(0),validPositionH.size(0)};
            #else 
            std::cerr<<"Pre-Slant Not implemented for non-flat transform!"<<std::endl;
            #endif
        }
    }
}
std::ostream &operator<<(std::ostream &os, std::array<int64_t,4> vec) { 
    return os << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << "]";
}
std::array<int64_t,4> Block4D::toTransformCoords(std::array<int64_t,4> coords) const{
    std::array<int64_t,4> transformCoords = {coords[0]/size[0],coords[0]/size[0],size[0]*coords[2],size[1]*coords[3]}; //DOES NOT WORK FOR VIEW SPLITTING

    return transformCoords;
}

void Block4D::copySubblockData(Block4D& destination, std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset) const{

    for( int i = 0; i < 4; i++){
        destination.lightFieldPosition[i] = this->lightFieldPosition[i] + sourceOffset[i];
    }


    std::array<int64_t,4> length = {std::min(subblockLength[0], this->size[0]-sourceOffset[0]),
                                    std::min(subblockLength[1], this->size[1]-sourceOffset[1]),
                                    std::min(subblockLength[2], this->size[2]-sourceOffset[2]),
                                    std::min(subblockLength[3], this->size[3]-sourceOffset[3])};
    
    destination.size = length;
    if(sgtDomain)std::cout<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
    if(!destination.includesInvalidCorners){
        destination.transformSize = toTransformCoords(length);
    }else{
        destination.transformSize = {1,1,destination.validPositions.valid_positions_v.size(0),destination.validPositions.valid_positions_h.size(0)};
    }

    if(this->sgtDomain){
        length = destination.transformSize;
        sourceOffset = toTransformCoords(sourceOffset);
        if(this->includesInvalidCorners){
           
            int offsetReductionV = computePreviousInvalidNumber(this->lightField->preSlantTan,this->lightFieldPosition[2],destination.lightFieldPosition[2],false);
            int offsetReductionH= computePreviousInvalidNumber(this->lightField->preSlantTan,this->lightFieldPosition[3],destination.lightFieldPosition[3],true);
            sourceOffset = {0,0,sourceOffset[2] - offsetReductionV,sourceOffset[3] -offsetReductionH };
        }
    }

    destination.data  = destination.data.index({at::indexing::Slice(sourceOffset[0],sourceOffset[0]+length[0]),
                                        at::indexing::Slice(sourceOffset[1],sourceOffset[1]+length[1]),
                                        at::indexing::Slice(sourceOffset[2],sourceOffset[2]+length[2]),
                                        at::indexing::Slice(sourceOffset[3],sourceOffset[3]+length[3])
                                        });
}
int Block4D::computePreviousInvalidNumber(double preSlantTan,int parentBlockN, int subblockN,bool isHorizontal) const{
    int angleVariable = isHorizontal ? 1 : 0; // 1 for horizontal, 0 for vertical
    int spaceVariable = isHorizontal ? 3 : 2; // 3 for horizontal, 2 for vertical
    int size_increase = (int)(abs(round(preSlantTan*(this->lightField->data.size(angleVariable)-1))));

    double d = preSlantTan/abs(preSlantTan) * (double)size_increase/((double)this->lightField->data.size(angleVariable)-1);
    double spaceSize = (double)this->lightField->data.size(spaceVariable) - size_increase;
    double a = 1/d;
    int maxAngle = this->lightField->data.size(angleVariable);
    int count_upper = 0;
    int count_lower = 0;
    int count = 0;
    for(int n = parentBlockN; n < subblockN; n++){
        int upper_l_boundary;
        int lower_l_boundary;
        if(d < 0){
            int offset = size_increase;
            lower_l_boundary = static_cast<int>(floor((n - offset + 1.0) * a));
            upper_l_boundary = static_cast<int>(ceil((n - offset - spaceSize) * a));
            
        }else{
            upper_l_boundary = static_cast<int>(ceil((n + 1.0) * a));
            lower_l_boundary = static_cast<int>(ceil((n - spaceSize) * a));
        }
        count_upper += std::max(0, maxAngle - upper_l_boundary);
        count_lower += std::max(lower_l_boundary + 1, 0);
        count += std::max(0, maxAngle - upper_l_boundary) + std::max(lower_l_boundary + 1, 0);

    }


    return count;

}

std::vector<int64_t> Block4D::copyValidSubblockPositions(std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset, bool isHorizontal){
    
    
    
    std::vector<int64_t> validPositions;
    int angleCoord = isHorizontal ? 1 : 0; // 1 for horizontal, 0 for vertical
    int spatialCoord = isHorizontal ? 3 : 2; // 3 for horizontal, 2 for vertical
    auto* data = isHorizontal ? this->validPositions.valid_positions_h.data_ptr<int64_t>() : this->validPositions.valid_positions_v.data_ptr<int64_t>();
    auto* end = isHorizontal ? data + this->validPositions.valid_positions_h.size(0) : data + this->validPositions.valid_positions_v.size(0);
    for(int l = sourceOffset[angleCoord]; l < sourceOffset[angleCoord] + subblockLength[angleCoord]; l++){
        if(l< 0 || l >= this->size[angleCoord]) throw std::runtime_error("Invalid sourceOffset for Block4D copy");
        for(int n = sourceOffset[spatialCoord]; n < sourceOffset[spatialCoord] + subblockLength[spatialCoord]; n++){
            if(n< 0 || n >= this->size[spatialCoord]) throw std::runtime_error("Invalid sourceOffset for Block4D copy");
            int64_t oldPosition = l * this->size[spatialCoord] + n;
            int64_t position = l * subblockLength[spatialCoord] + n - sourceOffset[spatialCoord];
            
            
            auto it = std::find(data, end, oldPosition);
            if(it != end){
                validPositions.push_back(position);
            }
        }
    }
    return validPositions;
}

Block4D Block4D::copySubblock(std::array<int64_t,4> subblockLength, std::array<int64_t,4> sourceOffset){



    Block4D deepCopy = this->clone();
        //std::cout<<"deepCopy.data.device(): "<<deepCopy.data.device()<<std::endl;
    
    if(includesInvalidCorners){
        std::vector<int64_t> validPositionsH = copyValidSubblockPositions(subblockLength, sourceOffset, true);
        std::vector<int64_t> validPositionsV = copyValidSubblockPositions(subblockLength, sourceOffset, false);
        

        if(validPositionsH.size() == subblockLength[1] * subblockLength[3] && validPositionsV.size() == subblockLength[0] * subblockLength[2]){
            deepCopy.includesInvalidCorners = false;
        }else{
            deepCopy.includesInvalidCorners = true;
            at::Tensor validPositionsTensorH = torch::from_blob(validPositionsH.data(), {static_cast<int64_t>(validPositionsH.size())}, torch::kLong).clone();
            at::Tensor validPositionsTensorV = torch::from_blob(validPositionsV.data(), {static_cast<int64_t>(validPositionsV.size())}, torch::kLong).clone();
            deepCopy.validPositions.valid_positions_h = validPositionsTensorH;
            deepCopy.validPositions.valid_positions_v = validPositionsTensorV;
        }


    }
            

    copySubblockData(deepCopy,subblockLength, sourceOffset);

    return deepCopy;    
}

/**
 * Copies a subblock from another Block4D object to the current Block4D object.
 *
 * @param B The Block4D object from which the subblock is copied.
 * @param sourceOffset An array of 4 integers representing the starting index of the source subblock.
 * @param targetOffset An array of 4 integers representing the starting index of the target subblock.
 *
 * @throws None
 */
void Block4D::CopySubblockFrom(const Block4D& B, std::array<int64_t,4> sourceOffset, std::array<int64_t,4> targetOffset){
    
    Block4D deepCopy = B.clone(); 
    std::array<int64_t,4> length = {std::min(this->data.size(0) - targetOffset[0],B.data.size(0)),
                                    std::min(this->data.size(1) - targetOffset[1],B.data.size(1)),
                                    std::min(this->data.size(2) - targetOffset[2],B.data.size(2)),
                                    std::min(this->data.size(3) - targetOffset[3],B.data.size(3))};

       
    this->data.index({at::indexing::Slice(targetOffset[0],targetOffset[0] + length[0]),
                          at::indexing::Slice(targetOffset[1],targetOffset[1] + length[1] ),
                          at::indexing::Slice(targetOffset[2],targetOffset[2] + length[2]),
                          at::indexing::Slice(targetOffset[3],targetOffset[3] + length[3])}) =
                          deepCopy.data.index({at::indexing::Slice(sourceOffset[0],sourceOffset[0]+length[0]),
                                        at::indexing::Slice(sourceOffset[1],sourceOffset[1]+length[1]),
                                        at::indexing::Slice(sourceOffset[2],sourceOffset[2]+length[2]),
                                        at::indexing::Slice(sourceOffset[3],sourceOffset[3]+length[3])
                                        });


    
}
at::Tensor Block4D::getTransformMatrix(const at::Tensor& cov, bool isHorizontal, at::Tensor& eigVals) const{


    at::Tensor transform = klt(cov,eigVals);

    
    at::Tensor normalizedTransform = transform.clone();
    double eps = 1/sqrt(transform.size(0)) *1e-5;

    for(int i = 0; i < transform.size(1); i++){
        int bias = 0;
        while(bias < transform.size(0)){

            double reference = transform[bias][i].item<double>();
            if(abs(reference) > eps){
        
                if(reference < 0){
                    transform.index({at::indexing::Slice(),i}) =  -1 * transform.index({at::indexing::Slice(),i});   
                }          
                break;
            }

            bias++;

        }
    }
    return transform;
}
  



at::Tensor Block4D::autoCorr(bool isHorizontal){
    at::Tensor flatBlock = getFlatBlock().to(at::kDouble);
    if(isHorizontal){
        flatBlock = flatBlock.t();
    }

    at::Tensor autoCorr = at::zeros({flatBlock.size(0),flatBlock.size(0)},at::kDouble);
    for(int i=0;i<flatBlock.size(0);i++){
        for(int j=0;j<flatBlock.size(0);j++){
            autoCorr[i][j] = (flatBlock[i]*flatBlock[j]).sum();
        }
    }
    return autoCorr;
}

void Block4D::kltTransform(double scale){

    at::Tensor flatBlock = getFlatBlock().to(at::kDouble);
    at::Tensor currCovH = autoCorr(true);
    at::Tensor currCovV = autoCorr(false);

    flatBlock*=scale;
    at::Tensor eigValsH, eigValsV;
    at::Tensor sgtMatrixH   = getTransformMatrix(currCovH,true,eigValsH);
    at::Tensor sgtMatrixV =   getTransformMatrix(currCovV,false,eigValsV);
    this->eigenValuesH = eigValsH;
    this->eigenValuesV = eigValsV;
    at::Tensor flatTransform = matrixTransform(flatBlock,sgtMatrixH,sgtMatrixV,eigValsH,eigValsV);
    flatTransform = flatTransform.unsqueeze(0).unsqueeze(0);
    this->data = flatTransform.round().to(at::kInt).contiguous();
    this->sgtDomain = true;
}

   
at::Tensor Block4D::ikltTransformData(double scale, at::Tensor covH, at::Tensor covV) {

    at::Tensor flatBlock = this->data.squeeze().to(at::kDouble)/scale;  
    at::Tensor eigValsH,eigValsV; 
    at::Tensor sgtMatrixH   = getTransformMatrix(covH,true,eigValsH);
    at::Tensor sgtMatrixV = getTransformMatrix(covV,false,eigValsV);     
    
    at::Tensor flatTransform = iMatrixTransform(flatBlock,sgtMatrixH,sgtMatrixV);
    at::Tensor newData = flat24D(flatTransform);
    return flat24D(flatTransform);
}


at::Tensor Block4D::klt(at::Tensor covMat, at::Tensor& eigVals){
    //std::cout<<covMat.sizes()<<std::endl;
    try{
        auto [L, Q] = torch::linalg::eigh(covMat, "U");
        eigVals = L.flip({-1});
        return Q.flip({-1});
    }
    catch (const c10::Error& e) {
        std::cerr << "Error computing eigen decomposition: " << e.what() << std::endl;
        std::cout<<"Covariance Matrix Size: "<<covMat.sizes()<<std::endl; 
        exit(-1);
    }
    
}



at::Tensor Block4D::matrixTransform(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV,const at::Tensor& eigValsH,const at::Tensor& eigValsV) {
    at::Tensor block = flatBlock.to(at::kDouble);
    at::Tensor transform = at::mm(at::mm(sgtMatrixV.t(), block), sgtMatrixH);
    
    return transform;
}

at::Tensor Block4D::iMatrixTransform(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) {
    //std::cout<<"sgtMatrixV: "<<sgtMatrixV.mean({1}).index({at::indexing::Slice(0,10)})<<std::endl;
    at::Tensor transform = at::mm(at::mm(sgtMatrixV, flatBlock), sgtMatrixH.t()).round().to(at::kInt);
    return transform;
}


at::Tensor Block4D::flat24D(const at::Tensor& flatBlock) const{
    at::Tensor block;
    if(includesInvalidCorners){
        block = flat24DValid(flatBlock);
    }else{
        block = flat24DAll(flatBlock);
    }
    return block;
}
at::Tensor Block4D::flat24DAll(const at::Tensor& flatBlock) const{
    //std::cout<<flatBlock.sizes()<<std::endl;
    std::array<int64_t,4> sizeShifted = {size[1],size[3],size[0],size[2]};
    c10::IntArrayRef permutedSizes(sizeShifted);
    at::Tensor unflattened_block = flatBlock.t().reshape(permutedSizes);
    unflattened_block = unflattened_block.permute({2,0,3,1});   
    //std::cout<<"size unflattened: "<< unflattened_block.size(0)<<" "<<unflattened_block.size(1)<<" "<<unflattened_block.size(2)<<" "<<unflattened_block.size(3)<<std::endl;

    return  unflattened_block;
}

at::Tensor Block4D::flat24DValid(const at::Tensor& flatBlock) const {
    at::Tensor padded_block = torch::zeros({size[0]*size[2],size[1]*size[3]}).to(at::kInt);
    auto a = at::meshgrid({validPositions.valid_positions_v, validPositions.valid_positions_h},"ij");
    padded_block = padded_block.index_put({a[0],a[1]},flatBlock);
    std::array<int64_t,4> sizeShifted = {size[1],size[3],size[0],size[2]};
    c10::IntArrayRef permutedSizes(sizeShifted);
    at::Tensor unflattened_block = padded_block.t().reshape(permutedSizes);
    unflattened_block = unflattened_block.permute({2,0,3,1});
    return  unflattened_block;

}
at::Tensor Block4D::getFlatBlock() {
    at::Tensor flatBlock;
    if(includesInvalidCorners){
        flatBlock = getFlatBlockValid();
    }else{
        //std::cout<<"All!"<<std::endl;
        flatBlock = getFlatBlockAll();
    }
    //std::cout<<"flatBlock size: "<<flatBlock.sizes()<<std::endl;
    return flatBlock;
}


at::Tensor Block4D::getFlatBlockAll(){
    if(sgtDomain){
        std::cerr<<"getFlatBlockAll: sgtDomain is not supported!"<<std::endl;
        abort();
    }
    std::array<int64_t,2> dims,cDims;
    //The Goal is for the block to be LN x KL (H x W format)
    dims = {1,3}; 
    cDims = {0,2};
    //std::cout<<"FlatBlock: "<<dims[0]<<" "<<dims[1]<<" | "<<cDims[0]<<" "<<cDims[1]<<std::endl;
    const std::int64_t sample_rank = 4;
    const auto t_dims = dims;
    std::array<int64_t,4> transposed_dims = {cDims[0],cDims[1],dims[0],dims[1]};
    //std::cout<<"transposed_dims: "<<transposed_dims[0]<<" "<<transposed_dims[1]<<" | "<<transposed_dims[2]<<" "<<transposed_dims[3]<<std::endl;
    auto flatBlock = this->data.permute(transposed_dims).flatten(0,1).flatten(-dims.size());
    return flatBlock;

}

at::Tensor Block4D::getFlatBlockValid(){
    std::array<int64_t,2> dims,cDims;
    //The Goal is for the block to be LN x KL (H x W format)
    at::Tensor flatBlock = getFlatBlockAll();
    flatBlock = flatBlock.index({at::indexing::Slice(), validPositions.valid_positions_h});
    flatBlock = flatBlock.index({validPositions.valid_positions_v,at::indexing::Slice()});
    return flatBlock;
}


void Block4D::ikltTransform(Block4D reconstructedBlock, double scale){
    at::Tensor covH = reconstructedBlock.autoCorr(true);
    at::Tensor covV = reconstructedBlock.autoCorr(false);
    at::Tensor recoveredBlock = ikltTransformData(scale,covH,covV);
    this->data = recoveredBlock;
    //std::cout<<"Final Data = "<<this->data.sizes()<<std::endl;
    this->sgtDomain = false;    
}



at::Tensor Block4D::normalizeCov(at::Tensor cov){
    //Get the index of the central value (variance)
    int64_t s_center = cov.sizes()[0]/2;
     int64_t u_center = cov.sizes()[1]/2;
     //normalize so variance = 1
     cov = cov / cov[s_center][u_center];
    return cov;
}




void Block4D::Ones(){
    this->data = torch::ones(this->data.sizes(),this->data.dtype());
}
void Block4D::Zeros(){
    this->data = torch::zeros(this->data.sizes(),this->data.dtype());
}
void Block4D::Shift_UVPlane(int shift, int position_t, int position_s){
    if(shift > 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_left_shift(shift);
    }
    if(shift < 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_right_shift(-shift);
    }
}


Block4D Block4D::operator + (const Block4D &B) const{
    Block4D newBlock = this->clone();
    newBlock.data = this->data + B.data;
    return newBlock;
}
Block4D Block4D::operator * (const Block4D &B) const{
    Block4D newBlock = this ->clone();
    newBlock.data = this->data * B.data;
    return newBlock;
}
Block4D Block4D::operator - (const Block4D &B) const {
    Block4D newBlock = this->clone();
    newBlock.data = this->data - B.data;
    return newBlock;
}
Block4D Block4D::operator + (const at::Tensor &B) const{
    Block4D newBlock = this->clone();
    newBlock.data = this->data + B;
    return newBlock;
}
Block4D Block4D::operator * (const at::Tensor &B) const{
    Block4D newBlock = this ->clone();
    newBlock.data = this->data * B;
    return newBlock;
}
Block4D Block4D::operator - (const at::Tensor &B) const {
    Block4D newBlock = this->clone();
    newBlock.data = this->data - B;
    return newBlock;
}

Block4D operator * (const int a,const Block4D &B ){
    return B * a;
}

Block4D operator + (const int a,const Block4D &B ){
    return B + a;
}

Block4D operator - (const int a,const Block4D &B ){
    return -1 * (B - a);
}

Block4D Block4D::operator / (const int a) const{
    Block4D newBlock = this->clone();
    newBlock.data = this->data / a;
    return newBlock;
}
Block4D Block4D::operator / (const double a) const{
    Block4D newBlock = this->clone();
    newBlock.data = this->data / a;
    return newBlock;
}
Block4D Block4D::operator + (const int a) const {
    Block4D newBlock = this->clone();
    newBlock.data = this->data + a;
    return newBlock;
}
Block4D Block4D::operator - (const int a) const{
    Block4D newBlock = this->clone();
    newBlock.data = this->data - a;
    return newBlock;

}
Block4D Block4D::operator * (const int a) const{
    Block4D newBlock = this->clone();
    newBlock.data = this->data *a;
    return newBlock;
}
void Block4D::operator += (const Block4D &B){
    this->data = this->data + B.data;
}
void Block4D::operator -= (const Block4D &B){
    this->data = this->data - B.data;
}
void Block4D::operator *= (const Block4D &B){
    this->data = this->data * B.data;
}
void Block4D::operator = (const Block4D &B){
    
    this->data = B.data;
    this->size = B.size;
    this->transformSize = B.transformSize;
    this->sgtDomain = B.sgtDomain;
    this->includesInvalidCorners = B.includesInvalidCorners;
    this->validPositions = B.validPositions;
    this->lightFieldPosition = B.lightFieldPosition;
    this->lightField = B.lightField;
}

Block4D Block4D::clone() const{

    Block4D newBlock(this->size,this->lightFieldPosition,this->lightField);
    newBlock.data = this->data.clone();    
    newBlock.size = this->size;
    newBlock.transformSize = this->transformSize;
    newBlock.sgtDomain = this->sgtDomain;
    newBlock.includesInvalidCorners = this->includesInvalidCorners;
    if(this->includesInvalidCorners){
        newBlock.validPositions = ValidPositions(this->validPositions.valid_positions_h.clone(), this->validPositions.valid_positions_v.clone());
    }
    //newBlock.validPositions = ValidPositions(this->validPositions.valid_positions_h.clone(), this->validPositions.valid_positions_v.clone());
    newBlock.lightFieldPosition = this->lightFieldPosition;
    newBlock.lightField = this->lightField;

    return newBlock;
}

/********Static Functions******/

at::Tensor Block4D::get_valid_position(double adjustment_d,std::array<int64_t,4> lf_shape,std::array<int64_t,4> block_shape,std::array<int64_t,4>block_start,bool is_horizontal){
  
    //std::cout<<"block_start: "<<block_start[0]<<" "<<block_start[1]<<" "<<block_start[2]<<" "<<block_start[3]<<std::endl;
    //std::cout<<"block_shape: "<<block_shape[0]<<" "<<block_shape[1]<<" "<<block_shape[2]<<" "<<block_shape[3]<<std::endl;
    //std::cout<<"lf_shape: "<<lf_shape[0]<<" "<<lf_shape[1]<<" "<<lf_shape[2]<<" "<<lf_shape[3]<<std::endl;
    
    if(block_start[2] >= lf_shape[2] || block_start[3] >= lf_shape[3]){
        if(block_shape[0] == 1){
            std::cerr<<"Block start is out of bounds! "<<block_start[2]<<" >= "<<lf_shape[2]<<" || "<<block_start[3]<<" >= "<<lf_shape[3]<<std::endl;

        }
        at::Tensor valid_positions = at::empty({0},at::kLong);
        return valid_positions;
    }
    int64_t view_coordinate = 0;
    int64_t spatial_coordinate = 2;
    if(is_horizontal){
        view_coordinate = 1;
        spatial_coordinate = 3;
    }
    int lf_extra_size = (int)(abs(round(adjustment_d*(lf_shape[view_coordinate]-1))));
    lf_shape[spatial_coordinate] = lf_shape[spatial_coordinate] - lf_extra_size;
    lf_shape[3] = lf_shape[3] - lf_extra_size;

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
        int n_start = floor(l_*true_alpha);
        int n_end = (lf_shape[spatial_coordinate]) + floor(l_*true_alpha);

        if(true_alpha < 0){
            n_start -= (lf_shape[view_coordinate]-1)*true_alpha;
            n_end -= (lf_shape[view_coordinate]-1)*true_alpha;
        }
        //std::cout<<l_<<": "<<n_start<<" "<<n_end<<std::endl;

        torch::TensorOptions options = torch::TensorOptions();
        at::Tensor indexes =  at::empty({0},options.dtype(at::kLong));
        if(n_start-block_start[spatial_coordinate] < block_shape[spatial_coordinate]){
            int64_t blk_n_start = std::max(n_start-block_start[spatial_coordinate],(int64_t)0);
            int64_t blk_n_end = std::min(n_end-block_start[spatial_coordinate],block_shape[spatial_coordinate]);
            
            if (blk_n_end - blk_n_start < 1) continue;
            indexes = (at::range(blk_n_start,blk_n_end-1,1)+block_shape[spatial_coordinate]*l_).to(at::kLong);
        }
        padding_coordinates.push_back(indexes);
         
    }
    
    at::Tensor vectorized_padding;
    if (padding_coordinates.empty()) {
        vectorized_padding = at::empty({0}, at::kLong);
    } else {
        vectorized_padding = torch::cat(padding_coordinates);
    }
    if(false){
    //if(block_start[3] == 512 && block_shape[3] == 64 && is_horizontal){
        std::cout<<"Block Shape: "<<block_shape[0]<<" "<<block_shape[1]<<" "<<block_shape[2]<<" "<<block_shape[3]<<std::endl;
        std::cout<<"Block Start: "<<block_start[0]<<" "<<block_start[1]<<" "<<block_start[2]<<" "<<block_start[3]<<std::endl;
        std::cout<<"vectorized_padding size: "<<vectorized_padding.sizes()<<std::endl;
        for(int i = 0; i<vectorized_padding.size(0);i++){
            std::cout<<"("<<(int) vectorized_padding[i].item<int64_t>()/block_shape[spatial_coordinate]<<" "<<vectorized_padding[i].item<int64_t>()%block_shape[spatial_coordinate]<<") ";
        }
        std::cout<<std::endl;
    }
    //std::cout<<"Initial Tenosr Type: "<<vectorized_padding.dtype()<<std::endl;
    return vectorized_padding;
}








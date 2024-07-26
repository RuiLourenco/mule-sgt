#include "LightField/Block4D_.h"
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>

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

std::pair<at::Tensor, at::Tensor> make_function_grid(at::IntArrayRef sizes, at::ScalarType dtype = c10::ScalarType::Double) {
    using namespace phoenix::placeholders;

    auto opts = at::TensorOptions{}.dtype(dtype);
    auto ranges = sizes
        | transformed([&](auto&& sz){ return at::arange(1-sz, sz, opts); })
        | to_t<tensor_arr_t>{};

    auto meshes = at::meshgrid(ranges, "ij");
    return {meshes[0], meshes[1]};
};

void Block4D_::Extend_T(int length_t) {
    this->data.index({at::indexing::Slice(length_t,data.size(0)),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(length_t-1,length_t),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice()}).repeat({data.size(0)-length_t,1,1,1});
 
}
void Block4D_::Extend_S(int length_s) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(length_s,data.size(1)),at::indexing::Slice(),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(length_s-1,length_s),at::indexing::Slice(),at::indexing::Slice()}).repeat({1,data.size(1)-length_s,1,1});
}
void Block4D_::Extend_V(int length_v) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_v,data.size(2)),at::indexing::Slice()}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_v-1,length_v),at::indexing::Slice()}).repeat({1,1,data.size(2)-length_v,1});
}
void Block4D_::Extend_U(int length_u) {
    this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_u,data.size(3))}) = this->data.index({at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(),at::indexing::Slice(length_u-1,length_u)}).repeat({1,1,1,data.size(1)-length_u});
}
void Block4D_::clip(int min, int max) {
    this->data = this->data.clamp(min,max);
}

Block4D_::operator at::Tensor() const{
    return this->data;
}

Block4D_::Block4D_(const at::Tensor& data){
    this->data = data;
}   
Block4D_::Block4D_(std::array<int64_t,4> size) {
    this->data = torch::zeros({size[0], size[1], size[2], size[3]}, torch::kInt);
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
    assert(B00.data.size(x1)+B10.data.size(x1) == B01.data.size(x1)+B11.data.size(x1) && "heights don't match" );
    assert(B00.data.size(x2)+B01.data.size(x2) == B10.data.size(x2)+B11.data.size(x2) && "widths don't match");
    this->data = torch::cat({torch::cat({B00.data, B10.data}, x1),torch::cat({B01.data, B11.data}, x1)},x2).to(at::kInt);
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
void Block4D_::CopySubblockFrom(const Block4D_& B, std::array<int64_t,4> sourceOffset, std::array<int64_t,4> targetOffset){
    
    Block4D_ deepCopy = B.clone(); 
    std::array<int64_t,4> length = {std::min(this->data.size(0) - targetOffset[0],B.data.size(0)),
                                    std::min(this->data.size(1) - targetOffset[1],B.data.size(1)),
                                    std::min(this->data.size(2) - targetOffset[2],B.data.size(2)),
                                    std::min(this->data.size(3) - targetOffset[3],B.data.size(3))};
    //std::cout<<"csf length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"original: "<<B.data.to(at::kDouble).sizes()<<std::endl;
       
    this->data.index({at::indexing::Slice(targetOffset[0],targetOffset[0] + length[0]),
                          at::indexing::Slice(targetOffset[1],targetOffset[1] + length[1] ),
                          at::indexing::Slice(targetOffset[2],targetOffset[2] + length[2]),
                          at::indexing::Slice(targetOffset[3],targetOffset[3] + length[3])}) =
                          deepCopy.data.index({at::indexing::Slice(sourceOffset[0],sourceOffset[0]+length[0]),
                                        at::indexing::Slice(sourceOffset[1],sourceOffset[1]+length[1]),
                                        at::indexing::Slice(sourceOffset[2],sourceOffset[2]+length[2]),
                                        at::indexing::Slice(sourceOffset[3],sourceOffset[3]+length[3])
                                        });
    //std::cout<<"indexedClone: "<<deepCopy.data.index({at::indexing::Slice(sourceOffset[0],this->data.size(0)),
                                        // at::indexing::Slice(sourceOffset[1],this->data.size(1)),
                                        // at::indexing::Slice(sourceOffset[2],this->data.size(2)),
                                        // at::indexing::Slice(sourceOffset[3],this->data.size(3))
                                        // }).to(at::kDouble).var()<<std::endl;                             
    //std::cout<<"copy: "<<this->data.to(at::kDouble).var()<<std::endl;

    
}
at::Tensor Block4D_::getSgtTransformMatrix(const at::Tensor& cov, bool isHorizontal){
    at::Tensor transform = klt(cov);
    //at::Tensor normalizedTransform = transform.clone();
    double eps = 1/sqrt(transform.size(0)) *1e-5;

    for(int i = 0; i < transform.size(1); i++){
        int bias = 0;
        while(true){
            //std::cout<<"Bias = bias"<<std::endl;
            double reference = transform[bias][i].item<double>();
            if(abs(reference) > eps){
                if(reference < 0){
                    transform.index({at::indexing::Slice(),i}) =  -1 * transform.index({at::indexing::Slice(),i});                }
                break;
            }

            bias++;
            //std::cout<<"Bias = "<<bias<<std::endl;

        }
    }
    //transform = zigZagTransformMatrix(transform,isHorizontal);
    return transform;
}
void Block4D_::sgtTransform(double scale){
    at::Tensor modelCovMatH = this->calcModelCovMatrix(ssi,true);

    at::Tensor modelCovMatV = this->calcModelCovMatrix(ssi,false);
    
    at::Tensor flatBlock = scale * getFlatBlock();



    at::Tensor sgtMatrixH   = getSgtTransformMatrix(modelCovMatH,true);
    at::Tensor sgtMatrixV =   getSgtTransformMatrix(modelCovMatV,false);
    at::Tensor flatTransform = sgt(flatBlock,sgtMatrixH,sgtMatrixV);
    this->data = (flat24D(flatTransform)).round().to(at::kInt).contiguous();
    this->sgtDomain = true;
    
}

at::Tensor Block4D_::getZigZagIndexes(std::array<int64_t,2> size){
    int64_t size2 = size[0]*size[1];

    at::Tensor indMatrix = at::range(0,size[0]*size[1]-1,1).reshape({size[0],size[1]});

    torch::TensorOptions options = torch::TensorOptions();
    at::Tensor zigZag = at::empty({0},options.dtype(at::kLong));

    for(int i = -(size[0] - 1); i < size[1] ; ++i){

        at::Tensor diag = at::diag(indMatrix.fliplr(),i).to(at::kLong);
        if(i%2 == 0){
            diag = diag.flip(0);
        }
        zigZag =at::cat({zigZag,diag},0);
    }
    zigZag = zigZag.flip(0);
    //std::cout<<zigZag.index({at::indexing::Slice(0,10)})<<std::endl;
    return zigZag;
}

at::Tensor Block4D_::getOrdered2DFromZigZagCoeffs(at::Tensor coeffs){
    std::array<int64_t,2> size = {data.size(0)*data.size(2),data.size(1)*data.size(3)};
    at::Tensor flatBlock = coeffs.reshape(size);
    //std::cout<<"data type = "<<flatBlock.dtype()<<std::endl;
    return flatBlock;
}
at::Tensor Block4D_::flatBlockFrom4DTensor(at::Tensor block){
    at::Tensor coeffs = diagonalOrder4DSampling(block);
    //std::cout<<"coeffs: "<<coeffs.index({at::indexing::Slice(0,10)})<<std::endl;
    coeffs = coeffs.index({getReverseZigZagIndexes({block.size(0)*block.size(2),block.size(1)*block.size(3)})});
    //std::cout<<"original coeffs: "<<coeffs.index({at::indexing::Slice(0,10)})<<std::endl;
    return getOrdered2DFromZigZagCoeffs(coeffs);
}
at::Tensor Block4D_::sgtFrom2DCoefficients(at::Tensor coefficients){
    at::Tensor indexes = getZigZagIndexes({coefficients.size(0),coefficients.size(1)});
    at::Tensor sgtCoefficients = coefficients.flatten().index({indexes});
    //std::cout<<" SGT Coeffs: "<<std::endl<<sgtCoefficients.index({at::indexing::Slice(0,10)})<<std::endl;
    //std::cout<<" SGT Orig Coeffs: "<<std::endl<<coefficients.flatten().index({at::indexing::Slice(0,10)})<<std::endl;

    return diagonalOrder4DBlock(sgtCoefficients);
}
at::Tensor Block4D_::diagonalOrder4DSampling(at::Tensor tensor4D){
   int maxSum = 0;
    std::array<int64_t,4> size = {data.size(0),data.size(1),data.size(2),data.size(3)};
    for(int i = 0; i < 4; i++){
        maxSum += size[i];
    }
    //std::cout<<"Max Sum = "<<maxSum<<std::endl;
    at::Tensor coefficients = at::zeros({size[0]*size[1]*size[2]*size[3]},at::kDouble);
    int i = 0;
    for(int sum = 0;sum<maxSum;sum++){
        for(int n = 0;n<size[0];n++){
            for(int m = 0;m<size[1];m++){
                for(int k = 0;k<size[3];k++){
                    for(int l = 0;l<size[2];l++){
                        int currentSum = m+n+k+l;
                        if(sum == currentSum){
                            coefficients[i] = tensor4D[m][n][k][l] ;
                            i++;
                        }
                    }
                }
            }
        }
    }
    if( i != coefficients.size(0)){
        std::cerr<<"ERROR:"<< i <<"!="<< coefficients.size(0)<<std::endl;
    }
    return coefficients;
}
at::Tensor Block4D_::diagonalOrder4DBlock(at::Tensor coefficients){
    int maxSum = 0;
    std::array<int64_t,4> size = {data.size(0),data.size(1),data.size(2),data.size(3)};
    for(int i = 0; i < 4; i++){
        maxSum += size[i];
    }
    //std::cout<<"Max Sum = "<<maxSum<<std::endl;
    at::Tensor tensor4D = at::zeros({size[0],size[1],size[2],size[3]});
    int i = 0;
    for(int sum = 0;sum<maxSum;sum++){
        for(int n = 0;n<size[0];n++){
            for(int m = 0;m<size[1];m++){
                for(int k = 0;k<size[3];k++){
                    for(int l = 0;l<size[2];l++){
                        int currentSum = m+n+k+l;
                        if(sum == currentSum){
                            tensor4D[m][n][k][l] = coefficients[i];
                            i++;
                        }
                    }
                }
            }
        }
    }
    if( i != coefficients.size(0)){
        std::cerr<<"ERROR:"<< i <<"!="<< coefficients.size(0)<<std::endl;
    }
    return tensor4D;
}
at::Tensor Block4D_::isgtTransformData(double scale, SgtSideInfo ssi) {
    at::Tensor modelCovMatH = this->calcModelCovMatrix(ssi,true);
    at::Tensor modelCovMatV = this->calcModelCovMatrix(ssi,false);  
    at::Tensor flatBlock = getFlatBlock().to(at::kDouble)/scale;
    at::Tensor sgtMatrixH   = getSgtTransformMatrix(modelCovMatH,true);
    at::Tensor sgtMatrixV = getSgtTransformMatrix(modelCovMatV,false);
    at::Tensor flatTransform = isgt(flatBlock,sgtMatrixH,sgtMatrixV);
    return flat24D(flatTransform);
}

at::Tensor Block4D_::klt(at::Tensor covMat){
    auto [L, Q] = torch::linalg::eigh(covMat, "U");
    return Q.flip({-1}); // flip such that coefficients are in DESCENDING order
}

at::Tensor Block4D_::sgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV){
    at::Tensor block = flatBlock.to(at::kDouble);
    
    //std::cout<<block.index({at::indexing::Slice()})<<std::endl;
    //std::cout<<"sgtMatrixV: "<<sgtMatrixV.mean({1}).index({at::indexing::Slice(0,10)})<<std::endl;

    at::Tensor hTransform = at::mm( block,sgtMatrixH);
    //std::cout<<"Transform: "<<hTransform.index({at::indexing::Slice(0,10),at::indexing::Slice(0,10)})<<std::endl;

    //std::cout<<"In SGT Function PartialCoefficient: "<<vTransform.mean({1}).index({at::indexing::Slice(0,10)})<<std::endl;

    at::Tensor transform = at::mm(at::mm(sgtMatrixV.t(), block), sgtMatrixH);

    //std::cout<<"In SGT Function Coefficient: "<<transform[0][2].item()<<std::endl;
    return transform;
}

at::Tensor Block4D_::isgt(const at::Tensor& flatBlock,const at::Tensor& sgtMatrixH, const at::Tensor& sgtMatrixV) {
    //std::cout<<"sgtMatrixV: "<<sgtMatrixV.mean({1}).index({at::indexing::Slice(0,10)})<<std::endl;
    at::Tensor transform = at::mm(at::mm(sgtMatrixV, flatBlock), sgtMatrixH.t()).round().to(at::kInt);
    return transform;
}
at::Tensor Block4D_::getReverseZigZagIndexes(std::array<int64_t,2> size){
    using namespace std;
    int64_t maxSize = max(size[0],size[1]);
    int64_t minSize = min(size[0],size[1]);
    at::Tensor cumm = at::zeros(size,at::kLong);
    int bias = 0;
    for (int64_t i = -size[0] + 1; i<size[1]; ++i){
        int diagSize = min(min(abs(i-size[1]),i+size[0]),minSize);
        at::Tensor diag = at::zeros(diagSize,at::kLong);
        for (int64_t j = 0; j<diagSize; ++j){
            int64_t m = min(i+size[0] -1,size[0] -1) - j;
            int64_t n = j+max((int64_t)0,i);
            if(i%2 == 0){
                cumm[m][n] = bias + (diagSize - 1 - j);
            }else{
                cumm[m][n] = bias + j;
            }
        }
        bias = bias + diagSize;
    }
    //cout<<"CUMM: "<<endl<<cumm.flatten().index({at::indexing::Slice(0,7)})<<endl;
    return cumm.flatten();
}
at::Tensor Block4D_::getTriangleIndexes(std::array<int64_t,2> size){
    using namespace std;
    int64_t maxSize = max(size[0],size[1]);
    int64_t minSize = min(size[0],size[1]);
    at::Tensor cumm = at::zeros(size,at::kLong);
    int bias = 0;
    for (int64_t i = -size[0] + 1; i<size[1]; ++i){
        int diagSize = min(min(abs(i-size[1]),i+size[0]),minSize);
        at::Tensor diag = at::zeros(diagSize,at::kLong);
        for (int64_t j = 0; j<diagSize; ++j){
            int64_t m = min(i+size[0] -1,size[0] -1) - j;
            int64_t n = j+max((int64_t)0,i);
            cumm[m][n] = bias + j;
        }
        bias = bias + diagSize;
    }
    return cumm.flatten();
}

at::Tensor Block4D_::zigZagTransformMatrix(at::Tensor transform,bool isHorizontal){
    std::array<int64_t,2> size;
    if(isHorizontal){
         size = {data.size(1),data.size(3)};
    }else{
        size = {data.size(0),data.size(2)};
    }
    at::Tensor zigZagIndices = getZigZagIndexes(size);
    return transform.index({at::indexing::Slice(),zigZagIndices});
}
at::Tensor Block4D_::triangleTransformMatrix(at::Tensor transform,bool isHorizontal){
    std::array<int64_t,2> size;
    if(isHorizontal){
         size = {data.size(1),data.size(3)};
    }else{
        size = {data.size(0),data.size(2)};
    }
    at::Tensor triangleIndices = getTriangleIndexes(size);
    return transform.index({at::indexing::Slice(),triangleIndices});
}
at::Tensor Block4D_::flat24D(const at::Tensor& flatBlock) const{
    at::Tensor block;
    if(includesInvalidCorners){
        block = flat24DValid(flatBlock);
    }else{
        block = flat24DAll(flatBlock);
    }
    return block;
}
at::Tensor Block4D_::flat24DAll(const at::Tensor& flatBlock) const{

    std::array<int64_t,4> sizeShifted = {data.size(1),data.size(3),data.size(0),data.size(2)};
    c10::IntArrayRef permutedSizes(sizeShifted);
    at::Tensor unflattened_block = flatBlock.t().reshape(permutedSizes);
    unflattened_block = unflattened_block.permute({2,0,3,1});   

    return  unflattened_block;
}
at::Tensor Block4D_::flat24DValid(const at::Tensor& flatBlock) const {
    at::Tensor padded_block = torch::zeros({data.size(1)*data.size(3),data.size(0)*data.size(2)}).to(at::kDouble);
    auto a = at::meshgrid({validPositions.valid_positions_v, validPositions.valid_positions_h},"ij");
    padded_block = padded_block.index_put({a[0],a[1]},flatBlock);
    std::array<int64_t,4> sizeShifted = {data.size(0),data.size(2),data.size(1),data.size(3)};
    c10::IntArrayRef permutedSizes(sizeShifted);
    at::Tensor unflattened_block = padded_block.t().reshape(permutedSizes);
    unflattened_block = unflattened_block.permute({2,0,3,1});
    return  unflattened_block;

}
at::Tensor Block4D_::getFlatBlock() {
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


at::Tensor Block4D_::getFlatBlockAll(){
    std::array<int64_t,2> dims,cDims;
    //The Goal is for the block to be LN x KL (H x W format)
    dims = {1,3}; 
    cDims = {0,2};
    //std::cout<<"FlatBlock: "<<dims[0]<<" "<<dims[1]<<" | "<<cDims[0]<<" "<<cDims[1]<<std::endl;
    const std::int64_t sample_rank = this->data.sizes().size();
    const auto t_dims = dims;
    std::array<int64_t,4> transposed_dims = {cDims[0],cDims[1],dims[0],dims[1]};
    std::cout<<"transposed_dims: "<<transposed_dims[0]<<" "<<transposed_dims[1]<<" | "<<transposed_dims[2]<<" "<<transposed_dims[3]<<std::endl;
    auto flatBlock = this->data.permute(transposed_dims).flatten(0,1).flatten(-dims.size());
    return flatBlock;

}

at::Tensor Block4D_::getFlatBlockValid(){
    std::array<int64_t,2> dims,cDims;
    //The Goal is for the block to be LN x KL (H x W format)
    at::Tensor flatBlock = getFlatBlockAll();
    flatBlock = flatBlock.index({at::indexing::Slice(), validPositions.valid_positions_h});
    flatBlock = flatBlock.index({validPositions.valid_positions_v,at::indexing::Slice()});
    return flatBlock;
}

void Block4D_::sgtTransform(double scale,std::array<double,2> dispRange){

    this->ssi = SgtSideInfo(*this,dispRange);
    // this->ssi.setDisparity(-this->ssi.getDisparity());
    // this->ssi.setRhoT(0.99);
    // this->ssi.setRhoS(0.99);
    // this->ssi.setRhoV(0.99);
    // this->ssi.setRhoU(0.99);
    
    //std::cout<<"d = "<<ssi.getDisparity()<<" rho_s = "<<ssi.getRhoS()<<" rho_t = "<<ssi.getRhoT()<<" rho_u = "<<ssi.getRhoU()<<" rho_v = "<<ssi.getRhoV()<<std::endl;
    sgtTransform(scale);
}
void Block4D_::isgtTransform(double scale,SgtSideInfo ssi){
    at::Tensor recoveredBlock = isgtTransformData(scale,ssi);
    this->data = recoveredBlock;
    this->sgtDomain = false;
    
}

double Block4D_::varianceFromCov(const at::Tensor& cov) {
    int64_t k_center = cov.size(0)/2;
    int64_t m_center = cov.size(1)/2; 
    double var = cov[k_center][m_center].item<double>(); 
    return var;
}

at::Tensor Block4D_::covFun(bool isHorizontal) const{
    using namespace phoenix::placeholders;
    using namespace torch::fft;
    using namespace at::indexing;
    

    std::array<int64_t,2> arrayDims, arraycDims;
    //std::cout<<"block: "<<std::endl<<this->data[0][1]<<std::endl;
    //std::cout << "block type = "<<data.dtype()<<std::endl;
    //std::cout<< "average = " <<data.mean(at::kDouble)<<std::endl;
    //std::cout<< "average Different= " <<data.to(at::kDouble).mean()<<std::endl;
    
    if(isHorizontal){
        arrayDims = {1,3};
        arraycDims = {0,2};
    }else{
        arrayDims = {0,2};
        arraycDims = {1,3};
        
    }
    at::IntArrayRef dims = arrayDims;
    at::IntArrayRef cDims = arraycDims;

    //std::cout<<"tDims = "<<dims<<std::endl;
    //std::cout<<"cDims = "<<cDims<<std::endl;

    // number of not batched dims
    const std::int64_t sample_rank = this->data.sizes().size();
    // compute average along unstacked sample dims
    auto mean = this->data.mean(cDims, /*keepdims=*/true,at::kDouble);
    //std::cout<<"mean: "<<mean.squeeze()<<std::endl;
    //std::cout<<"var mean: "<<mean.var()<<std::endl;


    auto dft_shape = dims
      | transformed([&](auto i){ return 2*this->data.size(i) - 1;})
      | to_t<int_arr_t>{};


    // dft of each block
    auto block_dft = fftn(this->data - mean, dft_shape, dims);

    // average power spectral density (biased)
    auto block_spd = block_dft.abs().pow(2).mean(cDims, true,at::kDouble); // keepdims here

    // invert cov and unshift along transformed axes
    auto cov = fftshift(at::real(ifftn(block_spd, {}, dims)), dims);
    // squeeze along reduced axes
    for (auto i : cDims | reversed)
      cov.squeeze_(i);

    //std::cout<<"cov raw: "<<cov<<std::endl;
    // compute ramp to compensate correlation average
    auto triang = [&](auto i){
      return at::concat({at::arange(1, this->data.size(i)), at::arange(this->data.size(i), 0, -1)});
    };
    auto ramp = boost::accumulate(dims | transformed(triang), at::ones({1}), bind(at::unsqueeze, _1, -1) * _2);

    return cov / ramp.squeeze(0);
}

SgtSideInfo::SgtSideInfo(int RhoSInt,int RhoTInt,int RhoUInt,int RhoVInt,int dInt,std::array<double,2> dispRange){
    this->rhoSInt = RhoSInt;
    this->rhoTInt = RhoTInt;
    this->rhoUInt = RhoUInt;
    this->rhoVInt = RhoVInt;
    this->disparityInt = dInt;
    this->disparityRange = dispRange;
}
SgtSideInfo::SgtSideInfo(std::array<double,2> dispRange){
    this->disparityRange = dispRange;
}


SgtSideInfo::SgtSideInfo(const Block4D_& block,std::array<double,2> dispRange){
    this->disparityRange = dispRange;
    estimateDisparity(block);
    //std::cout<<"Disparity = "<<this->getDisparity()<<std::endl;
    estimateRhos(block,1000);
}


int SgtSideInfo::codeRho(double rho) const{
    return rho * getRhoCodeScale() + getRhoCodeBias();
}
int SgtSideInfo::codeD(double d) const{
    return d * getDisparityCodeScale() + getDisparityCodeBias();
}

double SgtSideInfo::DecodeRho(int rhoCode) const{
    return (rhoCode - getRhoCodeBias()) / getRhoCodeScale();
}
double SgtSideInfo::DecodeD(int dCode) const{
    return (dCode - getDisparityCodeBias()) / getDisparityCodeScale();
}

void SgtSideInfo::setRhoSCode(int code){
    this->rhoSInt = code;
}
void SgtSideInfo::setRhoTCode(int code){
    this->rhoTInt = code;
}
void SgtSideInfo::setRhoUCode(int code){
    this->rhoUInt = code;
}
void SgtSideInfo::setRhoVCode(int code){
    this->rhoVInt = code;
}
void SgtSideInfo::setDCode(int code){
    this->disparityInt = code;
}
int SgtSideInfo::getRhoSCode(){
    return this->rhoSInt;
}
int SgtSideInfo::getRhoTCode(){
    return this->rhoTInt;
}
int SgtSideInfo::getRhoUCode(){
    return this->rhoUInt;
}
int SgtSideInfo::getRhoVCode(){
    return this->rhoVInt;
}
int SgtSideInfo::getDCode(){
    return this->disparityInt;
}
double SgtSideInfo::getRhoCodeBias() const{
    
    return -MIN_RHO/PRECISION_RHO;
    
}
double SgtSideInfo::getRhoCodeScale() const{
    return 1/PRECISION_RHO;
}
double SgtSideInfo::getDisparityCodeBias() const{   
    return -disparityRange[0]/PRECISION_D;
}
double SgtSideInfo::getDisparityCodeScale() const{
    return 1/PRECISION_D;
}
double SgtSideInfo::getRhoS(){
    double rhoS = DecodeRho(rhoSInt);
    return rhoS;
}
double SgtSideInfo::getRhoU(){
    double rhoU = DecodeRho(rhoUInt);
    return rhoU;
}
double SgtSideInfo::getRhoT(){
    double rhoT = DecodeRho(rhoTInt);
    return rhoT;
}
double SgtSideInfo::getRhoV(){
    double rhoV = DecodeRho(rhoVInt);
    return rhoV;
}
double SgtSideInfo::getDisparity(){
     double disparity = DecodeD(disparityInt);
    return disparity;
}
int SgtSideInfo::getRhoPrecision() const{
    return std::ceil(log2(codeRho(MAX_RHO)));
}
int SgtSideInfo::getDisparityPrecision() const{
    return std::ceil(log2(codeRho(this->disparityRange[1])));
}
void SgtSideInfo::setRhoS(double rhoS){
    rhoS = std::clamp(rhoS,MIN_RHO,MAX_RHO);
    this->rhoSInt = codeRho(rhoS);
}
void SgtSideInfo::setRhoU(double rhoU){
    rhoU = std::clamp(rhoU,MIN_RHO,MAX_RHO);
    this->rhoUInt = codeRho(rhoU);

}
void SgtSideInfo::setRhoT(double rhoT){
    rhoT = std::clamp(rhoT,MIN_RHO,MAX_RHO);
    this->rhoTInt = codeRho(rhoT);

}
void SgtSideInfo::setRhoV(double rhoV){
    rhoV = std::clamp(rhoV,MIN_RHO,MAX_RHO);
    this->rhoVInt = codeRho(rhoV);
}
void SgtSideInfo::setDisparity(double d){
    this->disparityInt =codeD(d);
}
void SgtSideInfo::setSpatialRhos(const at::Tensor& covFunH, const at::Tensor& covFunV){
    int64_t k_center = covFunH.size(0)/2;
    int64_t m_center = covFunH.size(1)/2; 
    double varH = covFunH[k_center][m_center].item<double>(); 
    setRhoU((covFunH[k_center][m_center+1]/varH).item<double>());
    int64_t l_center = covFunV.size(0)/2;
    int64_t n_center = covFunV.size(1)/2; 
    double varV= covFunV[l_center][n_center].item<double>(); 
    setRhoV( (covFunV[l_center][n_center+1]/varV).item<double>());
}

void SgtSideInfo::setSpatialRhos(const double rhoU, const double rhoV){
    setRhoU(rhoU);
    setRhoV(rhoV);
}
void SgtSideInfo::setAngularRhos(const double rhoS, const double rhoT){
    setRhoS(rhoS);
    setRhoT(rhoT);
}

at::Tensor Block4D_::calcModelCovMatrix(SgtSideInfo ssi,bool isHorizontal) const{
    double rhoSpt, rhoAng;
    int64_t sizeSpt,sizeAng;
    if(isHorizontal){
        rhoSpt = ssi.getRhoU();
        sizeSpt = data.size(3);
        rhoAng = ssi.getRhoS();
        sizeAng = data.size(1);

    }else{
        rhoSpt = ssi.getRhoV();
        sizeSpt = data.size(2);
        rhoAng = ssi.getRhoT(); 
        sizeAng = data.size(0);
    }
    auto [s,u] = make_function_grid({sizeAng, sizeSpt});
    

    at::Tensor modelCovFun = torch::pow(rhoSpt,(at::abs(u - ssi.getDisparity() * s))) * torch::pow(rhoAng,s.abs());

    at::Tensor modelCovMat = covFun2Mat(modelCovFun,isHorizontal);
    
    //modelCovMat = modelCovMat.round(20);
    
    // if(ssi.getDCode() == 45 && ssi.getRhoUCode() == 99){
    // if(isHorizontal){
        
    //     std::cout<<"HORIZONTAL MODEL COV: "<<modelCovFun.sizes()<<std::endl;
    //     std::cout<<modelCovFun.index({at::indexing::Slice(8,15),at::indexing::Slice(63,69)})<<std::endl;
    // }else{
    //     std::cout<<"Vertical MODEL COV: "<<modelCovFun.sizes()<<std::endl;
    //     std::cout<<modelCovFun.index({at::indexing::Slice(8,15),at::indexing::Slice(63,69)})<<std::endl;
    // }
    // }

    return modelCovMat;
}

at::Tensor Block4D_::normalizeCov(at::Tensor cov){
    //Get the index of the central value (variance)
    int64_t s_center = cov.sizes()[0]/2;
     int64_t u_center = cov.sizes()[1]/2;
     //normalize so variance = 1
     cov = cov / cov[s_center][u_center];
    return cov;
}
void SgtSideInfo::estimateDisparity(const Block4D_& block){
    at::Tensor covFunH = Block4D_::normalizeCov(block.covFun(true));
    at::Tensor covFunV = Block4D_::normalizeCov(block.covFun(false));

    SgtSideInfo temp;
    temp.setSpatialRhos(covFunH,covFunV);
    temp.setAngularRhos();
    //std::cout<<"RhoU = "<<temp.getRhoU()<<" RhoV = "<<temp.getRhoV()<<std::endl;

    at::Tensor iSqrtCovMatH = block.iSqrtCovMat(true);
    //std::cout<<"iSqrt: "<<std::endl<<iSqrtCovMatH[0]<<std::endl<<std::endl<<std::endl;
    at::Tensor iSqrtCovMatV = block.iSqrtCovMat(false);

    double min = std::numeric_limits<double>::max();
    for(double d =this->disparityRange[0];d<=this->disparityRange[1];d+=PRECISION_D){
        temp.setDisparity(d);
        at::Tensor modelCovMatH = block.calcModelCovMatrix(temp,true);
        at::Tensor modelCovMatV = block.calcModelCovMatrix(temp,false);
        // if(d == -3){
        //     std::cout<<modelCovMatH<<std::endl;
        // }
        double genDivH = genDivergence(modelCovMatH,iSqrtCovMatH);
        double genDivV = genDivergence(modelCovMatV,iSqrtCovMatV);
        double genDiv = genDivH + genDivV;
        //std::cout<<d<<":"<<genDiv<<" "<<genDivH<<" "<<genDivV<<std::endl;
        if (genDiv < min){
            min = genDiv;
            setDisparity(d);
        }
    }
}

double SgtSideInfo::genDivergence(const at::Tensor& p, const at::Tensor& qRsqrt){
    auto prod = at::einsum("...xm,...xy,...yn->...mn", {qRsqrt, p, qRsqrt});
    auto eigvals = torch::linalg::eigvalsh(prod, "U");
    return (eigvals.mean(-1).log() - eigvals.log().mean(-1)).item<double>();
}
at::Tensor Block4D_::iSqrtCovMat(bool isHorizontal ) const{
    at::Tensor covFun = normalizeCov(this->covFun(true));
    at::Tensor covMat = covFun2Mat(covFun,isHorizontal);
    auto [L, Q] = torch::linalg::eigh(covMat, "U");
    // get largest eigenvalue of each matrix
    auto&& [Lmax, Lloc] = L.max(-1, /*keepdims*/true);
    auto Lrel = L / Lmax; // 'relative' eigenvalues
    // take worst case across batches for condition number
    auto&& [Lmin, min_where]  = L.view({-1, L.size(-1)}).min(0);
    // indices of eigenpairs to keep
    auto Lkeep = Lmin > 1e-4;
    // take
    auto Lselect = L.index({"...", Lkeep});
    auto Qselect = Q.index({"...", Lkeep});

    // safely take the inverse sqrt
    auto Lrsqrt = Lselect.rsqrt_();
    auto result = at::einsum("...x,...x->...x", {Qselect, Lrsqrt});
    return result;

}

at::Tensor Block4D_::covFun2Mat(const at::Tensor& covFun,bool isHorizontal) const{
    at::Tensor covMat;
    if(includesInvalidCorners){
        //std::cout<<"Invalid Corners?????????"<<std::endl;
        covMat = covFun2MatValid(covFun,isHorizontal);
    }else{
        //std::cout<<"Correctimundo"<<std::endl;
        covMat = covFun2MatAll(covFun);
    }
    return covMat;
}
at::Tensor Block4D_::covFun2MatAll(const at::Tensor& covFun) const{

    std::int64_t sample_rank = covFun.sizes().size();

    // map from 0...size[i]
    tensor_arr_t aranges;
    for (auto i : irange(sample_rank))
      aranges.push_back(at::arange((covFun.size(i) + 1)/2));

    // stacked indices of proper shape
    auto cart = at::cartesian_prod(aranges).t();

    // initialize indexing vector with "..."
    std::vector<at::indexing::TensorIndex> indexing {"..."};
    for(auto i : irange(sample_rank)) {
      auto offset = (covFun.size(i) + 1) / 2 - 1;
      auto dim_index = cart[i].unsqueeze(-1) - cart[i] + offset;
      indexing.push_back(dim_index);
    }
    at::Tensor cov_mat = covFun.index(indexing);
    return cov_mat;
}
at::Tensor Block4D_::covFun2MatValid(const at::Tensor& covFun,bool isHorizontal) const {
    at::Tensor validIndexes;
    if(isHorizontal){
        validIndexes = this->validPositions.valid_positions_h;
    }else{
        validIndexes = this->validPositions.valid_positions_v;
    }
    at::Tensor covMat = covFun2MatAll(covFun);
    covMat = covMat.index({at::indexing::Slice(), validIndexes});
    covMat = covMat.index({validIndexes,at::indexing::Slice()});
    return covMat;
  }
  at::Tensor batched_cov_fun_to_mat(const at::Tensor& cov_fun, std::int64_t batch_rank) {

    std::int64_t sample_rank = cov_fun.sizes().size() - batch_rank;

    // map from 0...size[i]
    tensor_arr_t aranges;
    for (auto i : irange(sample_rank))
      aranges.push_back(at::arange((cov_fun.size(batch_rank + i) + 1)/2));

    // stacked indices of proper shape
    auto cart = at::cartesian_prod(aranges).t();

    // initialize indexing vector with "..."
    std::vector<at::indexing::TensorIndex> indexing {"..."};
    for(auto i : irange(sample_rank)) {
      auto offset = (cov_fun.size(batch_rank + i) + 1) / 2 - 1;
      auto dim_index = cart[i].unsqueeze(-1) - cart[i] + offset;
      indexing.push_back(dim_index);
    }

    return cov_fun.index(indexing);
  } 

void SgtSideInfo::estimateRhos(const Block4D_& block, double varianceThreshold){
    at::Tensor covFunH = block.covFun(true);
    at::Tensor covFunV = block.covFun(false);
    double varH = Block4D_::varianceFromCov(covFunH);
    double varV = Block4D_::varianceFromCov(covFunV);
    //std::cout<<"VarH = "<<varH<<" VarV = "<<varV<<std::endl;
    //if(varH < varianceThreshold){
    if(true){
        setRhoS(FIXED_ANGULAR_RHO);
        setRhoU(FIXED_SPATIAL_RHO);
    }else{
        estimateRhosLS(covFunH,true);
    }
    //if(varV < varianceThreshold){
    if(true){
        setRhoT(FIXED_ANGULAR_RHO);
        setRhoV(FIXED_SPATIAL_RHO);
    }else{
        estimateRhosLS(covFunV,false);
    }
    //std::cout<<"Finished Estimating Rhos"<<std::endl;
}

void SgtSideInfo::estimateRhosLS(const at::Tensor& cov, bool isHorizontal){
    int64_t k_center = cov.size(0)/2;
    int64_t m_center = cov.size(1)/2; 
    int64_t m_samples = std::min(4,(int)m_center);
    int64_t k_samples = std::min(2,(int)m_center);
    at::Tensor normalized_cov = cov.clone();
    normalized_cov /= cov[k_center][m_center]; 
    //std::cout<<"cov size: "<<normalized_cov.sizes()<<" m_center = "<<m_center<<std::endl;
    at::Tensor working_cov = at::clamp(normalized_cov.index({at::indexing::Slice(k_center - k_samples,k_center + k_samples+1),
                                                            at::indexing::Slice(m_center-m_samples,m_center+m_samples+1)}),0.01,2);
    //std::cout<<working_cov.sizes()<<std::endl;
    at::Tensor m = torch::arange(-m_samples,m_samples +1);
    at::Tensor k = torch::arange(- k_samples,k_samples+1);
    auto meshes = at::meshgrid({k.to(at::kDouble),m.to(at::kDouble)}, "ij");

    double best_alpha;
    auto b = log(working_cov).flatten().unsqueeze(1);
    at::Tensor temp = abs(meshes[1] - meshes[0]*this->getDisparity() );
    auto A = torch::cat({abs(meshes[0].flatten().unsqueeze(1)),temp.flatten().unsqueeze(1)},1); 


    auto A_t = A.t();
    //std::cout<<A.sizes()<<" "<<b.sizes()<<std::endl;
    auto beta = at::clamp(matmul(matmul(inverse(matmul(A_t,A)),A_t),b),log(0.01),log(0.999));

    //std::cout<<beta.sizes()<<std::endl;

    auto r = ((matmul(A,beta) - b)*(matmul(A,beta) - b));
    double current_cost = (r.sum()/(r.size(0)-1)).item<double>();
    at::Tensor best_beta = beta.exp();
    //std::cout<<"Rhos = "<<beta<<std::endl;
    if(isHorizontal){
    setRhoS(best_beta[0].item<double>());
    setRhoU(best_beta[1].item<double>());
    }else{
    setRhoT(best_beta[0].item<double>());
    setRhoV(best_beta[1].item<double>());
    }
}




void Block4D_::Ones(){
    this->data = torch::ones(this->data.sizes(),this->data.dtype());
}
void Block4D_::Zeros(){
    this->data = torch::zeros(this->data.sizes(),this->data.dtype());
}
void Block4D_::Shift_UVPlane(int shift, int position_t, int position_s){
    if(shift > 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_left_shift(shift);
    }
    if(shift < 0){
        this->data[position_t][position_s] = this->data[position_t][position_s].bitwise_right_shift(-shift);
    }
}
void Block4D_::YCbCr2RGB_BT601(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Cb, Block4D_ const &Cr, int Scale) {
    // std::cout<<"Data type = "<<Y.data.dtype()<<std::endl;
    // std::cout<<"Decoding Scale = "<<Scale<<std::endl;
    // std::cout<<"Chrominance Bias Correction: "<<((Scale+1)/2)<<std::endl;
    at::Tensor Ytemp = Y.data.to(at::kDouble);
    auto CbTemp = (Cb.data - ((Scale+1)/2)).to(at::kDouble);
    auto CrTemp = (Cr.data - ((Scale+1)/2)).to(at::kDouble);

    //std::cout<<Ytemp[0][0][0][0].item()<<" "<<CbTemp[0][0][0][0].item()<<" "<<CrTemp[0][0][0][0].item()<<std::endl;
    R = (Ytemp - 0.0000071525 * CbTemp + 1.4020 * CrTemp).round().to(at::kInt);
    G = (Ytemp- 0.34413 * CbTemp - 0.71414 * CrTemp).round().to(at::kInt);
    B = (Ytemp + 1.7720 * CbTemp - 0.000040249 * CrTemp).round().to(at::kInt);
   //std::cout<<Ytemp[0][0][0][0].item()<<" "<<-0.34413 * CbTemp[0][0][0][0].item<double>()<<" "<<- 0.71414 *(CrTemp[0][0][0][0].item<double>())<<" "<<G.data[0][0][0][0]<<std::endl;

}

void Block4D_::YCoCg2RGB(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Co, Block4D_ const &Cg, int Scale) {
    auto CoTemp = Co.data - (Scale+1)/2;
    auto CgTemp = Cg.data - (Scale+1)/2;
    auto t = Y - (CgTemp.bitwise_right_shift(1));
    G = CgTemp + t;
    B.data = t - (CoTemp.bitwise_right_shift(1));
    R.data = B.data + CoTemp;          
}
void Block4D_::RGB2YCbCr_BT601(Block4D_ &Y, Block4D_ &Cb, Block4D_ &Cr, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
    static const auto Y_weights = at::tensor({0.299, 0.587, 0.114}, at::kDouble).reshape({3, 1});  
    static const auto Cb_weights = at::tensor({-0.168736, -0.331264, 0.5}, at::kDouble).reshape({3, 1}); 
    static const auto Cr_weights = at::tensor({0.5, -0.418688, -0.081312}, at::kDouble).reshape({3, 1}); 
    static const int D = 1<<((int)log2(Scale+1)-8);
    static const int Y8bitBias = 0;
    static const int CbCr8bitBias = (1<<9);
    //std::cout<<"D = "<<D<<" CbCr8bitBias = "<<CbCr8bitBias<<" Y8bitBias = "<<Y8bitBias<<std::endl;

    auto Ey = R.data.to(at::kDouble)/Scale * Y_weights[0] + G.data.to(at::kDouble)/Scale * Y_weights[1] + B.data.to(at::kDouble)/Scale * Y_weights[2];
    //std::cout<<(R.data.to(at::kDouble)/Scale)[0][0][0][0].item()<<" "<<Y_weights[0].item()<<" Ey min = "<<Ey.min().item()<<" max = "<<Ey.max().item()<<std::endl;

    Y.data = ((1023 * Ey + Y8bitBias)).round().to(at::kInt);

    auto Ecb = R.data.to(at::kDouble)/Scale * Cb_weights[0] + G.data.to(at::kDouble)/Scale * Cb_weights[1] + B.data.to(at::kDouble)/Scale * Cb_weights[2];
    //std::cout<<"ECb min = "<<Ecb.min().item()<<" max = "<<Ecb.max().item()<<std::endl;
    Cb.data = ((1023 * Ecb + CbCr8bitBias)).round().to(at::kInt);
    auto Ecr = R.data.to(at::kDouble)/Scale * Cr_weights[0] + G.data.to(at::kDouble)/Scale * Cr_weights[1] + B.data.to(at::kDouble)/Scale * Cr_weights[2];
    //std::cout<<(R.data.to(at::kDouble)/Scale)[0][0][0][0].item()<<" "<<Cr_weights[0]<<" Ecr min = "<<Ecr.min().item()<<" max = "<<Ecr.max().item()<<std::endl;
    
    Cr.data = ((1023 * Ecr + CbCr8bitBias)).round().to(at::kInt); 
    Y.validPositions = R.validPositions;
    Cb.validPositions = R.validPositions;
    Cr.validPositions = R.validPositions;
}

void Block4D_::RGB2YCoCg(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
    Co = R.data - B.data;
    auto temp = B.data + Co.data.bitwise_right_shift(1);
    Cg = G.data - temp;
    Y = temp + Cg.data.bitwise_right_shift(1);
    Co.data+= (Scale + 1)/2;
    Cg.data+= (Scale + 1)/2;
    Y.validPositions = R.validPositions;
    Co.validPositions = R.validPositions;
    Cg.validPositions = R.validPositions;        
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




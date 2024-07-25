#include <LightField/LightField.h>
#include <LightField/Block4D.h>
#include <OldDCT/MultiscaleTransform.h>
#include <Encoder/Hierarchical4DEncoder.h>
#include <Encoder/TransformPartition.h>
//#include <LightField/Block4D_.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string.h>
#include <torch/torch.h>
#include "Decoder/PartitionDecoder.h"
#include <Decoder/Hierarchical4DDecoder.h>


using namespace std;

double mse(const at::Tensor& a, const at::Tensor& b) {
    //cout<<"MSE CALCULATION!"<<endl;
    auto mse =(a - b).pow(2).to(at::kDouble).mean(); 
    //cout<<"MSE CALCULATION Complete!"<<endl;

    return mse.item<double>();
};
double totalTransformGain(std::array<int64_t,4> length){
    
    double transformGain = 1;
    std::array<int64_t,4> maxLength = {9,9,64,64};
    for(int i = 0; i < 4; i++){
        transformGain*=length[i]/sqrt(length[i]);
        transformGain  *= sqrt(maxLength[i]/length[i]);
    }
    return transformGain*1;

}
void RGB2YCoCg(Block4D &Y, Block4D &Co, Block4D &Cg, Block4D const &R, Block4D const &G, Block4D const &B, int Scale) {
    
    for(int n = 0; n < R.mlength_t*R.mlength_s*R.mlength_v*R.mlength_u; n++) {
        int t;
        Co.mPixelData[n] = R.mPixelData[n] - B.mPixelData[n];
        t = B.mPixelData[n] + (Co.mPixelData[n]>>1);
        Cg.mPixelData[n] = G.mPixelData[n] - t;
        Y.mPixelData[n] = t + (Cg.mPixelData[n]>>1);
        Co.mPixelData[n] += (Scale + 1)/2;
        Cg.mPixelData[n] += (Scale + 1)/2;
    }
        
}
void RGB2YCoCg_(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
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
void YCoCg2RGB_(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Co, Block4D_ const &Cg, int Scale) {
    auto CoTemp = Co.data - (Scale+1)/2;
    auto CgTemp = Cg.data - (Scale+1)/2;
    auto t = Y - (CgTemp.bitwise_right_shift(1));
    G = CgTemp + t;
    B.data = t - (CoTemp.bitwise_right_shift(1));
    R.data = B.data + CoTemp;          
}

// TEST(EncoderTests,OldMuleEncode){
    
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     int s = 4;
//     int numberOfCacheViewLines = s;
//     LightField inputLF(9,9,numberOfCacheViewLines);
//     Block4D lfBlock, rBlock, gBlock, bBlock, yBlock, cbBlock, crBlock;
//     Hierarchical4DEncoder hdt;
//     TransformPartition tp;
//     tp.mPartitionData.SetDimension(9,9,s,s);
//     tp.mlength_t_min = 9;
//     tp.mlength_s_min = 9;
//     tp.mlength_v_min = s;
//     tp.mlength_u_min = s;
//     char inputDirectory_c_str[1024];
//     char outputString[1024];
//     strcpy(inputDirectory_c_str, inputDirectory.c_str());
//     strcpy(outputString, "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/mule-org.comp");
//     lfBlock.SetDimension(9,9,s,s);
//     rBlock.SetDimension(9,9,s,s);
//     gBlock.SetDimension(9,9,s,s);
//     bBlock.SetDimension(9,9,s,s);
//     yBlock.SetDimension(9,9,s,s);
//     cbBlock.SetDimension(9,9,s,s);
//     crBlock.SetDimension(9,9,s,s);
//     inputLF.mVerticalViewNumberOffset = 0;
//     inputLF.mHorizontalViewNumberOffset = 0;
//     inputLF.OpenLightFieldPPM(inputDirectory_c_str, ".ppm", 9, 9, 3, 3, 'r');
//     int extensionLength_t = 0;
//     int extensionLength_s = 0;
//     int extensionLength_u = 0;
//     int extensionLength_v = 0;

//     MultiscaleTransform DCTarray;
//     DCTarray.SetDimension(9, 9, s, s);
//     FILE *outputFileNamePointer;
//     if((outputFileNamePointer = fopen(outputString, "wb")) == NULL) {
//         printf("Error: input file %s not found\n", outputString);
//         exit(0);
//     }
//     hdt.StartEncoder(outputFileNamePointer);
//     rBlock.Zeros();
//     gBlock.Zeros();
//     bBlock.Zeros();
//     inputLF.ReadBlock4DfromLightField(&rBlock, 0, 0, 0, 0, 0);
//     inputLF.ReadBlock4DfromLightField(&gBlock, 0, 0, 0, 0, 1);
//     inputLF.ReadBlock4DfromLightField(&bBlock, 0, 0, 0, 0, 2);
//     RGB2YCoCg(yBlock, cbBlock, crBlock, rBlock, gBlock, bBlock, inputLF.mPGMScale);
//     lfBlock.CopySubblockFrom(yBlock, 0, 0, 0, 0);
//     lfBlock = lfBlock - (inputLF.mPGMScale+1)/2;
//     hdt.RestartProbabilisticModel();
//     tp.RDoptimizeTransform(lfBlock, DCTarray, hdt, 1);
//     tp.EncodePartition(hdt, 1);
// }
// TEST(EncoderTests,Dummy){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> position = {0,0,64,128};
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,2);

//     LightField outputLF({9,9,64,64,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');
    
// }

int fib(int64_t n){
    if(n == 0) return 0;
    if(n == 1) return 1;
    return fib(n-1) + fib(n-2);
}
at::Tensor getZigZagIndexes2(std::array<int64_t,2> size){
    int64_t maxSize = max(size[0],size[1]);
    int64_t minSize = min(size[0],size[1]);
    at::Tensor cumm = at::zeros(size,at::kLong);
    int bias = 0;
    for (int64_t i = -size[0] + 1; i<size[1]; ++i){
        int diagSize = min(min(abs(i-size[1]),i+size[0]),minSize);
        cout<<"Diag Size: "<<diagSize<< endl;
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
    cout<<"CUMM: "<<endl<<cumm.index({at::indexing::Slice(),at::indexing::Slice(0,7)})<<endl;
    return cumm.flatten();
}
at::Tensor getZigZagIndexes1(std::array<int64_t,2> size){
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
    //cout<<zigZag<<endl;
    return zigZag;
}
at::Tensor zigZagTransformMatrix(at::Tensor transform,std::array<int64_t,2> size){
    at::Tensor zigZagIndices = getZigZagIndexes2(size);
    cout<<"oiii"<<endl;
    return transform.index({at::indexing::Slice(),zigZagIndices});
}

at::Tensor diagonalOrder4DSampling(at::Tensor coefficients, std::array<int64_t,4> size){
    int maxSum = 0;
    for(int i = 0; i < 4; i++){
        maxSum += size[i];
    }
    std::cout<<"Max Sum = "<<maxSum<<std::endl;
    at::Tensor tensor4D = at::zeros({size[0],size[1],size[2],size[3]});
    int i = 0;
    for(int sum = 0;sum<maxSum;sum++){
        for(int n = 0;n<size[0];n++){
            for(int m = 0;m<size[1];m++){
                for(int k = 0;k<size[3];k++){
                    for(int l = 0;l<size[2];l++){
                        int currentSum = m+n+k+l;
                        if(sum == currentSum){
                            cout<<sum<<" "<<currentSum<<endl;
                            tensor4D[m][n][k][l] = coefficients[i];
                            i++;
                        }
                    }
                }
            }
        }
    }
    if( i != coefficients.size(0)){
        cout<<"ERROR:"<< i <<"!="<< coefficients.size(0)<<endl;
    }else{
        cout<<"Success i = "<<i<<endl;
    }
    return tensor4D;
}

TEST(EncoderTests,TriangleTest){
    Block4D_ isux;
    at::Tensor flatBlock = at::range(15,0,-1).reshape({4,4});
        std::cout <<flatBlock;

    at::Tensor indexes = isux.getZigZagIndexes({4,4});
    at::Tensor flatBlock2 = flatBlock.index({indexes});
    std::cout <<flatBlock2;
    //std::cout <<flatBlock;

}
// TEST(EncoderTests,ZigZag){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-COEFFICIENTS-MUTILATED/";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> position = {0,0,64,128};
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);



//     blockY.data = blockY.data - 512;
//     SgtSideInfo ssi(blockY,{-3.5,3.5});

//     at::Tensor modelCovMat = blockY.calcModelCovMatrix(ssi,true);
//     cout<<"modelCovMat size = "<<modelCovMat.sizes()<<endl;
//     auto[ L, Q] = torch::linalg::eigh(modelCovMat,"U");
//     cout<<"Q size = "<<Q.sizes()<<std::endl;
//     Q = Q.flip({-1});
//     at::Tensor flatBlock = blockY.getFlatBlock();

//     auto A = flatBlock.index({0,at::indexing::Slice()}).unsqueeze(0);
//     //auto sgtOrig = at::mm(A.to(at::kDouble),Q);
//     //sgtOrig = sgtOrig.reshape({blockY.data.size(1),blockY.data.size(3)});
    
//     auto QZigZag = zigZagTransformMatrix(Q,{blockY.data.size(1),blockY.data.size(3)});
//     //auto sgtZigZag = at::mm(A.to(at::kDouble),QZigZag);

//     //sgtZigZag = sgtZigZag.reshape({blockY.data.size(1),blockY.data.size(3)});
//     auto sgtOrig = at::mm(Q.t(),at::mm(flatBlock.to(at::kDouble),Q));
//     auto sgtZigZag = at::mm(QZigZag.t(),at::mm(flatBlock.to(at::kDouble),QZigZag));
//     //at::Tensor indMatrixZigZag = indMatrix.flatten().index({zigZag}).reshape(indMatrix.sizes());
//     //auto test = at::range(0,blockY.data.size(1)*blockY.data.size(3)-1,1).reshape({blockY.data.size(1),blockY.data.size(3)});
//     //auto testZigZag = test.flatten().index({getZigZagIndexes2({blockY.data.size(1),blockY.data.size(3)})}).reshape({blockY.data.size(1),blockY.data.size(3)});
//     auto sgtZigZagVec = sgtOrig.flatten().index({getZigZagIndexes1({sgtOrig.size(0),sgtOrig.size(1)})});

//     auto properDiagonalTensor = diagonalOrder4DSampling(sgtZigZagVec,{9,9,64,64});
//     auto currentTensor = blockY.flat24D(sgtOrig);
//     auto maybeWrongTryTensor = blockY.flat24D(sgtZigZag);

//     cout<<"View"<<" "<<properDiagonalTensor.index({0,0,at::indexing::Slice(0,6),at::indexing::Slice(0,6)}).abs().mean().item()<<std::endl<<properDiagonalTensor.index({0,0,at::indexing::Slice(0,6),at::indexing::Slice(0,6)})<<std::endl;
//     cout<<"ProjectionPlaneImage "<<properDiagonalTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),0,0}).abs().mean().item()<<std::endl<<properDiagonalTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),0,0})<<std::endl;
//     cout<<"EPI Horiz "<<properDiagonalTensor.index({0,at::indexing::Slice(0,6),0,at::indexing::Slice(0,6)}).abs().mean().item()<<std::endl<<properDiagonalTensor.index({0,at::indexing::Slice(0,6),0,at::indexing::Slice(0,6)})<<std::endl;
//     cout<<"EPI Vert "<<properDiagonalTensor.index({at::indexing::Slice(0,6),0,at::indexing::Slice(0,6),0}).abs().mean().item()<<std::endl<<properDiagonalTensor.index({at::indexing::Slice(0,6),0,at::indexing::Slice(0,6),0})<<std::endl<<std::endl;
//     cout<<"View Old "<<currentTensor.index({0,0,at::indexing::Slice(0,6),at::indexing::Slice(0,6)}).abs().mean().item()<<std::endl<<currentTensor.index({0,0,at::indexing::Slice(0,6),at::indexing::Slice(0,6)})<<std::endl;
//     cout<<"ProjectionPlaneImage Old "<<currentTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),0,0}).abs().mean().item()<<std::endl<<currentTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),0,0})<<std::endl;
//     cout<<"EPI Horiz Old "<<currentTensor.index({0,at::indexing::Slice(0,6),0,at::indexing::Slice(0,6)}).abs().mean().item()<<std::endl<<currentTensor.index({0,at::indexing::Slice(0,6),0,at::indexing::Slice(0,6)})<<std::endl;
//     cout<<"EPI Vert 0ld "<<currentTensor.index({at::indexing::Slice(0,6),0,at::indexing::Slice(0,6),0}).abs().mean().item()<<std::endl<<currentTensor.index({at::indexing::Slice(0,6),0,at::indexing::Slice(0,6),0})<<std::endl;
    
//     cout<<"View Wrong "<<maybeWrongTryTensor.index({0,0,at::indexing::Slice(0,6),at::indexing::Slice(0,6)}).abs().mean().item()<<std::endl<<maybeWrongTryTensor.index({0,0,at::indexing::Slice(0,6),at::indexing::Slice(0,6)})<<std::endl;
//     cout<<"ProjectionPlaneImage Wrong "<<maybeWrongTryTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),0,0}).abs().mean().item()<<std::endl<<maybeWrongTryTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),0,0})<<std::endl;
//     cout<<"EPI Horiz Wrong "<<maybeWrongTryTensor.index({0,at::indexing::Slice(0,6),0,at::indexing::Slice(0,6)}).abs().mean().item()<<std::endl<<maybeWrongTryTensor.index({0,at::indexing::Slice(0,6),0,at::indexing::Slice(0,6)})<<std::endl;
//     cout<<"EPI Vert Wrong "<<maybeWrongTryTensor.index({at::indexing::Slice(0,6),0,at::indexing::Slice(0,6),0}).abs().mean().item()<<std::endl<<maybeWrongTryTensor.index({at::indexing::Slice(0,6),0,at::indexing::Slice(0,6),0})<<std::endl;
    

//     cout<<"Proper Method \"Top Left\" Concentration = "<<properDiagonalTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),at::indexing::Slice(0,6),at::indexing::Slice(0,6)}).pow(2).mean().item()<<std::endl;
//     cout<<"Naive Method \"Top Left\" Concentration = "<<currentTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),at::indexing::Slice(0,6),at::indexing::Slice(0,6)}).pow(2).mean().item()<<std::endl;
//     cout<<"EPI ZigZag \"Top Left\" Concentration = "<<maybeWrongTryTensor.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6),at::indexing::Slice(0,6),at::indexing::Slice(0,6)}).pow(2).mean().item()<<std::endl;
    
//     //cout<<"test"<<std::endl<<test.index({at::indexing::Slice(),at::indexing::Slice(0,7)})<<std::endl;
//     //cout<<"testZigZag"<<std::endl<<testZigZag.index({at::indexing::Slice(),at::indexing::Slice(0,7)})<<std::endl;
// }
// TEST(EncoderTests,Quantize){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-COEFFICIENTS-MUTILATED/";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> position = {0,0,64,128};
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,64,64},position,2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     blockY.data = blockY.data - 512;
//     blockCb.data = blockCb.data - 512;
//     blockCr.data = blockCr.data - 512;

//     std::cout<<"First value: "<<blockY.data[0][0][0][0].item()<<std::endl<<std::endl;
//     std::cout<<"Y:"<<std::endl;
//     std::cout<<blockY.data.mean({2,3},false,at::kDouble)<<std::endl<<std::endl;


//     double transformGain = totalTransformGain({9,9,64,64});
//     cout<<transformGain<<endl;
//     blockY.sgtTransform(transformGain,{-3.5,3.5});
//     blockCb.sgtTransform(transformGain,{-3.5,3.5});
//     blockCr.sgtTransform(transformGain,{-3.5,3.5});
//     std::cout<<endl;
//     std::cout<<blockY.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

//     std::cout<<"SGT Mean = "<<blockY.data.to(at::kDouble).mean({2,3},false,at::kDouble)<<std::endl<<std::endl;
//     blockY.data = (blockY.data/(1<<15)).round()*(1<<15);
//     blockCb.data = (blockCb.data/(1<<15)).round()*(1<<15);
//     blockCr.data = (blockCr.data/(1<<15)).round()*(1<<15);
//     cout<<blockY.data.max().item()<<endl;
//     std::cout<<"SGT Mean = "<<blockY.data.to(at::kDouble).mean({2,3},false,at::kDouble)<<std::endl;

//     std::cout<<"RhoS: "<<blockY.ssi.getRhoS()<<" Code:"<<blockY.ssi.getRhoSCode()<<std::endl;
//     std::cout<<"RhoT: "<<blockY.ssi.getRhoT()<<" Code:"<<blockY.ssi.getRhoTCode()<<std::endl;
//     std::cout<<"RhoU: "<<blockY.ssi.getRhoU()<<" Code:"<<blockY.ssi.getRhoUCode()<<std::endl;
//     std::cout<<"RhoV: "<<blockY.ssi.getRhoV()<<" Code:"<<blockY.ssi.getRhoVCode()<<std::endl;
//     std::cout<<"Disparity: "<<blockY.ssi.getDisparity()<<" Code:"<<blockY.ssi.getDCode()<<std::endl;
//     blockY.isgtTransform(transformGain,blockY.ssi);
//     blockCb.isgtTransform(transformGain,blockCb.ssi);
//     blockCr.isgtTransform(transformGain,blockCr.ssi);
//     cout<<blockY.ssi.getDisparity()<<" "<<blockY.data.max().item()<<endl;

//     blockY.data = blockY.data + 512;
//     blockCb.data = blockCb.data + 512;
//     blockCr.data = blockCr.data + 512;
//     blockY.clip(0,1024);
//     blockCb.clip(0,1024);
//     blockCr.clip(0,1024);
//     YCoCg2RGB_(blockR, blockG, blockB, blockY, blockCb, blockCr, inputLF.mPGMScale);
//     LightField outputLF({9,9,64,64,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');
// }
// TEST(EncoderTests,OptimizePartition){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/example_l0.comp";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 1;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     Block4D_ R = inputLF.ReadBlock4DfromLightField_({9,9,64,64},{0,0,34,34},0);

    
//     std::cout<<"block var = "<<R.data.to(at::kDouble).var()<<endl;
//     //std::cout<<block.data[0][0]<<std::endl;
//     cout<<"Block is Ready!"<<endl;
//     FILE *outputFileNamePointer;
//     if((outputFileNamePointer = fopen(outputDirectory.c_str(), "wb")) == NULL) {
//         printf("Error: input file %s not found\n", outputDirectory.c_str());
//         exit(0);
//     }
    
//     Hierarchical4DEncoder hdt;
//     hdt.StartEncoder(outputFileNamePointer);

//     TransformPartition tp;
//     tp.mlength_t_min = 9;
//     tp.mlength_s_min = 9;
//     tp.mlength_v_min = 64;
//     tp.mlength_u_min = 64;
//     hdt.RestartProbabilisticModel();
//     tp.RDoptimizeTransform_(R, hdt,{-3,3},1, lambda);
//     tp.EncodePartition_(hdt, lambda);
//     hdt.DoneEncoding();
//     fclose(outputFileNamePointer);
//     cout<<"Encoded Max: "<<R.data.max().item()<<endl;

//     // PartitionDecoder pd;
//     // Hierarchical4DDecoder hdd;

//     // //hdd.mSuperiorBitPlane = hdt.mSuperiorBitPlane;
//     // hdd.mSuperiorBitPlane = 30;
//     // FILE *inputFileNamePointer;
//     // if((inputFileNamePointer = fopen(outputDirectory.c_str(), "rb")) == NULL) {
//     //     printf("Error: input file %s not found\n", outputDirectory.c_str());
//     //     exit(0);
//     // }

//     // hdd.StartDecoder(inputFileNamePointer);
//     // pd.mPartitionData = Block4D_({9,9,64,64});
//     // hdd.RestartProbabilisticModel();
//     // pd.DecodePartition(hdd);
//     // cout<<"Decoded Max: "<<pd.mPartitionData.data.max().item()<<endl;


//     // std::cout<<mse(pd.mPartitionData,R)<<std::endl;   
//     // fclose(inputFileNamePointer);


// }
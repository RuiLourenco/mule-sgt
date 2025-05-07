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
#include <filesystem>
#include <sys/stat.h>
#include <DebugTools/CodingPartitionInfo.h>
#include <DebugTools/CodingUnitInfo.h>



using namespace std;
using namespace filesystem;

long GetFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

double mse(const at::Tensor& a, const at::Tensor& b) {
    //cout<<"MSE CALCULATION!"<<endl;
    auto mse =(a - b).pow(2).to(at::kDouble).mean(); 
    //cout<<"MSE CALCULATION Complete!"<<endl;

    return mse.item<double>();
}
double totalTransformGain_(std::array<int64_t,4> length, std::array<int64_t,4> maxLength){
    
    double transformGain = 1;
    for(int i = 0; i < 4; i++){
        transformGain*=length[i]/sqrt(length[i]);
        transformGain  *= sqrt(maxLength[i]/length[i]);
    }
    return transformGain*1;

}
// void RGB2YCoCg(Block4D &Y, Block4D &Co, Block4D &Cg, Block4D const &R, Block4D const &G, Block4D const &B, int Scale) {
    
//     for(int n = 0; n < R.mlength_t*R.mlength_s*R.mlength_v*R.mlength_u; n++) {
//         int t;
//         Co.mPixelData[n] = R.mPixelData[n] - B.mPixelData[n];
//         t = B.mPixelData[n] + (Co.mPixelData[n]>>1);
//         Cg.mPixelData[n] = G.mPixelData[n] - t;
//         Y.mPixelData[n] = t + (Cg.mPixelData[n]>>1);
//         Co.mPixelData[n] += (Scale + 1)/2;
//         Cg.mPixelData[n] += (Scale + 1)/2;
//     }
        
// }
// void RGB2YCoCg_(Block4D_ &Y, Block4D_ &Co, Block4D_ &Cg, Block4D_ const &R, Block4D_ const &G, Block4D_ const &B, int Scale) {
//     Co = R - B;
//     auto temp = B + Co.data.bitwise_right_shift(1);
//     Cg = G - temp;
//     Y = temp + Cg.data.bitwise_right_shift(1);
//     Co.data+= (Scale + 1)/2;
//     Cg.data+= (Scale + 1)/2;
//     Y.validPositions = R.validPositions;
//     Co.validPositions = R.validPositions;
//     Cg.validPositions = R.validPositions;        
// }
// void YCoCg2RGB_(Block4D_ &R, Block4D_ &G, Block4D_ &B, Block4D_ const &Y, Block4D_ const &Co, Block4D_ const &Cg, int Scale) {
//     auto CoTemp = Co.data - (Scale+1)/2;
//     auto CgTemp = Cg.data - (Scale+1)/2;
//     auto t = Y - (CgTemp.bitwise_right_shift(1));
//     G = CgTemp + t;
//     B.data = t - (CoTemp.bitwise_right_shift(1));
//     R.data = B.data + CoTemp;          
// }
unsigned long int BigEndianUnsignedIntegerRead_(int precision, FILE *inputFilePointer) {

    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    
    unsigned char input_byte;
    unsigned long int value = 0;
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        int a = fread(&input_byte, 1, 1, inputFilePointer);
        value = (value << 8);
        value += input_byte;
    }
    return (value);
}

long int BigEndianSignedIntegerRead_(int precision, FILE *inputFilePointer) {

    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    
    unsigned char input_byte;
    unsigned long int value = 0;
    int sign;
    int a = fread(&input_byte, 1, 1, inputFilePointer);
    if(input_byte) {
        sign = -1;
    }else{
        sign = 1;
    }
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        int a = fread(&input_byte, 1, 1, inputFilePointer);
        value = (value << 8);
        value += input_byte;
    }
    return (value*sign);
}
void BigEndianUnsignedIntegerWrite_(unsigned long int value, int precision, FILE *outputFilePointer) {

    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    
    unsigned char output_byte;
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        output_byte = (value >> (8*byte_index))&255;
        fwrite(&output_byte, 1, 1, outputFilePointer);
    }
    
}
void BigEndianSignedIntegerWrite_(long int value, int precision, FILE *outputFilePointer) {
    if(precision > sizeof(long int)) {
        printf("Unsuported %d bytes integer precision\n", precision);
        exit(0);
    }
    unsigned char sign = (value < 0) ? 1 : 0;
    fwrite(&sign,1,1,outputFilePointer);
    unsigned int magnitude = abs(value);
    unsigned char output_byte;
    for(int byte_index = precision-1; byte_index >= 0; byte_index--) {
        output_byte = (magnitude >> (8*byte_index))&255;
        fwrite(&output_byte, 1, 1, outputFilePointer);
    }
    
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
// TEST(EncoderTests,BackwallDetail){

//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/BackwallDetail/";
//     create_directory(outputDirectory);

//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 32;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);

//     LightField outputLF({9,9,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');
    
    
// }

// TEST(EncoderTests,LeastSquares){
//     at::Tensor P = torch::tensor({{4,1},{1,2}},at::kDouble);
//     at::Tensor q = torch::tensor({1,1},at::kDouble);
//     SgtSideInfo ssi;
//     ssi.QPOptimization(P,q);
// }

// TEST(EncoderTests,SidewallDetail){

//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/SidewallDetail/";
//     create_directory(outputDirectory);

//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 32;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);

//     LightField outputLF({9,9,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');
    
    
// }
// TEST(EncoderTests,CabinetDetail){

//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/CabinetDetail/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     std::array<int64_t,4> position = {0,0,320,64};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 32;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);

//     LightField outputLF({9,9,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');
    
    
// }

// TEST(EncoderTests,GreekRightWallDummy){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/RightWall/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Greek Between Statues
//     std::array<int64_t,4> position = {0,0,64,448};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 64;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);

//     LightField outputLF({9,9,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }

// TEST(EncoderTests,FountainBug){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/Fountain_Vincent2/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/FountainBug/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(inputDirectory,pattern);
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Fountain Bug
//     std::array<int64_t,4> position = {0,0,0,512};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 32;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({15,15,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({15,15,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({15,15,length,length},position,2);

//     LightField outputLF({15,15,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }

// TEST(EncoderTests,GreekBug){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/GreekBug/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(inputDirectory,pattern);
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Fountain Bug
//     std::array<int64_t,4> position = {0,0,0,128};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 64;
//     int viewLength = 9;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,2);

//     LightField outputLF({viewLength,viewLength,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }

// TEST(EncoderTests,QuarterGreek){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/i8mylife/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(inputDirectory,pattern);
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Fountain Bug
//     std::array<int64_t,4> position = {0,0,3*64+32,2*64};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 32;
//     int viewLength = 9;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,2);

//     LightField outputLF({viewLength,viewLength,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }

// TEST(EncoderTests,QuarterGreek){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/QuarterGreek/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(inputDirectory,pattern);
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Fountain Bug
//     std::array<int64_t,4> position = {0,0,0,0};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 256;
//     int viewLength = 9;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({viewLength,viewLength,length,length},position,2);

//     LightField outputLF({viewLength,viewLength,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }

// TEST(DebugInfoTests,STPrinting){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/structureTensor/Greek/";
//     string inputInfo = inputDirectory + "greek_0.75_info.json";
//     std::vector<CodingPartitionInfo> codingPartitionInfos = CodingPartitionInfo::fromJsonFile(inputInfo);
//     string inputDirectory1 = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/TestingSpeed/Greek/";
//     string inputInfo1 = inputDirectory1 + "greek_0.75_info.json";
//     std::vector<CodingPartitionInfo> codingPartitionInfos1 = CodingPartitionInfo::fromJsonFile(inputInfo1);
    
//     std::array<int64_t,4> position = {0,0,0,448};
//     CodingPartitionInfo cpi = CodingPartitionInfo::findPartitionInfoByPosition(codingPartitionInfos,position);
//     CodingPartitionInfo cpi1 = CodingPartitionInfo::findPartitionInfoByPosition(codingPartitionInfos1,position);
//     //cpi.generatePythonScriptsForPartition(inputDirectory);
//     auto partitions = cpi.getCodingUnitInfos();
//     auto partitions1 = cpi1.getCodingUnitInfos();

//     std::sort(partitions.begin(), partitions.end(), [](const CodingUnitInfo& a, const CodingUnitInfo& b) {
//         return a.getLightFieldPosition() < b.getLightFieldPosition();
//     });

//     std::sort(partitions1.begin(), partitions1.end(), [](const CodingUnitInfo& a, const CodingUnitInfo& b) {
//         return a.getLightFieldPosition() < b.getLightFieldPosition();
//     });

//     for (size_t i = 0; i < partitions.size(); ++i) {
//         const auto& partition = partitions[i];
//         const auto& partition1 = partitions1[i];
//         std::cout << partition.getLightFieldPosition()[2] << " " << partition.getLightFieldPosition()[3]
//                   << ": " << partition.getBestStructureTensorAngle() << " (" << partition.getBestStructureTensorCost() << " )"
//                   << partition1.getLightFieldPosition()[2] << " " << partition1.getLightFieldPosition()[3]
//                   << ": " << partition1.getBestGridSearchAngle() << " (" << partition1.getBestGridSearchCost() << " )" << std::endl;
//     }
// }

TEST(DebugInfoTests,CostGraphPrinting){
    string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/TestingSpeed/Greek/";
    string inputInfo = inputDirectory + "greek_0.75_info.json";
    std::vector<CodingPartitionInfo> codingPartitionInfos = CodingPartitionInfo::fromJsonFile(inputInfo);
    std::array<int64_t,4> position = {0,0,7*64,0*64};
    CodingPartitionInfo cpi = CodingPartitionInfo::findPartitionInfoByPosition(codingPartitionInfos,position);
    cpi.generatePythonScriptsForPartition(inputDirectory);
}
TEST(DebugInfoTests,LoadAndPrint){
    
    string inputDirectory1 = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/TestingSpeed/Greek/";
    string inputDirectory ="/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/structureTensor/Greek/";
    string inputInfo = inputDirectory + "greek_0.1_info.json";
    string inputInfo1 = inputDirectory1 + "greek_0.1_info.json";
    std::array<int64_t,2> size = {512,512};
    cout<<inputInfo<<endl;
    //string inputDirectory1 = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/DebugData/GreekNew/info.json";
    std::vector<CodingPartitionInfo> codingPartitionInfos = CodingPartitionInfo::fromJsonFile(inputInfo);
    std::vector<CodingPartitionInfo> codingPartitionInfos1 = CodingPartitionInfo::fromJsonFile(inputInfo1);
    at::Tensor stH = CodingPartitionInfo::getStructureTensorHorizontal(codingPartitionInfos,size);
    at::Tensor stV = CodingPartitionInfo::getStructureTensorVertical(codingPartitionInfos,size);
    at::Tensor stA = CodingPartitionInfo::getStructureTensorAverage(codingPartitionInfos,size);
    at::Tensor st = CodingPartitionInfo::getBestStructureTensorAngle(codingPartitionInfos,size);
    // at::Tensor ldH = CodingPartitionInfo::getLogdetHorizontal(codingPartitionInfos,{512,512});
    // at::Tensor ldV = CodingPartitionInfo::getLogdetVertical(codingPartitionInfos,{512,512});
    // at::Tensor ldA = CodingPartitionInfo::getLogdetAverage(codingPartitionInfos,{512,512});
    // at::Tensor ld = CodingPartitionInfo::getBestLogdetAngle(codingPartitionInfos,{512,512});
    at::Tensor gs = CodingPartitionInfo::getBestGridSearchAngle(codingPartitionInfos1,size);
    at::Tensor gsCost = CodingPartitionInfo::getBestGridSearchCost(codingPartitionInfos1,size);
    at::Tensor chosenAngle = CodingPartitionInfo::getChosenAngle(codingPartitionInfos,size);
    // at::Tensor rate = CodingPartitionInfo::getRate(codingPartitionInfos,{512,512});
    // at::Tensor distortion = CodingPartitionInfo::getPSNR(codingPartitionInfos,{512,512});
    //at::Tensor heuristic = CodingPartitionInfo::getAngleHeuristicUsed(codingPartitionInfos,{512,512});
    at::Tensor stCost = CodingPartitionInfo::getBestStructureTensorCost(codingPartitionInfos,size);
    // at::Tensor angleErrorST = CodingPartitionInfo::getStructureTensorError(codingPartitionInfos,{512,512});
    //at::Tensor costDiffST = CodingPartitionInfo::getStructureTensorCostDiff(codingPartitionInfos,{512,512});
    // at::Tensor angleErrorLd = CodingPartitionInfo::getLogdetError(codingPartitionInfos,{512,512});
    // at::Tensor costDiffLd = CodingPartitionInfo::getLogdetCostDiff(codingPartitionInfos,{512,512});
    
    //at::Tensor stConfidence = CodingPartitionInfo::getStructureTensorConfidence(codingPartitionInfos,{512,512});
    //at::Tensor angleError = (gs - st).abs();
    // angleErrorST = angleErrorST.clamp(0, angleErrorST.quantile(0.9).item<double>());
    // angleErrorLd = angleErrorLd.clamp(0, angleErrorLd.quantile(0.9).item<double>());
    at::Tensor costDiff = (stCost - gsCost)/gsCost;
    // std::cout<<"stCost: "<<stCost.min().item()<<" "<<stCost.max().item()<<std::endl;
    std::cout<<"cost diff: "<<costDiff.min().item()<<" "<<costDiff.max().item()<<std::endl;
    //
    // std::cout<<"gs: "<<gs.min().item()<<" "<<gs.max().item()<<std::endl;
    // std::cout<<"angleErrorST: "<<angleErrorST.min().item()<<" "<<angleErrorST.max().item()<<std::endl;
    // std::cout<<"angleErrorLd: "<<angleErrorLd.min().item()<<" "<<angleErrorLd.max().item()<<std::endl;
    double gsMin = -74;
    double gsMax = 74;
    // double gsMin = gs.min().item<double>();
    // double gsMax = gs.max().item<double>();
    //std::cout<<"stConfidence: "<<stConfidence.min().item()<<" "<<stConfidence.max().item()<<std::endl;
    // std::cout<<"angle chosen: "<< stV[64*2][64*3].item()<<" "<< stH[64*2][64*3].item()<<" "<<gs[64*2][64*3].item()<<std::endl;
    // std::cout<<"costDiff: "<< costDiff[64*2][64*3].item()<<std::endl;
    write_tensor(stH,inputDirectory + "sth.png",{gsMin,gsMax});
    write_tensor(stV,inputDirectory + "stv.png",{gsMin,gsMax});
    write_tensor(stA,inputDirectory + "sta.png",{gsMin,gsMax});
    write_tensor(st,inputDirectory + "st.png",{gsMin,gsMax});
    write_tensor(gsCost,inputDirectory + "gsCost.png");
    write_tensor(stCost,inputDirectory + "stCost.png");
    //cout<<"written ST" <<std::endl;
    //write_tensor(error,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/DebugData/GreekST-3/error.png",{0,15.0});
    //write_tensor(stConfidence,inputDirectory + "Confidence1.png",{0,15});
    write_tensor(costDiff,inputDirectory + "CostDiffST.png",{0,1});

    // write_tensor(ldH,inputDirectory + "ldh.png",{gsMin,gsMax});
    // write_tensor(ldV,inputDirectory + "ldv.png",{gsMin,gsMax});
    // write_tensor(ldA,inputDirectory + "lda.png",{gsMin,gsMax});
    // write_tensor(ld,inputDirectory + "ld.png",{gsMin,gsMax});
    // write_tensor(rate,inputDirectory + "rate.png",{});
    // write_tensor(distortion,inputDirectory + "distortion.png");
    // write_tensor(heuristic,inputDirectory + "heuristic.png");
    // write_tensor(angleErrorST,inputDirectory + "AngleErrorST.png",{0,5});
    // write_tensor(costDiffST,inputDirectory + "CostDiffST.png",{0,0.5});
    // write_tensor(angleErrorLd,inputDirectory + "AngleErrorLd.png",{0,5});
    // write_tensor(costDiffLd,inputDirectory + "CostDiffLd.png",{0,0.5});
    write_tensor(gs,inputDirectory + "gs.png",{gsMin,gsMax});
    write_tensor(chosenAngle,inputDirectory + "chosenAngle.png",{gsMin,gsMax});
    
}
// TEST(EncoderTests,GreekSTPoor){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/GreekSTPoor/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(inputDirectory,pattern);
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Fountain Bug
//     std::array<int64_t,4> position = {0,0,64,128};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 64;
//     int view = 9;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({view,view,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({view,view,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({view,view,length,length},position,2);

//     LightField outputLF({view,view,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   

// }
// TEST(EncoderTests,FountainBug){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/Fountain_Vincent2/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/FountainBug/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(inputDirectory,pattern);
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Fountain Bug
//     std::array<int64_t,4> position = {0,0,0,512};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 32;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({15,15,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({15,15,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({15,15,length,length},position,2);

//     LightField outputLF({15,15,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }


// TEST(EncoderTests,GreekHairDummy){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/GreekHair/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Greek Between Statues
//     std::array<int64_t,4> position = {0,0,0,320};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 64;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);

//     LightField outputLF({9,9,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }

// TEST(EncoderTests,GreekBeetStatueDummy){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     //string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/sideboard/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/GreekBeetStat/";
//     create_directory(outputDirectory);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     //std::array<int64_t,4> position = {0,0,64,448};
    
//     //Sideboard Dummy BACKWALL
//     //std::array<int64_t,4> position = {0,0,114,90};
//     //Sideboard Dummy SIDEWALL
//     //std::array<int64_t,4> position = {0,0,114,480};
//     //Sideboard Dummy Cabinet
//     //std::array<int64_t,4> position = {0,0,320,64};

//     //Greek Between Statues
//     std::array<int64_t,4> position = {0,0,352,225};

//     //std::array<int64_t,4> position = {0,0,64,64};

    
//     //std::array<int64_t,4> position = {0,0,160,288};
    
//     //std::array<int64_t,4> position = {0,0,256,288};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     //std::array<int64_t,4> position = {0,0,64,128};
//     //std::array<int64_t,4> position = {0,0,384,256};   
//     //std::array<int64_t,4> position = {0,0,256,320};
//     int length = 32;
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);

//     LightField outputLF({9,9,length,length,3});
//     outputLF.WriteBlock4DtoLightField_(blockR,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockG,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockB,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');   
// }

void saveTensorAsMatlabScript2(at::Tensor tensor,std::string name){
    std::ofstream stuff;
    stuff.open("/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/modelCovariances/"+name+".m");
    stuff<<name+"_cpp = [";
    for(int n = 0; n < tensor.size(0); n++){
        if(n != 0) stuff<<";"<<std::endl;
        for(int m = 0; m < tensor.size(1); m++){
            if (m != 0) stuff<<",";
            stuff<<tensor[n][m].item();
        }
    }
    stuff<<"];"<<std::endl;
}

// TEST(EncoderTests,angleInfluence){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     int length = 64;
//     std::array<int64_t,4> position = {0,0,0,0};
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     Block4D_ temp = blockY;
//     SgtSideInfo ssi(0,0,{-3.5,3.5});
//     for(double alpha = 47.6; alpha <= 48.01; alpha += 0.2){
//         ssi.setAngleH(alpha);
//         ssi.setAngleV(alpha);
//         cout<<ssi.getDisparityH()<<", ";
//         blockY.ssi = ssi;
//         blockY.sgtTransform(1);
//         int alphaInt = (int)(alpha*10);
//         saveTensorAsMatlabScript2(blockY.data[0][0],"alpha_"+to_string(alphaInt)+"_d");
//         blockY = temp;
//     }
// }

// TEST(EncoderTests,rhoInfluence){
//      string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     int length = 64;
//     std::array<int64_t,4> position = {0,0,0,0};
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,length,length},position,2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     Block4D_ temp = blockY;
//     SgtSideInfo ssi(47.8,47.8,{-3.5,3.5});

//     std::array<double,3> rhoAngleVector = {0.999};
//     std::array<int,3> rhoAngleText = {999};
//     std::array<double,3> rhoSptVector = {0.3,0.4,0.5};
//     std::array<int,3> rhoSptText = {3,4,5};
//     for(int rhoAngIndex = 0; rhoAngIndex < rhoAngleVector.size(); rhoAngIndex += 1){
//         ssi.setRhoS(rhoAngleVector[rhoAngIndex]);
//         ssi.setRhoT(rhoAngleVector[rhoAngIndex]);
//         for(int rhoSptIndex = 0; rhoSptIndex < rhoSptVector.size(); rhoSptIndex += 1){
//             ssi.setRhoU(rhoSptVector[rhoSptIndex]);
//             ssi.setRhoV(rhoSptVector[rhoSptIndex]);
//             cout<<ssi.getDisparityH()<<", ";
//             blockY.ssi = ssi;
//             blockY.sgtTransform(1);
//             saveTensorAsMatlabScript2(blockY.data[0][0],"spt_"+to_string(rhoSptText[rhoSptIndex])+"_ang_"+to_string(rhoAngleText[rhoAngIndex])+"_r");
//             blockY = temp;
//         }
//     }
// }

// TEST(EncoderTests, modelCovFun){
//     Block4D_ block({9,9,32,32});

//     SgtSideInfo ssi(20,20,{-3.5,3.5});
//     std::vector<double> rhoAngleVector = {0.999};
//     std::vector<int> rhoAngleText = {999};
//     std::vector<double> rhoSptVector = {0.99402, 0.97972};
//     std::vector<int> rhoSptText = {99402, 97972};
//     for(int rhoAngIndex = 0; rhoAngIndex < rhoAngleVector.size(); rhoAngIndex += 1){
//         ssi.setRhoS(rhoAngleVector[rhoAngIndex]);
//         ssi.setRhoT(rhoAngleVector[rhoAngIndex]);
//         for(int rhoSptIndex = 0; rhoSptIndex < rhoSptVector.size(); rhoSptIndex += 1){
//             ssi.setRhoU(rhoSptVector[rhoSptIndex]);
//             ssi.setRhoV(rhoSptVector[rhoSptIndex]);
//             //cout<<ssi.getDisparityH()<<", ";
//             //blockY.ssi = ssi;
//             //blockY.sgtTransform(1);
//             at::Tensor modelCovFun = block.calcModelCovFun(ssi,true);
//             saveTensorAsMatlabScript2(modelCovFun,"covFun_spt_"+to_string(rhoSptText[rhoSptIndex])+"_ang_"+to_string(rhoAngleText[rhoAngIndex])+"_r");
//         }
//     }
// }
// TEST(EncoderTests,ordering){
//     std::vector<std::array<int64_t,4>> treeOrderedPositions = Block4D_::treeOrderedCoefficientPositions({1,1,8,8});
//     std::cout<<"size: "<<treeOrderedPositions.size()<<std::endl;
//     at::Tensor orderedRange = at::arange(0,8*8);
//     at::Tensor final = at::zeros({1,1,8,8});
//     for(int i = 0; i < treeOrderedPositions.size(); i++) {
//         final.index({treeOrderedPositions[i][0],treeOrderedPositions[i][1],treeOrderedPositions[i][2],treeOrderedPositions[i][3]}) = orderedRange[i];
//     }
//     std::cout<<final[0][0]<<std::endl;
// }
// TEST(EncoderTests,zeroLightField){

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 0;
//     LightField inputLF(9,9,512);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     //std::array<int64_t,4> position = {0,0,192,0};
//     //std::array<int64_t,4> position = {0,0,0,0};
//     std::array<int64_t,4> position = {0,0,64,128};
//     Block4D_ blockR = inputLF.ReadBlock4DfromLightField_({9,9,64+8,64+8},position,0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,64+8,64+8},position,1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,64+8,64+8},position,2);
//     Block4D_ blockRNew = Block4D_({9,9,64,64});
//     Block4D_ blockGNew = Block4D_({9,9,64,64});
//     Block4D_ blockBNew = Block4D_({9,9,64,64});
    
//     std::cout<<blockRNew.data.sizes()<<std::endl;
//     std::cout<<blockGNew.data.sizes()<<std::endl;
//     std::cout<<blockBNew.data.sizes()<<std::endl;
//     std::cout<<blockR.data.sizes()<<std::endl;
//     std::cout<<blockG.data.sizes()<<std::endl;
//     std::cout<<blockB.data.sizes()<<std::endl;
//     double disp = 0;
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-"+std::to_string(disp)+"-Disp-ORIGINAL/";
//     std::filesystem::create_directory(outputDirectory);
//     for (int k = 0; k < 9; ++k){
//         for (int l = 0; l<9; ++l){
//             //if(k==4 && l==4) continue;
//             blockRNew.data.index({(int)round((l)),(int)round((k)),at::indexing::Slice(),at::indexing::Slice()}) = blockR.data.index({4,4,at::indexing::Slice(0,0+64),at::indexing::Slice(0,0+64)});
//             cout<<"("<<8-l<<","<<8-k<<")"<< blockRNew.data[l][k][l][k].item()<<endl;
//             blockGNew.data.index({(int)round((l)),(int)round((k)),at::indexing::Slice(),at::indexing::Slice()}) = blockG.data.index({4,4,at::indexing::Slice(0,0+64),at::indexing::Slice(0,0+64)});
//             cout<<"2"<<endl;
//             blockBNew.data.index({(int)round((l)),(int)round((k)),at::indexing::Slice(),at::indexing::Slice()}) = blockB.data.index({4,4,at::indexing::Slice(0,0+64),at::indexing::Slice(0,0+64)});
//         }
//     }
//     LightField outputLF({9,9,64,64,3});
//     outputLF.WriteBlock4DtoLightField_(blockRNew,{0,0,0,0,0});
//     outputLF.WriteBlock4DtoLightField_(blockGNew,{0,0,0,0,1});
//     outputLF.WriteBlock4DtoLightField_(blockBNew,{0,0,0,0,2});
//     outputLF.OpenLightFieldPPM_(outputDirectory,"",'w');
    
// }
// int fib(int64_t n){
//     if(n == 0) return 0;
//     if(n == 1) return 1;
//     return fib(n-1) + fib(n-2);
// }

// at::Tensor getZigZagIndexes2(std::array<int64_t,2> size){
//     int64_t maxSize = max(size[0],size[1]);
//     int64_t minSize = min(size[0],size[1]);
//     at::Tensor cumm = at::zeros(size,at::kLong);
//     int bias = 0;
//     for (int64_t i = -size[0] + 1; i<size[1]; ++i){
//         int diagSize = min(min(abs(i-size[1]),i+size[0]),minSize);
//         cout<<"Diag Size: "<<diagSize<< endl;
//         at::Tensor diag = at::zeros(diagSize,at::kLong);
//         for (int64_t j = 0; j<diagSize; ++j){
//             int64_t m = min(i+size[0] -1,size[0] -1) - j;
//             int64_t n = j+max((int64_t)0,i);
//             if(i%2 == 0){
//                 cumm[m][n] = bias + (diagSize - 1 - j);
//             }else{
//                 cumm[m][n] = bias + j;
//             }
//         }
//         bias = bias + diagSize;
//     }
//     cout<<"CUMM: "<<endl<<cumm.index({at::indexing::Slice(),at::indexing::Slice(0,7)})<<endl;
//     return cumm.flatten();
// }
// at::Tensor getZigZagIndexes1(std::array<int64_t,2> size){
//     int64_t size2 = size[0]*size[1];

//     at::Tensor indMatrix = at::range(0,size[0]*size[1]-1,1).reshape({size[0],size[1]});

//     torch::TensorOptions options = torch::TensorOptions();
//     at::Tensor zigZag = at::empty({0},options.dtype(at::kLong));

//     for(int i = -(size[0] - 1); i < size[1] ; ++i){

//         at::Tensor diag = at::diag(indMatrix.fliplr(),i).to(at::kLong);


//         if(i%2 == 0){
//             diag = diag.flip(0);
//         }
//         zigZag =at::cat({zigZag,diag},0);
//     }

//     zigZag = zigZag.flip(0);
//     //cout<<zigZag<<endl;
//     return zigZag;
// }
// at::Tensor zigZagTransformMatrix(at::Tensor transform,std::array<int64_t,2> size){
//     at::Tensor zigZagIndices = getZigZagIndexes2(size);
//     cout<<"oiii"<<endl;
//     return transform.index({at::indexing::Slice(),zigZagIndices});
// }

// at::Tensor diagonalOrder4DSampling(at::Tensor coefficients, std::array<int64_t,4> size){
//     int maxSum = 0;
//     for(int i = 0; i < 4; i++){
//         maxSum += size[i];
//     }
//     //std::cout<<"Max Sum = "<<maxSum<<std::endl;
//     at::Tensor tensor4D = at::zeros({size[0],size[1],size[2],size[3]});
//     int i = 0;
//     for(int sum = 0;sum<maxSum;sum++){
//         for(int n = 0;n<size[0];n++){
//             for(int m = 0;m<size[1];m++){
//                 for(int k = 0;k<size[3];k++){
//                     for(int l = 0;l<size[2];l++){
//                         int currentSum = m+n+k+l;
//                         if(sum == currentSum){
//                             cout<<sum<<" "<<currentSum<<endl;
//                             tensor4D[m][n][k][l] = coefficients[i];
//                             i++;
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     if( i != coefficients.size(0)){
//         cout<<"ERROR:"<< i <<"!="<< coefficients.size(0)<<endl;
//     }else{
//         cout<<"Success i = "<<i<<endl;
//     }
//     return tensor4D;
// }

// TEST(EncoderTests,TriangleTest){
//     Block4D_ isux;
//     at::Tensor flatBlock = at::range(15,0,-1).reshape({4,4});
//         std::cout <<flatBlock;

//     at::Tensor indexes = isux.getZigZagIndexes({4,4});
//     at::Tensor flatBlock2 = flatBlock.index({indexes});
//     std::cout <<flatBlock2;
//     //std::cout <<flatBlock;

// }
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

// TEST(EncoderTests,BasisOrderShiftUnshift){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/decode-shift.comp";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     double lambda = 25;
//     LightField inputLF(9,9,32);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,32,32},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,32,32},0);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,32,32},0);

//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);

//     blockY.data = blockY.data - 512;
//     blockCb.data = blockCb.data - 512;
//     blockCr.data = blockCr.data - 512;
//     at::Tensor flatBlock = blockY.getFlatBlock();
//     blockY.ssi = SgtSideInfo(blockY,{-3.5,3.5});

//     at::Tensor modelCovMatH = blockY.calcModelCovMatrix(blockY.ssi,true);
//     at::Tensor modelCovMatV = blockY.calcModelCovMatrix(blockY.ssi,false);
//     at::Tensor eigValsH,eigValsV;
//     at::Tensor sgtMatrixH = blockY.getSgtTransformMatrix(modelCovMatH,true,eigValsH);
//     at::Tensor sgtMatrixV = blockY.getSgtTransformMatrix(modelCovMatV,false,eigValsH);

        
// }

void decodeHeader(FILE *inputFileNamePointer,Hierarchical4DDecoder& hdt,std::array<int64_t,4>& lfSize, std::array<int64_t,4>& maxPartitionSize,std::array<double,2>& disparityRange, int& PGMScale){

    hdt.mSuperiorBitPlane = BigEndianUnsignedIntegerRead_(2, inputFileNamePointer);
    std::cout<<"SuperiorBitPlane: "<<hdt.mSuperiorBitPlane<<std::endl;

    //reads the maximum Partition sizes
    for (int n = 0; n < 4; n++) {
        maxPartitionSize[n] = BigEndianUnsignedIntegerRead_(2, inputFileNamePointer);
    }
    std::cout<<"decodrrMaxPartitionSize: "<<maxPartitionSize[0]<<" "<<maxPartitionSize[1]<<" "<<maxPartitionSize[2]<<" "<<maxPartitionSize[3]<<std::endl;
    //reads the LightField Size
    for(int n = 0; n < 4; n ++ ){
        lfSize[n] = BigEndianUnsignedIntegerRead_(2, inputFileNamePointer);
    }
    std::cout<<"LightField Size: "<<lfSize[0]<<" "<<lfSize[1]<<" "<<lfSize[2]<<" "<<lfSize[3]<<std::endl;
    for(int n = 0; n < 2; n++) {
        disparityRange[n] =(double) BigEndianSignedIntegerRead_( 2, inputFileNamePointer)/100.0;
    }
    std::cout<<"DisparityRange: "<<disparityRange[0]<<" "<<disparityRange[1]<<std::endl;
    PGMScale = BigEndianUnsignedIntegerRead_( 2, inputFileNamePointer);
    std::cout<<"PGMScale: "<<PGMScale<<std::endl;
}

void encodeHeader(FILE *outputFileNamePointer, Hierarchical4DEncoder& hdt , std::array<int64_t,4> lfSize, std::array<int64_t,4> maxPartitionSize, std::array<double,2> disparityRange, int mPGMScale){
    BigEndianUnsignedIntegerWrite_(hdt.mSuperiorBitPlane, 2, outputFileNamePointer);
    for(int n = 0; n < 4; n++) {
        BigEndianUnsignedIntegerWrite_(maxPartitionSize[n], 2, outputFileNamePointer);
    }
    for(int n = 0; n < 4; n++) {
        BigEndianUnsignedIntegerWrite_(lfSize[n], 2, outputFileNamePointer);
    }
       //Writes an Integer Encoded Disparity Range of the LF
    for(int n = 0; n < 2; n++) {
        BigEndianSignedIntegerWrite_((int)(disparityRange[n]*100), 2, outputFileNamePointer);
    }
    BigEndianUnsignedIntegerWrite_(mPGMScale, 2, outputFileNamePointer);
}

void encodePartition(Hierarchical4DEncoder& entropyCoder, double lambda,Block4D_ inputBlock, std::array<double,2> disparityRange, double angleH, double angleV, double& J0,double& conditionNumberH, double& conditionNumberV){
    std::array<int64_t,4> length = {inputBlock.data.size(0),inputBlock.data.size(1),inputBlock.data.size(2),inputBlock.data.size(3)};
    double scaledLambda = length[0]*length[1]*length[2]*length[3]*lambda;
    inputBlock.data = inputBlock.data.contiguous();
    
    entropyCoder.LoadOptimizerState();
    //cout<<"Optimizer State Loaded!"<<endl;


    double currGain = totalTransformGain_(length,length);
    //cout<<"currGain: "<<currGain<<endl;
    inputBlock.ssi = SgtSideInfo(angleV,angleH,disparityRange);
    //inputBlock.ssi.print();
    inputBlock.sgtTransform(currGain);
    conditionNumberH = (inputBlock.eigenValuesH[0]/inputBlock.eigenValuesH[-1]).item<double>();
    conditionNumberV = (inputBlock.eigenValuesV[0]/inputBlock.eigenValuesV[-1]).item<double>();

    ///cout<<"Transformed!"<<endl;
    entropyCoder.mSubbandLF_ = inputBlock;
    entropyCoder.mInferiorBitPlane = entropyCoder.OptimumBitplaneFaster_(scaledLambda);

    entropyCoder.LoadOptimizerState();
    
    ProbabilityModel *currentCoderModelState;
    entropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    std::array<int64_t,4> lengthTransform = {entropyCoder.mSubbandLF_.data.size(0), entropyCoder.mSubbandLF_.data.size(1), entropyCoder.mSubbandLF_.data.size(2), entropyCoder.mSubbandLF_.data.size(3)};
    double Energy = 0;
    double rate = 0;
    double distortion = 0;
    if(entropyCoder.mSegmentationTreeCodeBuffer != NULL){
        delete [] entropyCoder.mSegmentationTreeCodeBuffer;
    }
    entropyCoder.mSegmentationTreeCodeBuffer = new char [2];
    strcpy(entropyCoder.mSegmentationTreeCodeBuffer,"");

    J0 = entropyCoder.RdOptimizeHexadecaTree_({0, 0, 0, 0}, lengthTransform, scaledLambda, entropyCoder.mSuperiorBitPlane, &entropyCoder.mSegmentationTreeCodeBuffer, Energy,rate,distortion);
    std::cout<<"J0: "<<J0<<endl;
    std::cout<<"Rate: "<<rate<<endl;
    std::cout<<"Distortion: "<<distortion<<endl;
    std::cout<<"Energy: "<<Energy<<endl;
    std::cout<<"Calculated J0 = "<<distortion + scaledLambda*rate <<endl;
    std::cout<<"Scaled Lambda = "<<scaledLambda<<endl;
    ProbabilityModel *coderModelState_0;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);

    entropyCoder.EncodeInteger(entropyCoder.mInferiorBitPlane, MINIMUM_BITPLANE_PRECISION);
    entropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
    entropyCoder.EncodeSSI_(inputBlock.ssi);
    
    entropyCoder.EncodeSubblock_(scaledLambda);

    //entropyCoder.mSubbandLF_.orderH = inputBlock.orderH;
    //entropyCoder.mSubbandLF_.orderV = inputBlock.orderV;
}
void encodeBlock(Block4D_ block, Hierarchical4DEncoder& hdt, double lambda, std::array<double,2> disparityRange,double angleH, double angleV, double& J0, double& conditionNumberH, double& conditionNumberV){
    
    hdt.RestartProbabilisticModel();
    //std::cout<<"ILY"<<std::endl;

    encodePartition(hdt, lambda,block, disparityRange, angleH, angleV, J0,conditionNumberH,conditionNumberV);
    hdt.DoneEncoding();
}



// at::Tensor decodeKLT(Hierarchical4DDecoder& hdt, double lambda, std::array<double,2> disparityRange, std::string inputImageFile, at::Tensor covH, at::Tensor covV){
     
//     hdt.RestartProbabilisticModel();

//     hdt.mSkipCount = 0;
//     hdt.mSkipMatrix = at::zeros({1,1,9*32,9*32});

//     //cout<<"Decoder Started!!"<<endl;
//     hdt.mInferiorBitPlane = hdt.DecodeInteger(MINIMUM_BITPLANE_PRECISION);
//     std::cout<<"Minimum Bit Plane: "<<hdt.mInferiorBitPlane<<std::endl;

//     int partitionFlag = hdt.DecodePartitionFlag();
//     cout<<"Partition Flag DECODED! "<<partitionFlag<<endl;
//     //SgtSideInfo ssi = hdt.DecodeSsi(disparityRange);
//     // cout<<"SSI DECODED!"<<endl;
//     //ssi.print();     
//     hdt.mSubbandLF = Block4D_({9,9,32,32});
//     cout<<"Block Created!"<<endl;
//     hdt.mSubbandLF.emptyTransform();
//     cout<<"Block Emptied!"<<endl;

//     hdt.DecodeBlock(0, 0, 0, 0, hdt.mSubbandLF.transformSize[0], hdt.mSubbandLF.transformSize[1], hdt.mSubbandLF.transformSize[2], hdt.mSubbandLF.transformSize[3], hdt.mSuperiorBitPlane); 
//     write_tensor(log2(1+hdt.mSubbandLF.data.squeeze().abs()),inputImageFile);

//     cout<<"Block DECODED!"<<endl;
//     double gain = 288;
//     cout<<"Gain: "<<gain<<endl;
//     cout<<hdt.mSubbandLF.data.sizes()<<endl;
//     hdt.mSubbandLF.ikltTransform(gain,covH,covV);
//     cout<<"transformed!"<<endl;
//     at::Tensor flatTransform = hdt.mSubbandLF.getFlatBlock();
//     cout<<"returning"<<endl;
//     return flatTransform;
// }

// void encodePartitionKLT(Hierarchical4DEncoder& entropyCoder, double lambda,Block4D_ inputBlock, std::array<double,2> disparityRange, double& J0,double& conditionNumberH, double& conditionNumberV){
//     std::array<int64_t,4> length = {inputBlock.data.size(0),inputBlock.data.size(1),inputBlock.data.size(2),inputBlock.data.size(3)};
//     double scaledLambda = length[0]*length[1]*length[2]*length[3]*lambda;
//     inputBlock.data = inputBlock.data.contiguous();
    
//     entropyCoder.LoadOptimizerState();
//     //cout<<"Optimizer State Loaded!"<<endl;


//     double currGain = totalTransformGain_(length,length);
//     //cout<<"currGain: "<<currGain<<endl;
//     //inputBlock.ssi.print();
//     inputBlock.kltTransform(currGain);
//     conditionNumberH = (inputBlock.eigenValuesH[0]/inputBlock.eigenValuesH[-1]).item<double>();
//     conditionNumberV = (inputBlock.eigenValuesV[0]/inputBlock.eigenValuesV[-1]).item<double>();

//     ///cout<<"Transformed!"<<endl;
//     entropyCoder.mSubbandLF_ = inputBlock;
//     entropyCoder.mInferiorBitPlane = entropyCoder.OptimumBitplaneFaster_(scaledLambda);
//     std::cout<<entropyCoder.mInferiorBitPlane<<std::endl;

//     entropyCoder.LoadOptimizerState();
    
//     ProbabilityModel *currentCoderModelState;
//     entropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
//     std::array<int64_t,4> lengthTransform = {entropyCoder.mSubbandLF_.data.size(0), entropyCoder.mSubbandLF_.data.size(1), entropyCoder.mSubbandLF_.data.size(2), entropyCoder.mSubbandLF_.data.size(3)};
//     double Energy = 0;
//     double rate = 0;
//     double distortion = 0;
//     if(entropyCoder.mSegmentationTreeCodeBuffer != NULL){
//         delete [] entropyCoder.mSegmentationTreeCodeBuffer;
//     }
//     entropyCoder.mSegmentationTreeCodeBuffer = new char [2];
//     strcpy(entropyCoder.mSegmentationTreeCodeBuffer,"");

//     J0 = entropyCoder.RdOptimizeHexadecaTree_({0, 0, 0, 0}, lengthTransform, lambda, entropyCoder.mSuperiorBitPlane, &entropyCoder.mSegmentationTreeCodeBuffer, Energy,rate,distortion);
//     ProbabilityModel *coderModelState_0;
//     entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
//     entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);

//     entropyCoder.EncodeInteger(entropyCoder.mInferiorBitPlane, MINIMUM_BITPLANE_PRECISION);
//     entropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
//     //entropyCoder.EncodeSSI_(inputBlock.ssi);
//     std::cout<<"SIZES:"<<entropyCoder.mSubbandLF_.data.sizes()<<std::endl;
//     entropyCoder.EncodeSubblock_(scaledLambda);
//     std::cout<<"Encoded!"<<std::endl;

// }
// void encodeBlockKLT(Block4D_ block, Hierarchical4DEncoder& hdt, double lambda, std::array<double,2> disparityRange, double& J0, double& conditionNumberH, double& conditionNumberV){
    
//     hdt.RestartProbabilisticModel();
//     //std::cout<<"ILY"<<std::endl;

//     encodePartitionKLT(hdt, lambda,block, disparityRange, J0,conditionNumberH,conditionNumberV);
//     hdt.DoneEncoding();
// }
// at::Tensor decodeBlock(Hierarchical4DDecoder& hdt, double lambda, std::array<double,2> disparityRange, std::string inputImageFile){
     
//     hdt.RestartProbabilisticModel();

//     hdt.mSkipCount = 0;
//     hdt.mSkipMatrix = at::zeros({1,1,9*32,9*32});

//     //cout<<"Decoder Started!!"<<endl;
//     hdt.mInferiorBitPlane = hdt.DecodeInteger(MINIMUM_BITPLANE_PRECISION);
//     std::cout<<"Minimum Bit Plane: "<<hdt.mInferiorBitPlane<<std::endl;

//     int partitionFlag = hdt.DecodePartitionFlag();
//     cout<<"Partition Flag DECODED! "<<partitionFlag<<endl;
//     SgtSideInfo ssi = hdt.DecodeSsi(disparityRange);
//      cout<<"SSI DECODED!"<<endl;
//     ssi.print();     
//     hdt.mSubbandLF = Block4D_({9,9,32,32});
//     cout<<"Block Created!"<<endl;
//     hdt.mSubbandLF.emptyTransform();
//     cout<<"Block Emptied!"<<endl;

//     hdt.DecodeBlock(0, 0, 0, 0, hdt.mSubbandLF.transformSize[0], hdt.mSubbandLF.transformSize[1], hdt.mSubbandLF.transformSize[2], hdt.mSubbandLF.transformSize[3], hdt.mSuperiorBitPlane); 
//     write_tensor(log2(1+hdt.mSubbandLF.data.squeeze().abs()),inputImageFile);

//     cout<<"Block DECODED!"<<endl;
//     double gain = 288;
//     cout<<"Gain: "<<gain<<endl;
//     cout<<hdt.mSubbandLF.data.sizes()<<endl;
//     hdt.mSubbandLF.isgtTransform(gain,ssi);
//     cout<<"transformed!"<<endl;
//     at::Tensor flatTransform = hdt.mSubbandLF.getFlatBlock();
//     cout<<"returning"<<endl;
//     return flatTransform;


// //     at::Tensor flatTransform =at::mm(at::mm(sgtMatrixV, flatBlock), sgtMatrixH.t()).round().to(at::kInt);
// //     std::cout<<"Now, somewhere between the sacred silence and the sacred noise!"<<std::endl;
// //     at::Tensor newData = hdd.mSubbandLF.flat24D(flatTransform);
// //     cout<<"encoded:"<<endl;
// //     cout<<blockY.getFlatBlock().index({at::indexing::Slice(0,6),at::indexing::Slice(0,6)})<<endl;
// //     cout<<"decoded"<<endl;
// //     cout<<flatTransform.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6)})<<endl;
// //     cout<<"PSNR_Y = "<<10*log10((1023*1023)/(mse(flatTransform+512,blockY.getFlatBlock()+512)))<<endl;

// //         write_tensor(log2(1+flatBlock.squeeze().abs()),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/flatBlock.png");
// //         write_tensor(flatTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/flatTransform.png");
// //         write_tensor(newData[4][4],"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/decoded.png");

// //     cout<<"Decoded Max: "<<newData.max().item()<<endl;
// //     hdd.RestartProbabilisticModel();    


// //     // //hdd.mSuperiorBitPlane = hdt.mSuperiorBitPlane;
// //     // hdd.mSuperiorBitPlane = 30;
// //     // FILE *inputFileNamePointer;
// //     // if((inputFileNamePointer = fopen(outputDirectory.c_str(), "rb")) == NULL) {
// //     //     printf("Error: input file %s not found\n", outputDirectory.c_str());
// //     //     exit(0);
// //     // }

// //     // hdd.StartDecoder(inputFileNamePointer);
// //     // pd.mPartitionData = Block4D_({9,9,64,64});
// //     // hdd.RestartProbabilisticModel();
// //     // pd.DecodePartition(hdd);
// //     // cout<<"Decoded Max: "<<pd.mPartitionData.data.max().item()<<endl;


// //     // std::cout<<mse(pd.mPartitionData,R)<<std::endl;   
// //     fclose(inputFileNamePointer);
// }
// TEST(DecoderTests,AngleInfluence){
//     vector<double> lambdaVec = {25,250,2500,25000,250000};
//     //double lambda = 25000;
//     string originalInputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     LightField originalLF(9,9,32);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     originalLF.OpenLightFieldPPM_(originalInputDirectory,pattern,'r');
//         Block4D_ blockR= originalLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = originalLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = originalLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     Block4D_ blockYOriginal, blockCb, blockCr;
//     RGB2YCoCg_(blockYOriginal, blockCb, blockCr, blockR, blockG, blockB, originalLF.mPGMScale);
//     blockYOriginal.data = blockYOriginal.data - 512;
//     at::Tensor flatBlockOriginal = blockYOriginal.getFlatBlock();


//     for(double lambda : lambdaVec){
//         std::string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/AngleTests_Narrow_CabinetSideboard_"+to_string((int)lambda)+"/";
//         std::string inputImageDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/AngleTestImages_Narrow_CabinetSideboard_"+to_string((int)lambda)+"/";
//         create_directory(inputImageDirectory);
//         create_directory(inputDirectory);
//         ofstream psnrCSV = ofstream(inputDirectory+"psnr.csv");

//         double angleStart = 10;
//         double angleEnd = 16;
//         double resolution = 0.5;
//         for(double angleV = angleStart; angleV < angleEnd; angleV+=resolution){   
//             psnrCSV<<";";
//             psnrCSV<<angleV;
//         }
//         psnrCSV<<endl;
    
        
//         for(double angleH = angleStart; angleH < angleEnd; angleH+=resolution){   
//             psnrCSV<<angleH;
//             for(double angleV = angleStart; angleV < angleEnd; angleV+=resolution){
                
//                 std::string inputFile = inputDirectory + "angles_"+to_string((int)(angleV*10))+"_"+to_string((int)(angleH*10))+".comp";
//                 std::string inputImageFile = inputImageDirectory + "image_"+to_string((int)(angleV*10))+"_"+to_string((int)(angleH*10))+".png";
//                 FILE *inputFileNamePointer;
//                 if((inputFileNamePointer = fopen(inputFile.c_str(), "rb")) == NULL) {
//                     printf("Error: input file %s not found\n", inputFile.c_str());
//                     exit(0);
//                 }
//                 Hierarchical4DDecoder hdt;
//                 std::array<int64_t,4> lfSize,maxPartitionSize;
//                 std::array<double,2> disparityRange;
//                 int PGMScale;
//                 decodeHeader(inputFileNamePointer,hdt,lfSize, maxPartitionSize,disparityRange, PGMScale);
//                 hdt.StartDecoder(inputFileNamePointer);
//                 at::Tensor decodedBlock = decodeBlock(hdt,lambda,disparityRange,inputImageFile);
//                 fclose(inputFileNamePointer);
//                 cout<<"Decoded Block Size: "<<decodedBlock.sizes()<<" originalBlockSize: "<<flatBlockOriginal.sizes()<<endl;
//                 double psnr = 10*log10((1023*1023)/(mse(decodedBlock+512,flatBlockOriginal+512)));
//                 psnrCSV<<";"<<to_string(psnr);
//                 cout<<"angleV: "<<angleV<<" angleH: "<<angleH<<" PSNR: "<<psnr<<endl;

//             }
//             psnrCSV<<endl;

//         }
//         psnrCSV.close();
//     }
// }


// TEST(EncoderTests,AngleInfluence){
//     vector<double> lambdaVec = {25,250,2500,25000,250000};
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     LightField inputLF(9,9,32);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> lfSize = {inputLF.data.size(0),inputLF.data.size(1),inputLF.data.size(2),inputLF.data.size(3)};
//     std::array<int64_t,4> maxPartitionSize = {9,9,32,32};
//     std::array<double,2> disparityRange = {-3.5,3.5};
//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     cout<<"LF Loaded"<<endl;


//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//        cout<<"Color Fixed"<<endl;
//     blockY.data = blockY.data - 512;
//     blockCb.data = blockCb.data - 512;
//     blockCr.data = blockCr.data - 512;
//     cout<<"Shifted"<<endl;

//     double angleStart = 10;
//     double angleEnd = 16;
//     double resolution = 0.5;

//     for(double lambda : lambdaVec){
        
   
//             std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/AngleTests_Narrow_CabinetSideboard_"+to_string((int)lambda)+"/";
//             create_directory(outputDirectory);

//             ofstream bitsizes = ofstream(outputDirectory + "bitsizes.csv");
//             ofstream cost = ofstream(outputDirectory + "cost.csv");
//             ofstream cndNH = ofstream(outputDirectory + "conditionNumberH.csv");
//             ofstream cndNV = ofstream(outputDirectory + "conditionNumberV.csv");
//             write_tensor(blockY.getFlatBlock(),outputDirectory + "originalBlock.png");
//             cout<<outputDirectory + "originalBlock.png"<<endl;
//             cout<<outputDirectory + "bitsizes.csv"<<endl;
//             //Block4D_ blockY_ = blockY.clone(); 
//             //blockY_.ssi = SgtSideInfo(0,0,disparityRange);
//             //blockY_.ssi.print();
//             //blockY_.sgtTransform(1);
//             std::cout<<"Transformed!"<<std::endl;
            
//             for(double angleV = angleStart; angleV < angleEnd; angleV+=resolution){   
//                 bitsizes<<";";
//                 cost<<";";
//                 cndNH<<";";
//                 cndNV<<";";
//                 bitsizes<<angleV;
//                 cost<<angleV;
//             }
//             bitsizes<<endl;
//             cost<<endl;
//             cndNH<<endl;
//             cndNV<<endl;
            
//             for(double angleH = angleStart; angleH < angleEnd; angleH+=resolution){   
//                 bitsizes<<angleH;
//                 cost<<angleH;
//                 cndNH<<angleH;
//                 cndNV<<angleH;
//                 for(double angleV = angleStart; angleV < angleEnd; angleV+=resolution){
//                     std::string outputFile = outputDirectory + "angles_"+to_string((int)(angleV*10))+"_"+to_string((int)(angleH*10))+".comp";
//                     FILE *outputFileNamePointer;
//                     if((outputFileNamePointer = fopen(outputFile.c_str(), "wb")) == NULL) {
//                         printf("Error: input file %s not found\n", outputFile.c_str());
//                         exit(0);
//                     }
//                     Hierarchical4DEncoder hdt;
//                     hdt.StartEncoder(outputFileNamePointer);
//                     encodeHeader(outputFileNamePointer,hdt,lfSize,maxPartitionSize,disparityRange,inputLF.mPGMScale); 
//                     double J0 = 0;
//                     double conditionNumberH = 0;
//                     double conditionNumberV = 0;
//                     encodeBlock(blockY,hdt,lambda,disparityRange,angleH,angleV,J0,conditionNumberH,conditionNumberV);
//                     cout<<"Block Encoded"<<endl;
//                     fclose(outputFileNamePointer);
//                     int size = GetFileSize(outputFile);
//                     bitsizes<<";"<<to_string(size);
//                     cost<<";"<<to_string(J0);
//                     cndNH<<";"<<to_string(conditionNumberH);
//                     cndNV<<";"<<to_string(conditionNumberV);
                    
//                     cout<<outputFile<<endl;
//                     cout<<"angleV: "<<angleV<<" angleH: "<<angleH<<" size: "<<size<<endl;


//                 }
//                 bitsizes<<endl;
//                 cost<<endl;
//                 cndNH<<endl;
//                 cndNV<<endl;
//             }
            
//         bitsizes.close();
//         cost.close();
//         cndNH.close();
//         cndNV.close();
//     }

// }
// TEST(FrequencyTests,FindBasisFrequency){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     LightField inputLF(9,9,32);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> lfSize = {inputLF.data.size(0),inputLF.data.size(1),inputLF.data.size(2),inputLF.data.size(3)};
//     std::array<int64_t,4> maxPartitionSize = {9,9,32,32};
//     std::array<double,2> disparityRange = {-3.5,3.5};
//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     blockY.data = blockY.data - 512;
//     SgtSideInfo ssi = SgtSideInfo(47,47,disparityRange); 
//     at::Tensor modelCovMatH = blockY.calcModelCovMatrix(ssi,true); 
//     at::Tensor eigValsH;
//     at::Tensor sgtMatrixH = blockY.getSgtTransformMatrix(modelCovMatH,true,eigValsH);
//     at::Tensor basis = blockY.get2DBasis(sgtMatrixH,2);
//     write_tensor(basis,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/aNewBasis.png");
//     std::array<double,2> freq = blockY.getMainFrequency(basis,47);
//     cout<<"Freq: "<<freq[0]<<" "<<freq[1]<<endl;
// }

// TEST(FrequencyTests, BackWallQuadTree){
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/BackwallDetail/";
//     LightField inputLF(9,9,32);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> lfSize = {inputLF.data.size(0),inputLF.data.size(1),inputLF.data.size(2),inputLF.data.size(3)};
//     std::array<int64_t,4> maxPartitionSize = {9,9,32,32};
//     std::array<double,2> disparityRange = {-3.5,3.5};
//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     blockY.data = blockY.data - 512;
//     SgtSideInfo ssi = SgtSideInfo(47,47,disparityRange); 
//     at::Tensor modelCovMatH = blockY.calcModelCovMatrix(ssi,true); 
//     at::Tensor modelCovMatV = blockY.calcModelCovMatrix(ssi,false);
//     at::Tensor eigValsH, eigValsV;
//     at::Tensor sgtMatrixH = blockY.getSgtTransformMatrix(modelCovMatH,true,eigValsH);
//     at::Tensor sgtMatrixV = blockY.getSgtTransformMatrix(modelCovMatV,false,eigValsV);
//     at::Tensor flatBlock = blockY.getFlatBlock().to(at::kDouble) * 288;
//     at::Tensor freqOrderedSgt = blockY.frequencyOrderedSgt(flatBlock, sgtMatrixH, sgtMatrixV);
// }

// TEST(EncoderTests,KLTRDBehaviour){
//     std::vector<double> lambdaVec = {0,25,250,2500,25000,250000};
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     LightField inputLF(9,9,32);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> lfSize = {inputLF.data.size(0),inputLF.data.size(1),inputLF.data.size(2),inputLF.data.size(3)};
//     std::array<int64_t,4> maxPartitionSize = {9,9,32,32};
//     std::array<double,2> disparityRange = {-3.5,3.5};
//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     blockY.data = blockY.data - 512;
//     blockCb.data = blockCb.data - 512;
//     blockCr.data = blockCr.data - 512;
//     cout<<"Shifted"<<endl;
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/Sideboard_Backwall_KLT_Corr/";
//     std::string inputImageDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/Sideboard_Backwall_KLT_Corr_Images/";
//     create_directory(inputImageDirectory);
//     create_directory(outputDirectory);

//     ofstream bpp = ofstream(outputDirectory + "bpp.csv");
//     ofstream psnr = ofstream(outputDirectory + "psnr.csv");
//     write_tensor(blockY.getFlatBlock(),outputDirectory + "originalBlock.png");
//     at::Tensor flatBlockOriginal = blockY.getFlatBlock().to(at::kDouble);
//     // at::Tensor currCovH = flatBlockOriginal.t().cov();
//     // at::Tensor currCovV = flatBlockOriginal.cov();
//     // at::Tensor currCovH = flatBlockOriginal.t().corrcoef();
//     // at::Tensor currCovV = flatBlockOriginal.corrcoef();
//     at::Tensor currCovH = blockY.autoCorr(true);
//     at::Tensor currCovV = blockY.autoCorr(false);
//     cout<<outputDirectory + "originalBlock.png"<<endl;
//     cout<<outputDirectory + "bitsizes.csv"<<endl;
 
//     for(int lambdaInd = 0; lambdaInd< lambdaVec.size(); lambdaInd++){
//         double lambda  = lambdaVec[lambdaInd];
//         bpp<<";"<<lambda;
//         psnr<<";"<<lambda;
//     }
//     bpp<<endl;
//     psnr<<endl;

//     for(int lambdaInd = 0; lambdaInd< lambdaVec.size(); lambdaInd++){
//         double lambda  = lambdaVec[lambdaInd];
//         cout<<lambda<<endl;
//         std::string outputFile = outputDirectory + "backwall2_klt_corr_"+to_string((int)(lambda))+".comp";
//         std::string inputImageFile = inputImageDirectory + "backwall2_klt_corr_"+to_string((int)(lambda))+".png";
//         FILE *outputFileNamePointer;
//         if((outputFileNamePointer = fopen(outputFile.c_str(), "wb")) == NULL) {
//             printf("Error: input file %s not found\n", outputFile.c_str());
//             exit(0);
//         }
//         Hierarchical4DEncoder hdt;
//         hdt.StartEncoder(outputFileNamePointer);
//         encodeHeader(outputFileNamePointer,hdt,lfSize,maxPartitionSize,disparityRange,inputLF.mPGMScale); 
//         double J0 = 0;
//         double conditionNumberH = 0;
//         double conditionNumberV = 0;
//         encodeBlockKLT(blockY,hdt,lambda,disparityRange,J0, conditionNumberH, conditionNumberV);
//         cout<<"Block Encoded"<<endl;
//         fclose(outputFileNamePointer);
//         int size = GetFileSize(outputFile);
//         bpp<<";"<<to_string((double)(size*8)/(double)(lfSize[0]*lfSize[1]*lfSize[2]*lfSize[3]));
        
//         if((outputFileNamePointer = fopen(outputFile.c_str(), "rb")) == NULL) {
//             printf("Error: input file %s not found\n", outputFile.c_str());
//             exit(0);
//         }
//         Hierarchical4DDecoder hdd;
//         std::array<int64_t,4> lfSizeD,maxPartitionSizeD;
//         std::array<double,2> disparityRangeD;
//         int PGMScale;
//         decodeHeader(outputFileNamePointer,hdd,lfSizeD, maxPartitionSizeD,disparityRangeD, PGMScale);
//         hdd.StartDecoder(outputFileNamePointer);
//         at::Tensor decodedBlock = decodeKLT(hdd,lambda,disparityRangeD,inputImageFile,currCovH,currCovV);
//         fclose(outputFileNamePointer);
//         cout<<"Decoded Block Size: "<<decodedBlock.sizes()<<" originalBlockSize: "<<flatBlockOriginal.sizes()<<endl;
//         double psnrDouble = 10*log10((1023*1023)/(mse(decodedBlock+512,flatBlockOriginal+512)));
//         psnr<<";"<<to_string(psnrDouble);
//         //cout<<"angleV: "<<angleV<<" angleH: "<<angleH<<" PSNR: "<<psnrDouble<<endl;
    
    
//     }
    
//    psnr.close();
//    bpp.close();

// }


// TEST(EncoderTests,OptimalRDBehaviour){
//     std::vector<double> lambdaVec = {25,250,2500,25000,250000};
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     LightField inputLF(9,9,32);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> lfSize = {inputLF.data.size(0),inputLF.data.size(1),inputLF.data.size(2),inputLF.data.size(3)};
//     std::array<int64_t,4> maxPartitionSize = {9,9,32,32};
//     std::array<double,2> disparityRange = {-3.5,3.5};
//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     blockY.data = blockY.data - 512;
//     blockCb.data = blockCb.data - 512;
//     blockCr.data = blockCr.data - 512;
//     cout<<"Shifted"<<endl;
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/Sideboard_Sidewall_Optimal/";
//     std::string inputImageDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/Sideboard_Sidewall_Optimal_Images/";
//     create_directory(inputImageDirectory);
//     create_directory(outputDirectory);

//     ofstream bpp = ofstream(outputDirectory + "bpp.csv");
//     ofstream psnr = ofstream(outputDirectory + "psnr.csv");
//     write_tensor(blockY.getFlatBlock(),outputDirectory + "originalBlock.png");
//     at::Tensor flatBlockOriginal = blockY.getFlatBlock();
//     cout<<outputDirectory + "originalBlock.png"<<endl;
//     cout<<outputDirectory + "bitsizes.csv"<<endl;
//     // Block4D_ blockY_ = blockY.clone(); 
//     // blockY_.ssi = SgtSideInfo(0,0,disparityRange);
//     // blockY_.ssi.print();
//     // blockY_.sgtTransform(1);
//     // std::cout<<"Transformed!"<<std::endl;
//     double angleH = -45;
//     double angleV = -45;
    
//     for(int lambdaInd = 0; lambdaInd< lambdaVec.size(); lambdaInd++){
//         double lambda  = lambdaVec[lambdaInd];
//         bpp<<";"<<lambda;
//         psnr<<";"<<lambda;
//     }
//     bpp<<endl;
//     psnr<<endl;

//     for(int lambdaInd = 0; lambdaInd< lambdaVec.size(); lambdaInd++){
//         double lambda  = lambdaVec[lambdaInd];
//         std::string outputFile = outputDirectory + "sdb_pln_"+to_string((int)(lambda))+".comp";
//         std::string inputImageFile = inputImageDirectory + "sdb_pln_"+to_string((int)(lambda))+".png";
//         FILE *outputFileNamePointer;
//         if((outputFileNamePointer = fopen(outputFile.c_str(), "wb")) == NULL) {
//             printf("Error: input file %s not found\n", outputFile.c_str());
//             exit(0);
//         }
//         Hierarchical4DEncoder hdt;
//         double J0 = 0; 
//         double ch = 0;
//         double cv = 0;
//         hdt.StartEncoder(outputFileNamePointer);
//         encodeHeader(outputFileNamePointer,hdt,lfSize,maxPartitionSize,disparityRange,inputLF.mPGMScale); 
//         encodeBlock(blockY,hdt,lambda,disparityRange,angleH,angleV,J0,ch,cv);
//         //cout<<"Block Encoded"<<endl;
//         fclose(outputFileNamePointer);
//         int size = GetFileSize(outputFile);
//         bpp<<";"<<to_string((double)(size*8)/(double)(lfSize[0]*lfSize[1]*lfSize[2]*lfSize[3]));
        
//         if((outputFileNamePointer = fopen(outputFile.c_str(), "rb")) == NULL) {
//             printf("Error: input file %s not found\n", outputFile.c_str());
//             exit(0);
//         }
//         Hierarchical4DDecoder hdd;
//         std::array<int64_t,4> lfSizeD,maxPartitionSizeD;
//         std::array<double,2> disparityRangeD;
//         int PGMScale;
//         decodeHeader(outputFileNamePointer,hdd,lfSizeD, maxPartitionSizeD,disparityRangeD, PGMScale);
//         hdd.StartDecoder(outputFileNamePointer);
//         at::Tensor decodedBlock = decodeBlock(hdd,lambda,disparityRangeD,inputImageFile);
//         fclose(outputFileNamePointer);
//         //cout<<"Decoded Block Size: "<<decodedBlock.sizes()<<" originalBlockSize: "<<flatBlockOriginal.sizes()<<endl;
//         double psnrDouble = 10*log10((1023*1023)/(mse(decodedBlock+512,flatBlockOriginal+512)));
//         psnr<<";"<<to_string(psnrDouble);
//         cout<<"size: "<<size*8<< "Distortion: "<<mse(decodedBlock*288,flatBlockOriginal*288)*(32*32*9*9)<<endl;
    
    
//     }
    
//    psnr.close();
//    bpp.close();

// }

// TEST(EncoderTests,EstimatedRDBehaviour){
//     std::vector<double> lambdaVec = {25,250,2500,25000,250000};
//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/SidewallDetail/";
//     LightField inputLF(9,9,32);
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> lfSize = {inputLF.data.size(0),inputLF.data.size(1),inputLF.data.size(2),inputLF.data.size(3)};
//     std::array<int64_t,4> maxPartitionSize = {9,9,32,32};
//     std::array<double,2> disparityRange = {-3.5,3.5};
//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     blockY.data = blockY.data - 512;
//     blockCb.data = blockCb.data - 512;
//     blockCr.data = blockCr.data - 512;
//     cout<<"Shifted"<<endl;
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/Sideboard_Sidewall_Estimated/";
//     std::string inputImageDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/Sideboard_Sidewall_Estimated_Images/";
//     create_directory(inputImageDirectory);
//     create_directory(outputDirectory);

//     ofstream bpp = ofstream(outputDirectory + "bpp.csv");
//     ofstream psnr = ofstream(outputDirectory + "psnr.csv");
//     write_tensor(blockY.getFlatBlock(),outputDirectory + "originalBlock.png");
//     at::Tensor flatBlockOriginal = blockY.getFlatBlock();
//     cout<<outputDirectory + "originalBlock.png"<<endl;
//     cout<<outputDirectory + "bitsizes.csv"<<endl;
//     // Block4D_ blockY_ = blockY.clone(); 
//     // blockY_.ssi = SgtSideInfo(0,0,disparityRange);
//     // blockY_.ssi.print();
//     // blockY_.sgtTransform(1);
//     // std::cout<<"Transformed!"<<std::endl;

    
//     for(int lambdaInd = 0; lambdaInd< lambdaVec.size(); lambdaInd++){
//         double lambda  = lambdaVec[lambdaInd];
//         bpp<<";"<<lambda;
//         psnr<<";"<<lambda;
//     }
//     bpp<<endl;
//     psnr<<endl;

//     for(int lambdaInd = 0; lambdaInd< lambdaVec.size(); lambdaInd++){
//         double lambda  = lambdaVec[lambdaInd];
//         std::string outputFile = outputDirectory + "sdb_pln_"+to_string((int)(lambda))+".comp";
//         std::string inputImageFile = inputImageDirectory + "sdb_pln_"+to_string((int)(lambda))+".png";
//         FILE *outputFileNamePointer;
//         if((outputFileNamePointer = fopen(outputFile.c_str(), "wb")) == NULL) {
//             printf("Error: input file %s not found\n", outputFile.c_str());
//             exit(0);
//         }
//         Hierarchical4DEncoder hdt;
//         hdt.StartEncoder(outputFileNamePointer);
//         encodeHeader(outputFileNamePointer,hdt,lfSize,maxPartitionSize,disparityRange,inputLF.mPGMScale); 
//         hdt.StartEncoder(outputFileNamePointer);
//         TransformPartition tp;
//         tp.mlength_t_min = lfSize[0];
//         tp.mlength_s_min = lfSize[1];
//         tp.mlength_v_min = lfSize[2];
//         tp.mlength_u_min = lfSize[3];
// //     at::Tensor shiftOrderH, shiftOrderV;
//         hdt.RestartProbabilisticModel();
//         tp.RDoptimizeTransform_(blockY, hdt,disparityRange,1, lambda);
//         tp.EncodePartition_(hdt, lambda);
//         hdt.DoneEncoding();
//         fclose(outputFileNamePointer);

//         int size = GetFileSize(outputFile);
//         bpp<<";"<<to_string((double)(size*8)/(double)(lfSize[0]*lfSize[1]*lfSize[2]*lfSize[3]));
        
//         if((outputFileNamePointer = fopen(outputFile.c_str(), "rb")) == NULL) {
//             printf("Error: input file %s not found\n", outputFile.c_str());
//             exit(0);
//         }
//         Hierarchical4DDecoder hdd;
//         std::array<int64_t,4> lfSizeD,maxPartitionSizeD;
//         std::array<double,2> disparityRangeD;
//         int PGMScale;
//         decodeHeader(outputFileNamePointer,hdd,lfSizeD, maxPartitionSizeD,disparityRangeD, PGMScale);
//         hdd.StartDecoder(outputFileNamePointer);
//         at::Tensor decodedBlock = decodeBlock(hdd,lambda,disparityRangeD,inputImageFile);
//         fclose(outputFileNamePointer);
//         //cout<<"Decoded Block Size: "<<decodedBlock.sizes()<<" originalBlockSize: "<<flatBlockOriginal.sizes()<<endl;
//         std::cout<<lambda<<":"<<mse(decodedBlock+512,flatBlockOriginal+512)<<std::endl<<std::endl;
//         double psnrDouble = 10*log10((1023*1023)/(mse(decodedBlock+512,flatBlockOriginal+512)));
//         psnr<<";"<<to_string(psnrDouble);
//         //cout<<"angleV: "<<angleV<<" angleH: "<<angleH<<" PSNR: "<<psnrDouble<<endl;
    
    
//     }
    
//     psnr.close();
//     bpp.close();

// }
// TEST(EncoderTests,OptimizePartition){
//     cout<<"OptimizePartition"<<endl;
//     double lambda = 250000;

//     string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/TEST-DUMMY-ORIGINAL/";
//     std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/decode-sideboard-"+to_string((int)lambda)+".comp";
//     string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
//     LightField inputLF(9,9,32);
//     inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
//     std::array<int64_t,4> lfSize = {inputLF.data.size(0),inputLF.data.size(1),inputLF.data.size(2),inputLF.data.size(3)};
//     std::array<int64_t,4> maxPartitionSize = {9,9,32,32};
//     std::array<double,2> disparityRange = {-3.5,3.5};

//     Block4D_ blockR= inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},0);
//     Block4D_ blockG = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},1);
//     Block4D_ blockB = inputLF.ReadBlock4DfromLightField_({9,9,32,32},{0,0,0,0},2);
//     cout<<"LF Loaded"<<endl;


//     Block4D_ blockY, blockCb, blockCr;
//     RGB2YCoCg_(blockY, blockCb, blockCr, blockR, blockG, blockB, inputLF.mPGMScale);
//     cout<<"Color Fixed"<<endl;
//     blockY.data = blockY.data - 512;
//     blockCb.data = blockCb.data - 512;
//     blockCr.data = blockCr.data - 512;
//     cout<<"Shifted"<<endl;

//     write_tensor(blockY.getFlatBlock(),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/originalY.png");
//     //at::Tensor currCovV = blockY.covFun2Mat(blockY.covFun(false),false);
//     //at::Tensor currCovH = blockY.covFun2Mat(blockY.covFun(true),true);
//     //at::Tensor currCovH = blockY.batchedCovMatrix(true);
//     //at::Tensor currCovV = blockY.batchedCovMatrix(false);
//     at::Tensor currCovH = blockY.getFlatBlock().to(at::kDouble).t().cov();
//     at::Tensor currCovV = blockY.getFlatBlock().to(at::kDouble).cov();

//     FILE *outputFileNamePointer;
//     if((outputFileNamePointer = fopen(outputDirectory.c_str(), "wb")) == NULL) {
//         printf("Error: input file %s not found\n", outputDirectory.c_str());
//         exit(0);
//     }
//     Hierarchical4DEncoder hdt;

//     encodeHeader(outputFileNamePointer,hdt,lfSize,maxPartitionSize,disparityRange,inputLF.mPGMScale); 
//     hdt.StartEncoder(outputFileNamePointer);
//     TransformPartition tp;
//     tp.mlength_t_min = 9;
//     tp.mlength_s_min = 9;
//     tp.mlength_v_min = 32;
//     tp.mlength_u_min = 32;
//     at::Tensor shiftOrderH, shiftOrderV;
//     hdt.RestartProbabilisticModel();
//     tp.RDoptimizeTransform_(blockY, hdt,disparityRange,1, lambda);
//     tp.EncodePartition_(hdt, lambda);
//     at::Tensor shift_orderH = hdt.mSubbandLF_.getOrderH();
//     at::Tensor shift_orderV = hdt.mSubbandLF_.getOrderV();
//     cout<<shift_orderH.sizes()<<endl;
//     cout<<shift_orderV.sizes()<<endl;
//     // tp.RDoptimizeTransform_(blockCb, hdt,{-3,3},1, lambda);
//     // tp.EncodePartition_(hdt, lambda);
//     // tp.RDoptimizeTransform_(blockCr, hdt,{-3,3},1, lambda);
//     // tp.EncodePartition_(hdt, lambda);
//     hdt.DoneEncoding();
//     fclose(outputFileNamePointer);
 


//     PartitionDecoder pd;
//     Hierarchical4DDecoder hdd;

//     FILE *inputFileNamePointer;
//     if((inputFileNamePointer = fopen(outputDirectory.c_str(), "rb")) == NULL) {
//         printf("Error: input file %s not found\n", outputDirectory.c_str());
//         exit(0);
//     }
//      hdt.mSuperiorBitPlane = BigEndianUnsignedIntegerRead_(2, inputFileNamePointer);
//     std::cout<<"SuperiorBitPlane: "<<hdt.mSuperiorBitPlane<<std::endl;

//     //reads the maximum Partition sizes
//     array<int64_t, 4> decoderMaxPartitionSize;
//     for (int n = 0; n < 4; n++) {
//         decoderMaxPartitionSize[n] = BigEndianUnsignedIntegerRead_(2, inputFileNamePointer);
//     }
//     std::cout<<"decodrrMaxPartitionSize: "<<decoderMaxPartitionSize[0]<<" "<<decoderMaxPartitionSize[1]<<" "<<decoderMaxPartitionSize[2]<<" "<<decoderMaxPartitionSize[3]<<std::endl;
//     //reads the LightField Size
//     std::array<int64_t,4> lfSizeDecoder;
//     for(int n = 0; n < 4; n ++ ){
//         lfSizeDecoder[n] = BigEndianUnsignedIntegerRead_(2, inputFileNamePointer);
//     }
//     std::cout<<"LightField Size: "<<lfSize[0]<<" "<<lfSize[1]<<" "<<lfSize[2]<<" "<<lfSize[3]<<std::endl;
//     std::array<double,2> decoderDisparityRange;
//     for(int n = 0; n < 2; n++) {
//         decoderDisparityRange[n] =(double) BigEndianSignedIntegerRead_( 2, inputFileNamePointer)/100.0;
//     }
//     std::cout<<"DisparityRange: "<<decoderDisparityRange[0]<<" "<<decoderDisparityRange[1]<<std::endl;
//     int PGMScale = BigEndianUnsignedIntegerRead_( 2, inputFileNamePointer);
//     std::cout<<"PGMScale: "<<PGMScale<<std::endl;

//     hdd.StartDecoder(inputFileNamePointer);
//     hdd.RestartProbabilisticModel();

//     hdd.mSkipCount = 0;
//     hdd.mSkipMatrix = at::zeros({1,1,9*32,9*32});

//     cout<<"Decoder Started!!"<<endl;
//     hdd.mInferiorBitPlane = hdd.DecodeInteger(MINIMUM_BITPLANE_PRECISION);
//     std::cout<<"Minimum Bit Plane: "<<hdd.mInferiorBitPlane<<std::endl;




//     int partitionFlag = hdd.DecodePartitionFlag();
//     cout<<"Partition Flag DECODED! "<<partitionFlag<<endl;
//     SgtSideInfo ssi = hdd.DecodeSsi(decoderDisparityRange);
//     cout<<"SSI DECODED!"<<endl;
//     ssi.print();
//     hdd.mSubbandLF = Block4D_({9,9,32,32});
//     cout<<"Block Created!"<<endl;
//     hdd.mSubbandLF.emptyTransform();
//     cout<<"Block Emptied!"<<endl;

//     hdd.DecodeBlock(0, 0, 0, 0, hdd.mSubbandLF.transformSize[0], hdd.mSubbandLF.transformSize[1], hdd.mSubbandLF.transformSize[2], hdd.mSubbandLF.transformSize[3], hdd.mSuperiorBitPlane); 
    
//     cout<<"Block DECODED!"<<endl;
//     double gain = 288;
//     cout<<"Gain: "<<gain<<endl;
//     //hdd.mSubbandLF.isgtTransform(gain,ssi);
//     cout<<hdd.mSubbandLF.data.sizes()<<endl;
//     write_tensor(log2(1+hdd.mSubbandLF.data.squeeze().abs()),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/decodedSGT_sideboard_"+to_string((int)lambda) + ".png");
//     at::Tensor flatBlock = hdd.mSubbandLF.data.squeeze().to(at::kDouble)/gain; 
//     cout<<"Decoder Gain: "<<gain<<endl;
//     write_tensor(log2(1+flatBlock.squeeze().abs()),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/flatBlockEarly.png");

//     at::Tensor modelCovMatH = hdd.mSubbandLF.calcModelCovMatrix(ssi,true); 
//     at::Tensor modelCovMatV = hdd.mSubbandLF.calcModelCovMatrix(ssi,false);
//     at::Tensor eigValsH,eigValsV;
//     //at::Tensor sgtMatrixH = hdd.mSubbandLF.getSgtTransformMatrix(currCovH,true,eigValsH);
//     //at::Tensor sgtMatrixV = hdd.mSubbandLF.getSgtTransformMatrix(currCovV,false,eigValsV);
//     //at::Tensor sgtMatrixH   = hdd.mSubbandLF.getSgtTransformMatrix(currCovH,true,eigValsH);
//     //at::Tensor sgtMatrixV   = hdd.mSubbandLF.getSgtTransformMatrix(currCovV,false,eigValsV);
//     at::Tensor sgtMatrixH = hdd.mSubbandLF.getSgtTransformMatrix(modelCovMatH,true,eigValsH);
//     at::Tensor sgtMatrixV = hdd.mSubbandLF.getSgtTransformMatrix(modelCovMatV,false,eigValsH);
//     std::cout<<"Did I get here?"<<std::endl;
//     //sgtMatrixH = sgtMatrixH.index({at::indexing::Slice(),shift_orderH.to(at::kLong)});
//     //sgtMatrixV = sgtMatrixV.index({at::indexing::Slice(),shift_orderV.to(at::kLong)});
//     std::cout<<"HOW DO YOU OWN DISORDER DISORDER! "<<sgtMatrixH.sizes()<<" "<<sgtMatrixV.sizes()<<" "<<flatBlock.sizes()<<std::endl; 
//     at::Tensor flatTransform =at::mm(at::mm(sgtMatrixV, flatBlock), sgtMatrixH.t()).round().to(at::kInt);
//     std::cout<<"Now, somewhere between the sacred silence and the sacred noise!"<<std::endl;
//     at::Tensor newData = hdd.mSubbandLF.flat24D(flatTransform);
//     cout<<"encoded:"<<endl;
//     cout<<blockY.getFlatBlock().index({at::indexing::Slice(0,6),at::indexing::Slice(0,6)})<<endl;
//     cout<<"decoded"<<endl;
//     cout<<flatTransform.index({at::indexing::Slice(0,6),at::indexing::Slice(0,6)})<<endl;
//     cout<<"PSNR_Y = "<<10*log10((1023*1023)/(mse(flatTransform+512,blockY.getFlatBlock()+512)))<<endl;

//         write_tensor(log2(1+flatBlock.squeeze().abs()),"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/flatBlock.png");
//         write_tensor(flatTransform,"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/flatTransform.png");
//         write_tensor(newData[4][4],"/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/decoded.png");

//     cout<<"Decoded Max: "<<newData.max().item()<<endl;
//     hdd.RestartProbabilisticModel();    


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
//     fclose(inputFileNamePointer);


// }
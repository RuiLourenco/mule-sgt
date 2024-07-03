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

TEST(EncoderTests,OptimizePartition){
    // string inputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/Mule_Slant/LightFields/greek/";
    std::string outputDirectory = "/nfs/home/ruilourenco.it/Documents/Code/mule-sgt/data/example_l0.comp";
    // string pattern = R"((?P<U>.*)_(?P<V>.*)\.ppm)";
    // double lambda = 0;
    // LightField inputLF(9,9,512);
    // inputLF.OpenLightFieldPPM_(inputDirectory,pattern,'r');
    // Block4D_ block = inputLF.ReadBlock4DfromLightField_({9,9,64,64},{0,0,34,34},0);
    
    // std::cout<<"block var = "<<block.data.to(at::kDouble).var()<<endl;
    // //std::cout<<block.data[0][0]<<std::endl;
    // cout<<"Block is Ready!"<<endl;
    // FILE *outputFileNamePointer;
    // if((outputFileNamePointer = fopen(outputDirectory.c_str(), "wb")) == NULL) {
    //     printf("Error: input file %s not found\n", outputDirectory.c_str());
    //     exit(0);
    // }
    
    // Hierarchical4DEncoder hdt;
    // hdt.StartEncoder(outputFileNamePointer);

    // TransformPartition tp;
    // tp.mlength_t_min = 9;
    // tp.mlength_s_min = 9;
    // tp.mlength_v_min = 4;
    // tp.mlength_u_min = 4;
    // hdt.RestartProbabilisticModel();
    // tp.RDoptimizeTransform_(block, hdt,{-3,3},1, lambda);
    // tp.EncodePartition_(hdt, lambda);
    // hdt.DoneEncoding();
    // fclose(outputFileNamePointer);
    // cout<<"Encoded Max: "<<block.data.max().item()<<endl;

    PartitionDecoder pd;
    Hierarchical4DDecoder hdd;

    //hdd.mSuperiorBitPlane = hdt.mSuperiorBitPlane;
    hdd.mSuperiorBitPlane = 30;
    FILE *inputFileNamePointer;
    if((inputFileNamePointer = fopen(outputDirectory.c_str(), "rb")) == NULL) {
        printf("Error: input file %s not found\n", outputDirectory.c_str());
        exit(0);
    }

    hdd.StartDecoder(inputFileNamePointer);
    pd.mPartitionData = Block4D_({9,9,64,64});
    hdd.RestartProbabilisticModel();
    pd.DecodePartition(hdd);
    cout<<"Decoded Max: "<<pd.mPartitionData.data.max().item()<<endl;


    //std::cout<<mse(pd.mPartitionData,block)<<std::endl;   
    fclose(inputFileNamePointer);


}
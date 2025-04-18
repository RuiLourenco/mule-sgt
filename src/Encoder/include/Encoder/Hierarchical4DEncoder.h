/* 
 * File:   Hierarchical4DEncoder.h
 * Author: murilo
 *
 * Created on December 28, 2017, 11:41 AM
 */
#include "LightField/Block4D.h"
#include "LightField/Block4D_.h"
#include "Encoder/ABACoder.h"
#include "ProbabilityModel/ProbabilityModel.h"



#ifndef HIERARCHICAL4DENCODER_H
#define HIERARCHICAL4DENCODER_H

//#define MAX_DEPH_CONDICIONING 9
#define BITPLANE_BYPASS -1
#define BITPLANE_BYPASS_FLAGS -1
#define SYMBOL_PROBABILITY_MODEL_INDEX 1
#define SEGMENTATION_PROB_MODEL_INDEX 32
#define NUMBER_OF_MODELS 161

class Hierarchical4DEncoder {
public:
    double mRate  = 0;
    double mDistortion = 0;
    Block4D mSubbandLF;   
    Block4D_ mSubbandLF_;   
    at::Tensor ignored;
    double currCost;
    ABACoder mEntropyCoder;
    ProbabilityModel *mPmodel;
    ProbabilityModel *mOptimizationPmodel;
    int flagZero = 0;
    int flagOne = 0;
    int flagTwo = 0;
    int mIgnored = 0;
    double mIgnoreEfficiency;
    int mSuperiorBitPlane, mInferiorBitPlane;
    int mSegmentationFlagProbabilityModelIndex;
    int mSymbolProbabilityModelIndex;
    int mPreSegmentation;
    char *mSegmentationTreeCodeBuffer;
    long int mSegmentationTreeCodeBufferSize;
    int OptimumBitplaneFaster_(double lambda);

    Hierarchical4DEncoder(void);
    ~Hierarchical4DEncoder(void);
    void StartEncoder(FILE *outputFilePointer);
    void RestartProbabilisticModel(void);
    void EncodeBlock(int position_t, int position_s, int position_v, int position_u, int length_t, int length_s, int length_v, int length_u, int bitplane);
    void EncodeBlock_(std::array<int64_t,4> position,std::array<int64_t, 4> length, int bitplane);
    void EncodeCoefficient(int coefficient, int bitplane);
    void EncodeSegmentationFlag(int flag, int bitplane);
    void EncodePartitionFlag(int flag);
    void EncodeSSI_(SgtSideInfo ssi);
    void EncodeInteger(int integerValue, int precision);
    void EncodeAll(double lambda, int inferiorBitPlane);
    void EncodeSubblock(double lambda);
    void EncodeSubblock_(double lambda);
    double RdOptimizeHexadecaTree(int position_t, int position_s, int position_v, int position_u, int length_t, int length_s, int length_v, int length_u, double lambda, int bitplane, char **codeString, double &signalEnergy);
    double RdOptimizeHexadecaTree_(std::array<int64_t,4> position,std::array<int64_t, 4> length, double lambda, int bitplane, char **codeString, double &signalEnergy,double& rate, double& distortion);
    void RdEncodeHexadecatree(int position_t, int position_s, int position_v, int position_u, int length_t, int length_s, int length_v, int length_u, int bitplane, int &flagIndex);
    void RdEncodeHexadecatree_(std::array<int64_t,4> position,std::array<int64_t, 4> length, int bitplane, int &flagIndex);
    void DoneEncoding(void);
    void SetDimension(int length_t, int length_s, int length_v, int length_u);
    int OptimumBitplane_(double lambda);
    int OptimumBitplane(double lambda);
    void LoadOptimizerState(void);
    void GetOptimizerProbabilisticModelState(ProbabilityModel **state);
    void SetOptimizerProbabilisticModelState(ProbabilityModel *state);
    void DeleteProbabilisticModelState(ProbabilityModel *state);
};
#endif /* HIERARCHICAL4DENCODER_H */


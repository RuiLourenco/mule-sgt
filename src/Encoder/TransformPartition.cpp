#include "Encoder/TransformPartition.h"
#include <omp.h>
#include <vector>
#include <numeric>   // For std::iota
#include <algorithm> // For std::min_element
#include <iterator>  // For std::distance
#include <chrono>


/*******************************************************************************/
/*                      TransformPartition class methods                    */
/*******************************************************************************/

// TransformPartition :: TransformPartition(void) {
//     mPartitionCode = NULL;
    
//     //mUseSameBitPlane = 1;
// }
TransformPartition :: TransformPartition(std::array<int64_t,4> minLength, Hierarchical4DEncoder& entropyCoder,std::array<double,2> disparityRange, double transformGain)
    : mEntropyCoder(entropyCoder), mDisparityRange(disparityRange), mGain(transformGain) {
    
    create_encoder_pool(omp_get_max_threads(),mEntropyCoder.mProcessingContext.image_height ,mEntropyCoder.mProcessingContext.image_width);
    std::cout<<"Using "<<m_encoder_pool.size()<<" threads for encoding."<<std::endl;
    //std::cout<<"MBP : "<<mEntropyCoder.mInferiorBitPlane<<std::endl;

    mPartitionCode = NULL;
    mlength_t_min = minLength[0];
    mlength_s_min = minLength[1];
    mlength_v_min = minLength[2];
    mlength_u_min = minLength[3];

    std::cout<<"We gucci"<<std::endl;
    
}
void TransformPartition::create_encoder_pool(size_t num_threads, size_t height, size_t width) {
    
    if (num_threads == 0) num_threads = 1; // Sanity check

    // This logic is now cleanly isolated.
    m_encoder_pool.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        m_encoder_pool.emplace_back(std::make_unique<Hierarchical4DEncoder>(height, width));
    }
    // Return the fully constructed vector
}
TransformPartition :: ~TransformPartition(void) {
    if(mPartitionCode != NULL)
        delete [] mPartitionCode;
}
double TransformPartition :: totalTransformGain(void){

    return mGain*sqrt(mPartitionData_.size[0]*mPartitionData_.size[1]*mPartitionData_.size[2]*mPartitionData_.size[3]);

}
void TransformPartition :: RDoptimizeTransform(Block4D &inputBlock, double lambda){
    std::cout<<"Starting RDoptimizeTransform with lambda: " << lambda << std::endl;
    mEntropyCoder.RestartProbabilisticModel();
    for(int i = 0; i < m_encoder_pool.size(); i++) {
        m_encoder_pool[i]->RestartProbabilisticModel();
    }

    inputBlock.data = inputBlock.data.contiguous();


    mCodingUnitIndex = 0;
    if(mPartitionCode != NULL)
        delete [] mPartitionCode;
    mPartitionCode = new char [1];
    mPartitionCode[0] = 0;          //initializes the partition code string as the null string
    mEvaluateOptimumBitPlane = 1;
    mPartitionData_ = Block4D(inputBlock.size,inputBlock.lightFieldPosition,inputBlock.lightField);

    double scaledLambda = lambda;
    for (int i = 0; i < 4; i++){
        scaledLambda *= inputBlock.size[i];
    }
    mLambda = scaledLambda;
    mEntropyCoder.LoadOptimizerState();

    Block4D transformedBlock(inputBlock.size,inputBlock.lightFieldPosition,inputBlock.lightField);


    transformedBlock.emptyTransform();
    mDepth = 0;
    getOptimalMinimumBitPlane(inputBlock);
    mLagrangianCost = RDoptimizeTransformStep(inputBlock, transformedBlock, {0,0,0,0}, inputBlock.size, &mPartitionCode);
    std::cout<<"TRANSFORMED BLOCK SIZE: "<<transformedBlock.data.sizes()<<std::endl;
    mPartitionData_ = transformedBlock;

    mEntropyCoder.LoadOptimizerState();

}

void TransformPartition :: getOptimalMinimumBitPlane(Block4D& inputBlock){
    // This function is used to find the optimal minimum bit plane for the input block.
    // It sets the mInferiorBitPlane of the encoder to the optimal value.
    Block4D block_0 = inputBlock.clone();
    block_0.kltTransform(this->totalTransformGain());

    mEntropyCoder.mSubbandLF_ = block_0;
    mEntropyCoder.RestartProbabilisticModel();
    mEntropyCoder.mInferiorBitPlane = mEntropyCoder.OptimumBitplaneFaster_(mLambda);
    mEntropyCoder.LoadOptimizerState();
    for(int i = 0; i < m_encoder_pool.size(); i++) {
        m_encoder_pool[i]->mInferiorBitPlane = mEntropyCoder.mInferiorBitPlane;
    }
}



double TransformPartition :: EvaluatePartition(Hierarchical4DEncoder& encoder, Block4D &block_0, double currGain){
    

    
    
    block_0.kltTransform(currGain);

    encoder.mSubbandLF_ = block_0;


    double Energy;
    double rate = 0;
    double distortion = 0;

    std::array<int64_t,4> lengthTransform = {encoder.mSubbandLF_.data.size(0), encoder.mSubbandLF_.data.size(1), encoder.mSubbandLF_.data.size(2), encoder.mSubbandLF_.data.size(3)};
    std::chrono::steady_clock::time_point sEncode = std::chrono::steady_clock::now();
    double J0 = encoder.build_optimal_tree_from_pool(lengthTransform,{0,0,0,0}, encoder.mSuperiorBitPlane, mLambda);
    std::chrono::steady_clock::time_point fEncode = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> eEncode= fEncode- sEncode;

    double weight = totalTransformGain();
    distortion = distortion/(block_0.size[0]*block_0.size[1]*block_0.size[2]*block_0.size[3]);
    rate = rate/(block_0.size[0]*block_0.size[1]*block_0.size[2]*block_0.size[3]);
    distortion = (double) distortion/(weight*weight);
    distortion = 10 * log10((1024*1024)/distortion);
    return J0;
}


double TransformPartition :: RDoptimizeTransformStep(Block4D &inputBlock, Block4D &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length,char **partitionCode) {

    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    for(int i = 0; i < m_encoder_pool.size(); i++){
        m_encoder_pool[i]->SetOptimizerProbabilisticModelState(currentCoderModelState);
    }   
    
    //partitionCodeS handles splitting in the spatial dimension, partitionCodeV handles splitting in the view dimension.
    char *partitionCodeS=NULL;
    std::array<int64_t,4> lightFieldPosition = inputBlock.lightFieldPosition;
    for(int i = 0; i < 4; i++){
        lightFieldPosition[i] += position[i]; 
    }


    Block4D block_0 = inputBlock.copySubblock(length,position);

    Block4D blockOrig = block_0;
    Block4D temp_block_0 = block_0;

    ProbabilityModel *coderModelState_0;
    ProbabilityModel *coderModelStateInitial;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelStateInitial);
    double currGain = totalTransformGain();

   
    double J0 = EvaluatePartition(mEntropyCoder,block_0,currGain);
    //block_0 = temp_block_0;
    //We save the state of the entropy coder after encoding block_0 without further partitioning
    mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    //We reset the state of the entropy coder to the initial state
    mEntropyCoder.SetOptimizerProbabilisticModelState(coderModelStateInitial);
    
    

    double JS = -1.0;
    Block4D transformedBlockS(length,lightFieldPosition,inputBlock.lightField);
    //Trivial transformation to 2D of an empty block in the basic case. 
    //Some complexity is needed if the block includes invalid corners.
    transformedBlockS.emptyTransform();

    //If you can split more in the spatial dimension
    if((length[3] >= 2*mlength_u_min)&&(length[2] >= 2*mlength_v_min)) {
        mDepth++;
        JS = 0.0;

        //Create partition codes for each subblock
        char *partitionCodeS00 = new char[1];
        char *partitionCodeS01 = new char[1];
        char *partitionCodeS10 = new char[1];
        char *partitionCodeS11 = new char[1];
        
        //set partition codes to the null string
        partitionCodeS00[0] = 0;
        partitionCodeS01[0] = 0;
        partitionCodeS10[0] = 0;
        partitionCodeS11[0] = 0;
       
        std::array<int64_t,4> new_position, new_length, new_lightField_position;
        //position remains the same
        new_position = position;
        new_lightField_position = lightFieldPosition;

        
        //length is halved in the spatial dimensions!
        new_length[0] = length[0];
        new_length[1] = length[1];
        new_length[2] = length[2]/2;
        new_length[3] = length[3]/2;
          
        //optimize partition for Block_S returning JS, the transformed Block_S, partitionCode_S and arithmetic_model_S
        Block4D transformedBlockS00(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS00.emptyTransform();

        
        //Need to see what this is actually doing...
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS00, new_position, new_length, &partitionCodeS00);

        new_position[3] = position[3] + length[3]/2;
        //new_lightField_position[3] = lightFieldPosition[3] + new_position[3];
        new_length[3] = length[3] - length[3]/2; 
                 
        Block4D transformedBlockS01(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS01.emptyTransform();
        
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS01, new_position, new_length, &partitionCodeS01);

        new_position[2] = position[2] + length[2]/2;

        new_length[2] = length[2] - length[2]/2;
        
        Block4D transformedBlockS11(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS11.emptyTransform();
        
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS11, new_position, new_length, &partitionCodeS11);

        new_position[3] = position[3];
        new_lightField_position[3] = lightFieldPosition[2] + new_position[3];
        
        new_length[3] = length[3]/2;
        
        Block4D transformedBlockS10(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS10.emptyTransform();
        
        
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS10, new_position, new_length, &partitionCodeS10);


        
        //concatenate the partition codes of the four blocks 
        //(I haven't optimized this exterior recursion yet AT ALL
        //We are creating new partitions at each level of recursion
        //So if the image got too large or we have too many recursion levels
        //this could get problematic memory-wise. But this wasn't a bottleneck in my usecase. 
        //Might be the new bottleneck for the KLT version, though.
        //The arena implementation similar to the one used for the arithmetic coder could help here.) 
        
        partitionCodeS = new char [2+strlen(partitionCodeS00)+strlen(partitionCodeS01)+strlen(partitionCodeS10)+strlen(partitionCodeS11)];
        strcpy(partitionCodeS, partitionCodeS00);
        strcat(partitionCodeS, partitionCodeS01);
        strcat(partitionCodeS, partitionCodeS11);
        strcat(partitionCodeS, partitionCodeS10);
        

        transformedBlockS = Block4D(transformedBlockS00,transformedBlockS01,transformedBlockS10,transformedBlockS11,false);

        transformedBlockS.sgtDomain = true;
        
        delete [] partitionCodeS00;
        delete [] partitionCodeS01;
        delete [] partitionCodeS10;
        delete [] partitionCodeS11; 
    }

    ProbabilityModel *coderModelState_s=NULL;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_s);
    
    //Restores the current arithmetic model using current_model. 
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);

    if(J0 > 0) 
        J0 += 1.0*mLambda;
    if(JS > 0)
        JS += 2.0*mLambda;
    
    
    
    //choose the lower cost and returns the corresponding cost,  the partition code and the arithmetic coder model
    //find best J
    int interview_split = 0;
    int intraview_split = 0;
    int no_split = 0;


    if(JS >= 0) {
        if(JS < J0) {
            intraview_split = 1;
        }
        else {
            no_split = 1;
        }
    }else {
        no_split = 1;
    }
    
    double optimumJ=0;
    if((interview_split + intraview_split + no_split) != 1) {
        printf("ERROR: partition fail/n");
        exit(0);
    }

    //reallocates memory for the partitionCode string based on the current length and the length of the chosen one
    //copies data from the chosen arithmetic coder model to the current model
    char flagCode[2];
    flagCode[1] = 0;
    if(intraview_split == 1) {
        optimumJ = JS;
        char *code = new char[2+strlen(*partitionCode)+strlen(partitionCodeS)];
        strcpy(code, *partitionCode);
        flagCode[0] = INTRAVIEWSPLITFLAG;
        strcat(code, flagCode);
        strcat(code, partitionCodeS);
        delete(*partitionCode);
        *partitionCode = code;
        mEntropyCoder.SetOptimizerProbabilisticModelState(coderModelState_s);
        
        //transformedBlock.CopySubblockFrom(transformedBlockS, {0,0,0,0},{0,0,0,0});
        transformedBlock = transformedBlockS.clone();


    }
    if(no_split == 1) {
        optimumJ = J0;   
        char *code = new char[2+strlen(*partitionCode)];
        strcpy(code, *partitionCode);
        flagCode[0] = NOSPLITFLAG;
        strcat(code, flagCode);
        delete(*partitionCode);
        *partitionCode = code;
        mEntropyCoder.SetOptimizerProbabilisticModelState(coderModelState_0);
        //transformedBlock.CopySubblockFrom(block_0, {0,0,0,0},{0,0,0,0});
        transformedBlock = block_0.clone();

    }

    if(partitionCodeS != NULL) {
        delete [] partitionCodeS;
    }

    mEntropyCoder.DeleteProbabilisticModelState(currentCoderModelState);
    mEntropyCoder.DeleteProbabilisticModelState(coderModelState_0);
    mEntropyCoder.DeleteProbabilisticModelState(coderModelState_s);

    return(optimumJ);     
}

void TransformPartition :: EncodePartition(double lambda){

    double scaledLambda = lambda;
    std::array<int64_t,4> length;

    for(int i = 0; i < 4; ++i){
        length[i] = mPartitionData_.size[i];
        scaledLambda*=length[i];
    }

    mLambda = scaledLambda;
    
    std::array<int64_t,4> position = {0,0,0,0};    
    mPartitionCodeIndex = 0;
      
    mEntropyCoder.EncodeInteger(mEntropyCoder.mInferiorBitPlane, MINIMUM_BITPLANE_PRECISION);

    EncodePartitionStep(position, length, scaledLambda);
}

void TransformPartition :: EncodePartitionStep(std::array<int64_t,4> position, std::array<int64_t,4>length,  double lambda) {
    if(mPartitionCode[mPartitionCodeIndex] == NOSPLITFLAG) {

        mPartitionCodeIndex++;
        mEntropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
        mEntropyCoder.mSubbandLF_ = mPartitionData_.copySubblock(length,position);
        std::array<int64_t,4> trueLength = {mEntropyCoder.mSubbandLF_.data.size(0), mEntropyCoder.mSubbandLF_.data.size(1), mEntropyCoder.mSubbandLF_.data.size(2), mEntropyCoder.mSubbandLF_.data.size(3)};
        std::cout<<"Encoding subblock at position: "<<position[0]<<","<<position[1]<<","<<position[2]<<","<<position[3]<<" with length: "<<trueLength[0]<<","<<trueLength[1]<<","<<trueLength[2]<<","<<trueLength[3]<<std::endl;
        std::cout<<"first few elements subblock: "<<mEntropyCoder.mSubbandLF_.data.index({at::indexing::Slice(0),at::indexing::Slice(0),0,at::indexing::Slice(0,3)})<<std::endl;

        if(trueLength[2] * trueLength[3] > 0) mEntropyCoder.encodeSubblockFromPool(trueLength, {0,0,0,0},mEntropyCoder.mSuperiorBitPlane,mLambda);
        return;
    }
    if(mPartitionCode[mPartitionCodeIndex] == INTRAVIEWSPLITFLAG) {
        
        mPartitionCodeIndex++;
        
        mEntropyCoder.EncodePartitionFlag(INTRAVIEWSPLITFLAGSYMBOL);
        
        std::array<int64_t,4> new_position, new_length;
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0];
        new_length[1] = length[1];
        new_length[2] = length[2]/2;
        new_length[3] = length[3]/2;
        
        //Encode four spatial subblocks 
        EncodePartitionStep(new_position, new_length, lambda);

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
        
        EncodePartitionStep(new_position, new_length, lambda);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        EncodePartitionStep(new_position, new_length, lambda);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        EncodePartitionStep(new_position, new_length, lambda);
        return;
    }
}


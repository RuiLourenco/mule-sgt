#include "Encoder/TransformPartition.h"
#include <chrono>


/*******************************************************************************/
/*                      TransformPartition class methods                    */
/*******************************************************************************/

TransformPartition :: TransformPartition(void) {
    mPartitionCode = NULL;
    //mUseSameBitPlane = 1;
}
TransformPartition :: ~TransformPartition(void) {
    if(mPartitionCode != NULL)
        delete [] mPartitionCode;
}
double TransformPartition :: totalTransformGain(std::array<int64_t,4> length){
    
    double transformGain = 1;
    for(int i = 0; i < 4; i++){
        transformGain*=length[i]/sqrt(length[i]);
        transformGain  *= sqrt(mPartitionData_.data.size(i)/length[i]);
    }
    return transformGain*mGain;

}
void TransformPartition :: RDoptimizeTransform_(Block4D_ &inputBlock, Hierarchical4DEncoder& entropyCoder,std::array<double,2>disparityRange,double transformGain, double lambda){
    mDisparityRange = disparityRange;
    mGain = transformGain;
    inputBlock.data = inputBlock.data.contiguous();
    if(!mSsiBuffer.empty()) mSsiBuffer.clear();
    mSsiBufferIndex = 0;
    if(mPartitionCode != NULL)
        delete [] mPartitionCode;
    mPartitionCode = new char [1];
    mPartitionCode[0] = 0;          //initializes the partition code string as the null string
    mEvaluateOptimumBitPlane = 1;
    std::array<int64_t,4> length = {inputBlock.data.size(0),inputBlock.data.size(1),inputBlock.data.size(2),inputBlock.data.size(3)};
    mPartitionData_ = Block4D_(length);
    double scaledLambda = length[0]*length[1]*length[2]*length[3]*lambda;


    std::array<int64_t,4> position = {0,0,0,0};
    entropyCoder.LoadOptimizerState();

    Block4D_ transformedBlock(length);
    transformedBlock.emptyTransform();

    mLagrangianCost = RDoptimizeTransformStep_(inputBlock, transformedBlock, position, length, entropyCoder, scaledLambda,mSsiBuffer, &mPartitionCode);
    //std::cout<<"optimized!"<<std::endl;
    this->costImage = mLagrangianCost*at::ones({length[0],length[1],length[2],length[3]},at::kDouble);
    //std::cout<<"mLagrangianCost = "<<mLagrangianCost<<std::endl;
        
    mPartitionData_ = transformedBlock;
    entropyCoder.LoadOptimizerState();
    printf(" Full PartitionCode = %s\n", mPartitionCode);    
    //printf("mInferiorBitPlane = %d\n", entropyCoder.mInferiorBitPlane);
    //std::cout<<"Full Number of Compressed Blocks"<<mSsiBuffer.size()<<std::endl;

}

double TransformPartition :: RDoptimizeTransformStep_(Block4D_ &inputBlock, Block4D_ &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length, Hierarchical4DEncoder &entropyCoder, double lambda, std::vector<SgtSideInfo>& currSsiBuffer,char **partitionCode) {
    


    //std::cout<<"Hello?"<<std::endl;
    //inputBlock never changes for the recursive calls. Instead block_0 is copied from a different position. 
    //I should eventually check if this needs to be a copy or if it can just be a reference but inputBlock could very well be a const & from my understanding.
    ProbabilityModel *currentCoderModelState;
    entropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //partitionCodeS handles splitting in the spatial dimension, partitionCodeV handles splitting in the view dimension.
    char *partitionCodeS=NULL, *partitionCodeV=NULL; 
    
    std::vector<SgtSideInfo> ssiBufferS, ssiBufferV;
    //Copy from inp
    Block4D_ block_0(length);
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"We're starting!"<<std::endl;

    block_0.CopySubblockFrom(inputBlock,position,{0,0,0,0});
   //std::cout<<"Block Copied!"<<std::endl;

    
    double currGain = totalTransformGain(length);
    //std::cout<<"3,2,1 Let's jam!"<<std::endl;

    block_0.sgtTransform(currGain,mDisparityRange);
    //std::cout<<block_0.data.sizes()<<std::endl;
    //std::cout<<"Param param param pararam1"<<std::endl;
    //std::cout<<"TRANSFORMED! "<<block_0.data.sizes()<<std::endl;
    SgtSideInfo ssi0 = block_0.ssi; 
    entropyCoder.mSubbandLF_ = block_0;
    //std::cout<<"Param param param pararam2"<<std::endl;

    //std::cout<<"HERE"<<std::endl;

    double Energy;
    double rate = 0;
    double distortion = 0;
    if(mEvaluateOptimumBitPlane == 1){
        entropyCoder.mInferiorBitPlane = entropyCoder.OptimumBitplaneFaster_(lambda);
        entropyCoder.LoadOptimizerState();
        mEvaluateOptimumBitPlane = 0;
        //std::cout<<"Minimum Bit Plane: "<<entropyCoder.mInferiorBitPlane<<std::endl;
    }
    


    if(entropyCoder.mSegmentationTreeCodeBuffer != NULL){
        delete [] entropyCoder.mSegmentationTreeCodeBuffer;
    }
    entropyCoder.mSegmentationTreeCodeBuffer = new char [2];
    strcpy(entropyCoder.mSegmentationTreeCodeBuffer,"");

    //std::cout<<"To Compress:"<<std::endl<<inputBlock.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;
    //std::cout<<entropyCoder.mSubbandLF_.data.sizes()<<std::endl;
    std::array<int64_t,4> lengthTransform = {entropyCoder.mSubbandLF_.data.size(0), entropyCoder.mSubbandLF_.data.size(1), entropyCoder.mSubbandLF_.data.size(2), entropyCoder.mSubbandLF_.data.size(3)};
    //std::cout<<"HERE "<<lengthTransform[0]<<" "<<lengthTransform[1]<<" "<<lengthTransform[2]<<" "<<lengthTransform[3]<<std::endl;
    double J0 = entropyCoder.RdOptimizeHexadecaTree_({0, 0, 0, 0}, lengthTransform, lambda, entropyCoder.mSuperiorBitPlane, &entropyCoder.mSegmentationTreeCodeBuffer, Energy,rate,distortion);
    //std::cout<<"HERE"<<std::endl;
    //std::cout<<"Param param param pararam 3"<<std::endl;

    //saves the resulting entropyCoder arithmetic model to model_0
    ProbabilityModel *coderModelState_0;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    //std::cout<<"HERE"<<std::endl;

    double JS = -1.0;
    Block4D_ transformedBlockS(length);
    transformedBlockS.emptyTransform();
    //std::cout<<"HERE"<<std::endl;

    //If you can split more in the spatial dimension
    if((length[3] >= 2*mlength_u_min)&&(length[2] >= 2*mlength_v_min)) {
        JS = 0.0;
        std::vector<SgtSideInfo> ssiBufferS00, ssiBufferS01, ssiBufferS10, ssiBufferS11;
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
       
        std::array<int64_t,4> new_position, new_length;
        //position remains the same
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        //length is halved in the spatial dimensions!
        new_length[0] = length[0];
        new_length[1] = length[1];
        new_length[2] = length[2]/2;
        new_length[3] = length[3]/2;
          
        //optimize partition for Block_S returning JS, the transformed Block_S, partitionCode_S and arithmetic_model_S
        Block4D_ transformedBlockS00(new_length);
        transformedBlockS00.emptyTransform();
        
        //Need to see what this is actually doing...
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS00, new_position, new_length, entropyCoder, lambda,ssiBufferS00, &partitionCodeS00);
        
        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2; //???? Why?
                 
        Block4D_ transformedBlockS01(new_length);
        transformedBlockS01.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS01, new_position, new_length, entropyCoder, lambda,ssiBufferS01, &partitionCodeS01);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        Block4D_ transformedBlockS11(new_length);
        transformedBlockS11.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS11, new_position, new_length,  entropyCoder, lambda,ssiBufferS10, &partitionCodeS11);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        Block4D_ transformedBlockS10(new_length);
        transformedBlockS10.emptyTransform();
        
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS10, new_position, new_length, entropyCoder, lambda, ssiBufferS11, &partitionCodeS10);
        //concatenates side info buffers
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS00.begin(), ssiBufferS00.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS01.begin(), ssiBufferS01.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS10.begin(), ssiBufferS10.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS11.begin(), ssiBufferS11.end());
        //std::cout<<ssiBufferS00.size()<<"+"<<ssiBufferS01.size()<<"+"<<ssiBufferS10.size()<<"+"<<ssiBufferS11.size()<<"="<<ssiBufferS.size()<<std::endl;
       
        //concatenate the partition codes doe the four blocks   
        
        
        partitionCodeS = new char [2+strlen(partitionCodeS00)+strlen(partitionCodeS01)+strlen(partitionCodeS10)+strlen(partitionCodeS11)];
        //std::cout<<"B00: data size"<<transformedBlockS00.data.sizes()<<" ";
        //std::cout<<"size:"<<transformedBlockS00.size[0]<<"x"<<transformedBlockS00.size[1]<<"x"<<transformedBlockS00.size[2]<<"x"<<transformedBlockS00.size[3]<<" ";
        //std::cout<<"transform size:"<<transformedBlockS00.transformSize[0]<<"x"<<transformedBlockS00.transformSize[1]<<"x"<<transformedBlockS00.transformSize[2]<<"x"<<transformedBlockS00.transformSize[3]<<std::endl;
        
        strcpy(partitionCodeS, partitionCodeS00);
        strcat(partitionCodeS, partitionCodeS01);
        strcat(partitionCodeS, partitionCodeS11);
        strcat(partitionCodeS, partitionCodeS10);
        transformedBlockS = Block4D_(transformedBlockS00,transformedBlockS01,transformedBlockS10,transformedBlockS11,false);
        //std::cout<<"Final Block: data size"<<transformedBlockS.data.sizes()<<" ";
        //std::cout<<"size:"<<transformedBlockS.size[0]<<"x"<<transformedBlockS.size[1]<<"x"<<transformedBlockS.size[2]<<"x"<<transformedBlockS.size[3]<<" ";
        //std::cout<<"transform size:"<<transformedBlockS.transformSize[0]<<"x"<<transformedBlockS.transformSize[1]<<"x"<<transformedBlockS.transformSize[2]<<"x"<<transformedBlockS.transformSize[3]<<std::endl<<std::endl;

        transformedBlockS.sgtDomain = true;
        //std::cout<<transformedBlockS.data.sizes()<<std::endl;

        // std::cout<<"Size: "<<length[2]<<"x"<<length[3]<<std::endl;
        // std::cout<<"First Coefficent 00:"<<transformedBlockS00.data[0][0][0][0].item()<<"  "<<transformedBlockS.data[0][0][0][0].item()<<std::endl;
        // std::cout<<"First Coefficent 01:"<<transformedBlockS01.data[0][0][0][0].item()<<"  "<<transformedBlockS.data[0][0][0][length[3]/2].item()<<std::endl;
        // std::cout<<"First Coefficent 10:"<<transformedBlockS10.data[0][0][0][0].item()<<"  "<<transformedBlockS.data[0][0][length[2]/2][0].item()<<std::endl;
        // std::cout<<"First Coefficent 11:"<<transformedBlockS11.data[0][0][0][0].item()<<"  "<<transformedBlockS.data[0][0][length[2]/2][length[3]/2].item()<<std::endl<<std::endl;
        
        
        
        delete [] partitionCodeS00;
        delete [] partitionCodeS01;
        delete [] partitionCodeS10;
        delete [] partitionCodeS11; 
    }
    //std::cout<<"HERE"<<std::endl;

    //std::cout<<"Reached the End of a recursive stream!"<<std::endl;
    //saves the resulting entropyCoder arithmetic model to model_s
    ProbabilityModel *coderModelState_s=NULL;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_s);
    
    //JV = cost of four quarter view subblocks
    //Restores the current arithmetic model using current_model. 
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    double JV = -1.0;
    Block4D_ transformedBlockV(length);
    transformedBlockV.emptyTransform();
    //std::cout<<"HERE"<<std::endl;

    if((length[0] >= 2*mlength_t_min)&&(length[1] >= 2*mlength_s_min)) {
        JV = 0.0;
        
        std::vector<SgtSideInfo> ssiBufferV00, ssiBufferV01, ssiBufferV10, ssiBufferV11;

        char *partitionCodeV00 = new char[1];
        char *partitionCodeV01 = new char[1];
        char *partitionCodeV10 = new char[1];
        char *partitionCodeV11 = new char[1];
        
        partitionCodeV00[0] = 0;
        partitionCodeV01[0] = 0;
        partitionCodeV10[0] = 0;
        partitionCodeV11[0] = 0;
       
        std::array<int64_t,4> new_position, new_length;
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0]/2;
        new_length[1] = length[1]/2;
        new_length[2] = length[2];
        new_length[3] = length[3];
        
        //optimize partition for Block_V returning JV, the transformed Block_V, partitionCode_S and arithmetic_model_S
        Block4D_ transformedBlockV00(new_length);
        
        JV += RDoptimizeTransformStep_(inputBlock, transformedBlockV00, new_position, new_length, entropyCoder, lambda,ssiBufferV00, &partitionCodeV00);

        new_position[1] = position[1] + length[1]/2;
        new_length[1] = length[1] - length[1]/2;
        
        Block4D_ transformedBlockV01(new_length);
        
        JV += RDoptimizeTransformStep_(inputBlock, transformedBlockV01, new_position, new_length, entropyCoder, lambda,ssiBufferV01, &partitionCodeV01);

        new_position[0] = position[0] + length[0]/2;
        new_length[0] = length[0] - length[0]/2;
        
        Block4D_ transformedBlockV11(new_length);
        
        JV += RDoptimizeTransformStep_(inputBlock, transformedBlockV11, new_position, new_length, entropyCoder, lambda,ssiBufferV10, &partitionCodeV11);
        
        new_position[1] = position[1];
        new_length[1] = length[1]/2;
        
        Block4D_ transformedBlockV10(new_length);
        
        JV += RDoptimizeTransformStep_(inputBlock, transformedBlockV10, new_position, new_length, entropyCoder, lambda,ssiBufferV11, &partitionCodeV10);
        
        //concatenates side info buffers
        ssiBufferV.insert(ssiBufferV.end(), ssiBufferV00.begin(), ssiBufferV00.end());
        ssiBufferV.insert(ssiBufferV.end(), ssiBufferV01.begin(), ssiBufferV01.end());
        ssiBufferV.insert(ssiBufferV.end(), ssiBufferV10.begin(), ssiBufferV10.end());
        ssiBufferV.insert(ssiBufferV.end(), ssiBufferV11.begin(), ssiBufferV11.end());        

        partitionCodeV = new char [2+strlen(partitionCodeV00)+strlen(partitionCodeV01)+strlen(partitionCodeV10)+strlen(partitionCodeV11)];
        strcpy(partitionCodeV, partitionCodeV00);
        strcat(partitionCodeV, partitionCodeV01);
        strcat(partitionCodeV, partitionCodeV11);
        strcat(partitionCodeV, partitionCodeV10);
        
        transformedBlockV = Block4D_(transformedBlockV00,transformedBlockV01,transformedBlockV11,transformedBlockV10,true);
       
        
        delete [] partitionCodeV00;
        delete [] partitionCodeV01;
        delete [] partitionCodeV10;
        delete [] partitionCodeV11;

        
   }
    //std::cout<<"HERE"<<std::endl;


        //saves the resulting entropyCoder arithmetic model to model_v
    ProbabilityModel *coderModelState_v=NULL;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_v);
    int RHO_PRECISION = ssi0.getRhoPrecision();
    int DISP_PRECISION = ssi0.getAnglePrecision();
    J0 += RHO_PRECISION*4*lambda + DISP_PRECISION*lambda;

    //std::cout<<"Estimated Flag Size = "<<RHO_PRECISION*4 + DISP_PRECISION<<std::endl;
    //std::cout<<"HERE"<<std::endl;


    if(J0 > 0) 
        J0 += 1.0*lambda;
    if(JV > 0)
        JV += 2.0*lambda;
    if(JS > 0)
        JS += 2.0*lambda;
    //std::cout<<"superiorBitPlane = "<<entropyCoder.mSuperiorBitPlane<<std::endl;
    
    
    
    //choose the lower cost and returns the corresponding cost,  the partition code and the arithmetic coder model
    //find best J
    int interview_split = 0;
    int intraview_split = 0;
    int no_split = 0;
    //std::cout<<length[2]<<"x"<<length[3]<<std::endl;
    //std::cout<<" J0 = "<<J0<<" JV = "<<JV<<" JS = "<<JS<<std::endl;
    if(JV >= 0) {
        if(JS >= 0) {
            if(JV < JS) {
                if(JV < J0) {
                    interview_split = 1;
                }
                else {
                    no_split = 1;                    
                }
            }
            else {                
                if(JS < J0) {
                    intraview_split = 1;
                }
                else {
                    no_split = 1;                    
                }
            }
        }
        else {
            if(JV < J0) {
                interview_split = 1;
            }
            else {
                no_split = 1;
            }            
        }
    }
    else {
        if(JS >= 0) {
            if(JS < J0) {
                intraview_split = 1;
            }
            else {
                no_split = 1;
            }
        }
        else {
            no_split = 1;
        }
    }
    double optimumJ=0;
    //std::cout<<"HERE"<<std::endl;

    if((interview_split + intraview_split + no_split) != 1) {
        printf("ERROR: partition fail/n");
        exit(0);
    }
    //std::cout<<"Param param param pararam final"<<std::endl;

    //reallocates memory for the partitionCode string based on the current length and the length of the chosen one
    //copies data from the chosen arithmetic coder model to the current model
    char flagCode[2];
    flagCode[1] = 0;
    if(interview_split == 1) {
        optimumJ = JV;
        char *code = new char[2+strlen(*partitionCode)+strlen(partitionCodeV)];
        strcpy(code, *partitionCode);
        flagCode[0] = INTERVIEWSPLITFLAG;
        strcat(code, flagCode);
        strcat(code, partitionCodeV);
        delete(*partitionCode);
        *partitionCode = code;
        entropyCoder.SetOptimizerProbabilisticModelState(coderModelState_v);
        transformedBlock.CopySubblockFrom(transformedBlockV,{0,0,0,0},{0,0,0,0});
        currSsiBuffer.insert(currSsiBuffer.end(),ssiBufferV.begin(), ssiBufferV.end());

        std::cout<<"Busios e Borboletas! Algo errado aconteceu! O algoritmo escolheu dividir as vistas!!!"<<std::endl;
    }
    if(intraview_split == 1) {
        optimumJ = JS;
        char *code = new char[2+strlen(*partitionCode)+strlen(partitionCodeS)];
        strcpy(code, *partitionCode);
        flagCode[0] = INTRAVIEWSPLITFLAG;
        strcat(code, flagCode);
        strcat(code, partitionCodeS);
        delete(*partitionCode);
        *partitionCode = code;
        entropyCoder.SetOptimizerProbabilisticModelState(coderModelState_s);
        transformedBlock.CopySubblockFrom(transformedBlockS, {0,0,0,0},{0,0,0,0});
        currSsiBuffer.insert(currSsiBuffer.end(),ssiBufferS.begin(), ssiBufferS.end());
        //std::cout<<"Split! BufferSize is now: "<<currSsiBuffer.size();

    }
    if(no_split == 1) {
        //std::cout<<"Param param param pararam 4"<<std::endl;

        optimumJ = J0;   
        char *code = new char[2+strlen(*partitionCode)];
        strcpy(code, *partitionCode);
        flagCode[0] = NOSPLITFLAG;
        strcat(code, flagCode);
        delete(*partitionCode);
        *partitionCode = code;
                 //std::cout<<"Param param param pararam 5"<<std::endl;

        entropyCoder.SetOptimizerProbabilisticModelState(coderModelState_0);
                 //std::cout<<block_0.data.sizes()<<std::endl;
                 //std::cout<<transformedBlock.data.sizes()<<std::endl;

        //mPartitionData.CopySubblockFrom(block_0, 0, 0, 0, 0, position[0], position[1], position[2], position[3]);
        transformedBlock.CopySubblockFrom(block_0, {0,0,0,0},{0,0,0,0});
             //std::cout<<"Param param param pararam 7"<<std::endl;

        //transformedBlock.orderH = block_0.getOrderH();
        //transformedBlock.orderV = block_0.getOrderV();
        currSsiBuffer.push_back(block_0.ssi);
                 //std::cout<<"Param param param pararam "<<std::endl;



        //std::cout<<"didn't split"<<std::endl;
    }
    //std::cout<<no_split<<" "<<intraview_split<<" "<<interview_split<<std::endl;
    //std::cout<<"currBufferSizeAtEnd = "<<currSsiBuffer.size()<<std::endl;
    //deletes temporary strings, blocks and models
    //block_0.SetDimension(0, 0, 0, 0);
   
    if(partitionCodeV != NULL) {
        delete [] partitionCodeV;
    }
    if(partitionCodeS != NULL) {
        delete [] partitionCodeS;
    }

    entropyCoder.DeleteProbabilisticModelState(currentCoderModelState);
    entropyCoder.DeleteProbabilisticModelState(coderModelState_0);
    entropyCoder.DeleteProbabilisticModelState(coderModelState_s);
    entropyCoder.DeleteProbabilisticModelState(coderModelState_v);

    //return optimum J
    //std::cout<<"ENDED OPTIMIZATION"<<std::endl;
        //std::cout<<"Param param param pararam final"<<std::endl;

    return(optimumJ);     
}

void TransformPartition :: EncodePartition_(Hierarchical4DEncoder&entropyCoder, double lambda){

    double scaledLambda = lambda;
    std::array<int64_t,4> length;
    for(int i = 0; i < 4; ++i){
        length[i] = mPartitionData_.data.size(i);
        scaledLambda*=length[i];
    }

    //std::cout<<"Partition Data Size: "<<mPartitionData_.size[0]<<" "<<mPartitionData_.size[1]<<" "<<mPartitionData_.size[2]<<" "<<mPartitionData_.size[3]<<std::endl;
    //std::cout<<"Partition Data Transform Size: "<<mPartitionData_.transformSize[0]<<" "<<mPartitionData_.transformSize[1]<<" "<<mPartitionData_.transformSize[2]<<" "<<mPartitionData_.transformSize[3]<<std::endl;
    this->costImage = torch::zeros(mPartitionData_.size, torch::kDouble);
    //std::cout<<"Cost Image Size: "<<costImage.size(0)<<" "<<costImage.size(1)<<" "<<costImage.size(2)<<" "<<costImage.size(3)<<std::endl;

    std::array<int64_t,4> position = {0,0,0,0};    
    mPartitionCodeIndex = 0;
      
    entropyCoder.EncodeInteger(entropyCoder.mInferiorBitPlane, MINIMUM_BITPLANE_PRECISION);
    //std::cout<<"Minimum Bit Plane: "<<entropyCoder.mInferiorBitPlane<<std::endl;


    EncodePartitionStep_(position, length, entropyCoder, scaledLambda);
}

void TransformPartition :: EncodePartitionStep_(std::array<int64_t,4> position, std::array<int64_t,4>length, Hierarchical4DEncoder &entropyCoder, double lambda) {
    if(mPartitionCode[mPartitionCodeIndex] == NOSPLITFLAG) {
      
        mPartitionCodeIndex++;
        
        entropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
        // std::cout<< "RhoS = "<<mSsiBuffer[mSsiBufferIndex].getRhoS()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoSCode()<<std::endl;
        // std::cout<< "RhoT = "<<mSsiBuffer[mSsiBufferIndex].getRhoT()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoTCode()<<std::endl;
        // std::cout<< "RhoU = "<<mSsiBuffer[mSsiBufferIndex].getRhoU()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoUCode()<<std::endl;
        // std::cout<< "RhoV = "<<mSsiBuffer[mSsiBufferIndex].getRhoV()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoVCode()<<std::endl;
        // std::cout<<"Disparity = "<<mSsiBuffer[mSsiBufferIndex].getDisparity()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getDCode()<<std::endl;
        
        
        entropyCoder.EncodeSSI_(mSsiBuffer[mSsiBufferIndex++]);
        std::array<int64_t,4> trueLength = {9,9,length[2]/9,length[3]/9};
        this->rhoSImage = mSsiBuffer[mSsiBufferIndex-1].getRhoS()*at::ones({trueLength[0],trueLength[1],trueLength[2],trueLength[3]},at::kDouble);
        this->rhoTImage = mSsiBuffer[mSsiBufferIndex-1].getRhoT()*at::ones({trueLength[0],trueLength[1],trueLength[2],trueLength[3]},at::kDouble);
        this->rhoUImage = mSsiBuffer[mSsiBufferIndex-1].getRhoU()*at::ones({trueLength[0],trueLength[1],trueLength[2],trueLength[3]},at::kDouble);
        this->rhoVImage = mSsiBuffer[mSsiBufferIndex-1].getRhoV()*at::ones({trueLength[0],trueLength[1],trueLength[2],trueLength[3]},at::kDouble);
        this->angleImageH = mSsiBuffer[mSsiBufferIndex-1].getDisparityH()*at::ones({trueLength[0],trueLength[1],trueLength[2],trueLength[3]},at::kDouble);
        this->angleImageV = mSsiBuffer[mSsiBufferIndex-1].getDisparityV()*at::ones({trueLength[0],trueLength[1],trueLength[2],trueLength[3]},at::kDouble);

        entropyCoder.mSubbandLF_ = Block4D_(length);
        entropyCoder.mSubbandLF_.CopySubblockFrom(mPartitionData_, position,{0,0,0,0});
        entropyCoder.mSubbandLF_.orderH = mPartitionData_.orderH;
        entropyCoder.mSubbandLF_.orderV = mPartitionData_.orderV;
        //std::cout<<"mSubbandLF_ Size: "<<entropyCoder.mSubbandLF_.data.size(2)<<"x"<<entropyCoder.mSubbandLF_.data.size(3)<<std::endl;
        //std::cout<<"mPartitionData_ Size: "<<mPartitionData_.data.size(2)<<"x"<<mPartitionData_.data.size(3)<<std::endl;
        //std::cout<<"Size: "<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"First Coefficent BEING Compressed:"<<entropyCoder.mSubbandLF_.data[0][0][0][0].item()<<std::endl;
        //std::cout<<"Is data contiguous?"<<entropyCoder.mSubbandLF_.data.is_contiguous()<<std::endl;
         int fCoeff = entropyCoder.mSubbandLF_.data[0][0][0][0].item<int>();


        //std::cout<<entropyCoder.mSubbandLF_.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

        // if (abs(fCoeff) < 1e5){
        //     std::cout<<"("<<position[2]<<","<<position[3]<<") "<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"Compressed:"<<std::endl<<entropyCoder.mSubbandLF_.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;
        
        //     std::cout<<"d = "<<mSsiBuffer[mSsiBufferIndex-1].getDisparity()<<std::endl;
        // }
        //std::cout<<"SGT Mean = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).mean().item()<<std::endl;
        //std::cout<<"SGT STD = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).std().item()<<std::endl;
        //std::cout<<"SGT Mean = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).mean({2,3},false,at::kDouble)<<std::endl;

        entropyCoder.EncodeSubblock_(lambda);
        this->rateImage = entropyCoder.mRate*at::ones({trueLength[0],trueLength[1],trueLength[2],trueLength[3]},at::kDouble);
        //std::cout<<"mRate = "<<entropyCoder.mRate<<std::endl;

        //std::cout<<"Position: = "<<position[0]<<","<<position[1]<<","<<position[2]/9<<","<<position[3]/9<<" Length = "<<length[0]+8<<","<<length[1]+8<<","<<length[2]/9<<","<<length[3]/9<<" "<<(entropyCoder.currCost/(double)size)<< std::endl;

        return;
    }
    if(mPartitionCode[mPartitionCodeIndex] == INTRAVIEWSPLITFLAG) {
        
        mPartitionCodeIndex++;
        
        entropyCoder.EncodePartitionFlag(INTRAVIEWSPLITFLAGSYMBOL);
        
        std::array<int64_t,4> new_position, new_length;;
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0];
        new_length[1] = length[1];
        new_length[2] = length[2]/2;
        new_length[3] = length[3]/2;
        
        //Encode four spatial subblocks 
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
        
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);
        return;
    }
        if(mPartitionCode[mPartitionCodeIndex] == INTERVIEWSPLITFLAG) {
        
        mPartitionCodeIndex++;
        
        entropyCoder.EncodePartitionFlag(INTERVIEWSPLITFLAGSYMBOL);
        std::cout<<"INTER VIEW PARTITION"<<std::endl;
        std::array<int64_t,4> new_position, new_length;;
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0]/2;
        new_length[1] = length[1]/2;
        new_length[2] = length[2];
        new_length[3] = length[3];
        
        //Encode four view subblocks 
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);
        //optimize partition for Block_V returning JV, the transformed Block_V, partitionCode_S and arithmetic_model_S

        new_position[1] = position[1] + length[1]/2;
        new_length[1] = length[1] - length[1]/2;
        
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);

        new_position[0] = position[0] + length[0]/2;
        new_length[0] = length[0] - length[0]/2;
        
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);
        
        new_position[1] = position[1];
        new_length[1] = length[1]/2;
        
        EncodePartitionStep_(new_position, new_length, entropyCoder, lambda);
        return;
    }

}


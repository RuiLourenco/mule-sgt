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
    //std::cout<<"SetUp Complete."<<std::endl;
    entropyCoder.LoadOptimizerState();
    //std::cout<<"EntropyCoder Loaded."<<std::endl;

    Block4D_ transformedBlock(length);
    //std::cout<<"Ready for First Step."<<std::endl;
    //std::cout<<mlength_u_min<<" "<<mlength_v_min<<" "<<mlength_s_min<<" "<<mlength_t_min<<std::endl;

    //std::cout<<"inputBlock var = "<<inputBlock.data.to(at::kDouble).var()<<std::endl;
    //std::cout<<"First value: "<<inputBlock.data[0][0][0][0]<<std::endl;
    mLagrangianCost = RDoptimizeTransformStep_(inputBlock, transformedBlock, position, length, entropyCoder, scaledLambda,mSsiBuffer, &mPartitionCode);
    mPartitionData_.CopySubblockFrom(transformedBlock,{0,0,0,0},{0,0,0,0});
    entropyCoder.LoadOptimizerState();
    printf(" Full PartitionCode = %s\n", mPartitionCode);    
    //printf("mInferiorBitPlane = %d\n", entropyCoder.mInferiorBitPlane);
    //std::cout<<"Full Number of Compressed Blocks"<<mSsiBuffer.size()<<std::endl;

}

void TransformPartition :: RDoptimizeTransform(Block4D &inputBlock, MultiscaleTransform &mt, Hierarchical4DEncoder &entropyCoder, double lambda) {
/*! Evaluates the Lagrangian cost of the optimum multiscale transform for the input block as well as the transformed block */   
    if(mPartitionCode != NULL)
        delete [] mPartitionCode;
    mPartitionCode = new char [1];
    mPartitionCode[0] = 0;          //initializes the partition code string as the null string
    mEvaluateOptimumBitPlane = 1;
    mPartitionData.SetDimension(inputBlock.mlength_t, inputBlock.mlength_s, inputBlock.mlength_v, inputBlock.mlength_u);
   
    double scaledLambda = mPartitionData.mlength_t*mPartitionData.mlength_s;
    scaledLambda *= lambda*mPartitionData.mlength_v*mPartitionData.mlength_u;
    
    int position[4];
    position[0] = 0;
    position[1] = 0;
    position[2] = 0;
    position[3] = 0;

    int length[4];
    length[0] = inputBlock.mlength_t;
    length[1] = inputBlock.mlength_s;
    length[2] = inputBlock.mlength_v;
    length[3] = inputBlock.mlength_u;
   
    //copies the current entropyCoder arithmetic model to the optimizer model. 
    entropyCoder.LoadOptimizerState();
    
    Block4D transformedBlock;
    transformedBlock.SetDimension(length[0], length[1], length[2], length[3]);
        
    mLagrangianCost = RDoptimizeTransformStep(inputBlock, transformedBlock, position, length, mt, entropyCoder, scaledLambda, &mPartitionCode);
    
    mPartitionData.CopySubblockFrom(transformedBlock, 0, 0, 0, 0);
    
    //Restores state since the encoder will reevaluate it    
    entropyCoder.LoadOptimizerState();
    
    //printf("Finale PartitionCode = %s\n", mPartitionCode);    
    //printf("mInferiorBitPlane = %d\n", entropyCoder.mInferiorBitPlane);    
}

double TransformPartition :: RDoptimizeTransformStep_(Block4D_ &inputBlock, Block4D_ &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length, Hierarchical4DEncoder &entropyCoder, double lambda, std::vector<SgtSideInfo>& currSsiBuffer,char **partitionCode) {
    
    std::chrono::time_point<std::chrono::steady_clock> starter;
    std::chrono::time_point<std::chrono::steady_clock> ender;
    //std::cout<<"IsContinuousOutside = "<<inputBlock.data.is_contiguous()<<std::endl;
    //std::cout<<"Stride = "<<inputBlock.data.strides()<<std::endl;
    //std::cout<<"Step with size: "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
    
    //inputBlock never changes for the recursive calls. Instead block_0 is copied from a different position. 
    //I should eventually check if this needs to be a copy or if it can just be a reference but inputBlock could very well be a const & from my understanding.
    ProbabilityModel *currentCoderModelState;
    entropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Guess: partitionCodeS handles splitting in the view dimension, partitionCodeV handles splitting in the spacial dimension.
    char *partitionCodeS=NULL, *partitionCodeV=NULL; 
    
    std::vector<SgtSideInfo> ssiBufferS, ssiBufferV;
    //Copy from inp
    Block4D_ block_0(length);

    block_0.CopySubblockFrom(inputBlock,position,{0,0,0,0});
        //std::cout<<"IsContinuousOutside = "<<block_0.data.is_contiguous()<<std::endl;

    //std::cout<<block_0.data.strides()<<" "<<block_0.data.sizes()<<std::endl;
    //std::cout<<"position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
    //std::cout<<"inputBlock var = "<<inputBlock.data.to(at::kDouble).var()<<std::endl;
    //std::cout<<"block_0 var = "<<block_0.data.to(at::kDouble).var()<<std::endl;
    //std::cout<<block_0.data.sizes()<<std::endl;
    
    double currGain = totalTransformGain(length);
    // if(position[0]==0 && position[1]==0 && position[2]==48 && position[3]==0){
    //     // if(length[2] == 16 && length[3] == 16){
    //     //     std::cout<<"Gain = "<<currGain<<std::endl;
    //     //     std::cout<<"First Coefficient: "<<block_0.data[0][0][0][0].item<int>()/currGain<<std::endl;
    //     //     std::cout<<"Second Coefficient: "<<block_0.data[0][0][0][1].item<int>()/currGain<<std::endl;
    //     // }
    // }
    //double currGain = 1;
    //std::cout<<"gain ="<<currGain<<std::endl;
    //starter = std::chrono::steady_clock::now();
    //std::cout<<"Y:"<<std::endl;
    //std::cout<<block_0.data.mean({2,3},false,at::kDouble)<<std::endl<<std::endl;
    block_0.sgtTransform(currGain,mDisparityRange);
    //std::cout<<mDisparityRange[0]<<" "<<mDisparityRange[1]<<std::endl;
    //std::cout<<"Before:"<<std::endl;
    //std::cout<<block_0.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

    //std::cout<<"Gain:"<<currGain<<std::endl;
    //std::cout<<"First Coefficient: "<<block_0.data[0][0][0][0].item()<<std::endl;
    //ender = std::chrono::steady_clock::now(); 
    // if(position[0]==0 && position[1]==0 && position[2]==48 && position[3]==0){
    //     if(length[2] == 16 && length[3] == 16){
    //         std::cout<<"Gain = "<<currGain<<std::endl;
    //         std::cout<<"First Coefficient: "<<block_0.data[0][0][0][0].item<int>()/currGain<<std::endl;
    //         std::cout<<"Second Coefficient: "<<block_0.data[0][0][0][1].item<int>()/currGain<<std::endl;
    //     }
    // }
    //std::cout<<"Transform Time = "<<std::chrono::duration_cast<std::chrono::nanoseconds>(ender - starter).count()/1e6<<"ms"<<std::endl;
    
    //std::cout<<"Transformed the block!"<<std::endl;
    SgtSideInfo ssi0 = block_0.ssi; 
    starter = std::chrono::steady_clock::now();
    entropyCoder.mSubbandLF_ = block_0;
    double Energy;
    if(mEvaluateOptimumBitPlane == 1){
        //std::cout<<"PLEASE BE CONTIGUOUS! "<<entropyCoder.mSubbandLF_.data.is_contiguous()<<std::endl;
        entropyCoder.mInferiorBitPlane = entropyCoder.OptimumBitplaneFaster_(lambda);
        entropyCoder.LoadOptimizerState();
        mEvaluateOptimumBitPlane = 0;
        //std::cout<<"Optimum Bit Plane: "<<entropyCoder.mInferiorBitPlane<<std::endl;
        //std::cout<<"OPTIMUM BIT PLANE OLD: "<<entropyCoder.OptimumBitplane_(lambda)<<std::endl;


    }



    if(entropyCoder.mSegmentationTreeCodeBuffer != NULL){
        delete [] entropyCoder.mSegmentationTreeCodeBuffer;
    }
    entropyCoder.mSegmentationTreeCodeBuffer = new char [2];
    strcpy(entropyCoder.mSegmentationTreeCodeBuffer,"");

    ender = std::chrono::steady_clock::now(); 
    //std::cout<<"Encoder Setup Time = "<<std::chrono::duration_cast<std::chrono::nanoseconds>(ender - starter).count()/1e6<<"ms"<<std::endl;
   
    
    starter = std::chrono::steady_clock::now();
    double J0 = entropyCoder.RdOptimizeHexadecaTree_({0, 0, 0, 0}, length, lambda, entropyCoder.mSuperiorBitPlane, &entropyCoder.mSegmentationTreeCodeBuffer, Energy);
    ender = std::chrono::steady_clock::now(); 
    //std::cout<<"Optimization Time = "<<std::chrono::duration_cast<std::chrono::nanoseconds>(ender - starter).count()/1e6<<"ms"<<std::endl;
   
    //saves the resulting entropyCoder arithmetic model to model_0
    ProbabilityModel *coderModelState_0;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    double JS = -1.0;
    Block4D_ transformedBlockS(length);
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
        
        //Need to see what this is actually doing...
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS00, new_position, new_length, entropyCoder, lambda,ssiBufferS00, &partitionCodeS00);
        
        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2; //???? Why?
                 
        Block4D_ transformedBlockS01(new_length);
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS01, new_position, new_length, entropyCoder, lambda,ssiBufferS01, &partitionCodeS01);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        Block4D_ transformedBlockS11(new_length);
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS11, new_position, new_length,  entropyCoder, lambda,ssiBufferS10, &partitionCodeS11);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        Block4D_ transformedBlockS10(new_length);
        
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS10, new_position, new_length, entropyCoder, lambda, ssiBufferS11, &partitionCodeS10);
        //concatenates side info buffers
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS00.begin(), ssiBufferS00.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS01.begin(), ssiBufferS01.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS10.begin(), ssiBufferS10.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS11.begin(), ssiBufferS11.end());
        //std::cout<<ssiBufferS00.size()<<"+"<<ssiBufferS01.size()<<"+"<<ssiBufferS10.size()<<"+"<<ssiBufferS11.size()<<"="<<ssiBufferS.size()<<std::endl;
       
        //concatenate the partition codes doe the four blocks   
        
        
        partitionCodeS = new char [2+strlen(partitionCodeS00)+strlen(partitionCodeS01)+strlen(partitionCodeS10)+strlen(partitionCodeS11)];
        
        
        strcpy(partitionCodeS, partitionCodeS00);
        strcat(partitionCodeS, partitionCodeS01);
        strcat(partitionCodeS, partitionCodeS11);
        strcat(partitionCodeS, partitionCodeS10);
        transformedBlockS = Block4D_(transformedBlockS00,transformedBlockS01,transformedBlockS10,transformedBlockS11,false);
        transformedBlockS.sgtDomain = true;

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
    //std::cout<<"Reached the End of a recursive stream!"<<std::endl;
    //saves the resulting entropyCoder arithmetic model to model_s
    ProbabilityModel *coderModelState_s=NULL;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_s);
    
    //JV = cost of four quarter view subblocks
    //Restores the current arithmetic model using current_model. 
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    double JV = -1.0;
    Block4D_ transformedBlockV(length);
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

        //saves the resulting entropyCoder arithmetic model to model_v
    ProbabilityModel *coderModelState_v=NULL;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_v);
    int RHO_PRECISION = ssi0.getRhoPrecision();
    int DISP_PRECISION = ssi0.getDisparityPrecision();
    J0 += RHO_PRECISION*4*lambda + DISP_PRECISION*lambda;

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
    if((interview_split + intraview_split + no_split) != 1) {
        printf("ERROR: partition fail/n");
        exit(0);
    }
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
        optimumJ = J0;   
        char *code = new char[2+strlen(*partitionCode)];
        strcpy(code, *partitionCode);
        flagCode[0] = NOSPLITFLAG;
        strcat(code, flagCode);
        delete(*partitionCode);
        *partitionCode = code;
        entropyCoder.SetOptimizerProbabilisticModelState(coderModelState_0);
        //mPartitionData.CopySubblockFrom(block_0, 0, 0, 0, 0, position[0], position[1], position[2], position[3]);
        transformedBlock.CopySubblockFrom(block_0, {0,0,0,0},{0,0,0,0});
        currSsiBuffer.push_back(block_0.ssi);
       

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
    return(optimumJ);     
}

double TransformPartition :: RDoptimizeTransformStep(Block4D &inputBlock, Block4D &transformedBlock, int *position, int *length,  MultiscaleTransform &mt,Hierarchical4DEncoder &entropyCoder, double lambda, char **partitionCode) {
/*! returns the Lagrangian cost of one step of the optimization of the multiscale transform for the input block as well as the transformed block */   
 
    //J0 = cost of full transform
    //saves the current entropyCoder arithmetic model to current_model. 
    ProbabilityModel *currentCoderModelState;
    entropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    char *partitionCodeS=NULL, *partitionCodeV=NULL; 
    
    //copy the inputBlock to block_0 and apply transformation using the appropriate scale from mt    
    Block4D block_0;
    block_0.SetDimension(length[0], length[1], length[2], length[3]);
    block_0.CopySubblockFrom(inputBlock, position[0], position[1], position[2], position[3]);
    mt.Transform4D(block_0);
    //std::cout<<"First Coefficient: "<<block_0.mPixel[0][0][0][0]<<std::endl;
    
   
    //copy the transformed input block to entropyCoder.mSubbandLF    
    entropyCoder.mSubbandLF.SetDimension(block_0.mlength_t, block_0.mlength_s, block_0.mlength_v, block_0.mlength_u);
    entropyCoder.mSubbandLF.CopySubblockFrom(block_0, 0, 0, 0, 0);
    double Energy;
    if(mEvaluateOptimumBitPlane == 1) {
        entropyCoder.mInferiorBitPlane = entropyCoder.OptimumBitplane(lambda);
        entropyCoder.LoadOptimizerState();
        mEvaluateOptimumBitPlane = 0;
        //std::cout<<"Optimum Bit Plane: "<<entropyCoder.mInferiorBitPlane<<std::endl;
    }
    
    //call RdOptimizeHexadecaTree method from entropyCoder to evaluate J0
    if(entropyCoder.mSegmentationTreeCodeBuffer != NULL)
        delete [] entropyCoder.mSegmentationTreeCodeBuffer;
    entropyCoder.mSegmentationTreeCodeBuffer = new char [2];
    strcpy(entropyCoder.mSegmentationTreeCodeBuffer,"");
    double J0 = entropyCoder.RdOptimizeHexadecaTree(0, 0, 0, 0, entropyCoder.mSubbandLF.mlength_t, entropyCoder.mSubbandLF.mlength_s, entropyCoder.mSubbandLF.mlength_v, entropyCoder.mSubbandLF.mlength_u, lambda, entropyCoder.mSuperiorBitPlane, &entropyCoder.mSegmentationTreeCodeBuffer, Energy);

    //saves the resulting entropyCoder arithmetic model to model_0
    ProbabilityModel *coderModelState_0;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    
    //JS = cost of four quarter spatial subblocks
    //Restores the current arithmetic model using current_model. 
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    double JS = -1.0;
    Block4D transformedBlockS;
    transformedBlockS.SetDimension(length[0], length[1], length[2], length[3]);
    
    if((length[3] >= 2*mlength_u_min)&&(length[2] >= 2*mlength_v_min)) {
        JS = 0.0;
        
        
        char *partitionCodeS00 = new char[1];
        char *partitionCodeS01 = new char[1];
        char *partitionCodeS10 = new char[1];
        char *partitionCodeS11 = new char[1];
        
        partitionCodeS00[0] = 0;
        partitionCodeS01[0] = 0;
        partitionCodeS10[0] = 0;
        partitionCodeS11[0] = 0;
       
        int new_position[4], new_length[4];
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0];
        new_length[1] = length[1];
        new_length[2] = length[2]/2;
        new_length[3] = length[3]/2;
          
        //optimize partition for Block_S returning JS, the transformed Block_S, partitionCode_S and arithmetic_model_S
        Block4D transformedBlockS00;
        transformedBlockS00.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS00, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeS00);

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
                 
        Block4D transformedBlockS01;
        transformedBlockS01.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS01, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeS01);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        Block4D transformedBlockS11;
        transformedBlockS11.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS11, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeS11);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        Block4D transformedBlockS10;
        transformedBlockS10.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JS += RDoptimizeTransformStep(inputBlock, transformedBlockS10, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeS10);
              
        partitionCodeS = new char [2+strlen(partitionCodeS00)+strlen(partitionCodeS01)+strlen(partitionCodeS10)+strlen(partitionCodeS11)];
        strcpy(partitionCodeS, partitionCodeS00);
        strcat(partitionCodeS, partitionCodeS01);
        strcat(partitionCodeS, partitionCodeS11);
        strcat(partitionCodeS, partitionCodeS10);
        
        transformedBlockS.CopySubblockFrom(transformedBlockS00, 0, 0, 0, 0);
        transformedBlockS.CopySubblockFrom(transformedBlockS01, 0, 0, 0, 0, 0, 0, 0, length[3]/2);
        transformedBlockS.CopySubblockFrom(transformedBlockS11, 0, 0, 0, 0, 0, 0, length[2]/2, length[3]/2);
        transformedBlockS.CopySubblockFrom(transformedBlockS10, 0, 0, 0, 0, 0, 0, length[2]/2, 0);
        
        delete [] partitionCodeS00;
        delete [] partitionCodeS01;
        delete [] partitionCodeS10;
        delete [] partitionCodeS11;
        
        transformedBlockS00.SetDimension(0, 0, 0, 0);
        transformedBlockS01.SetDimension(0, 0, 0, 0);
        transformedBlockS11.SetDimension(0, 0, 0, 0);
        transformedBlockS10.SetDimension(0, 0, 0, 0);
        
     }
    //saves the resulting entropyCoder arithmetic model to model_s
    ProbabilityModel *coderModelState_s=NULL;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_s);
    
    //JV = cost of four quarter view subblocks
    //Restores the current arithmetic model using current_model. 
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    double JV = -1.0;
    Block4D transformedBlockV;
    transformedBlockV.SetDimension(length[0], length[1], length[2], length[3]);
    if((length[0] >= 2*mlength_t_min)&&(length[1] >= 2*mlength_s_min)) {
        JV = 0.0;
        
        
        char *partitionCodeV00 = new char[1];
        char *partitionCodeV01 = new char[1];
        char *partitionCodeV10 = new char[1];
        char *partitionCodeV11 = new char[1];
        
        partitionCodeV00[0] = 0;
        partitionCodeV01[0] = 0;
        partitionCodeV10[0] = 0;
        partitionCodeV11[0] = 0;
       
        int new_position[4], new_length[4];
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0]/2;
        new_length[1] = length[1]/2;
        new_length[2] = length[2];
        new_length[3] = length[3];
        
        //optimize partition for Block_V returning JV, the transformed Block_V, partitionCode_S and arithmetic_model_S
        Block4D transformedBlockV00;
        transformedBlockV00.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JV += RDoptimizeTransformStep(inputBlock, transformedBlockV00, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeV00);

        new_position[1] = position[1] + length[1]/2;
        new_length[1] = length[1] - length[1]/2;
        
        Block4D transformedBlockV01;
        transformedBlockV01.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JV += RDoptimizeTransformStep(inputBlock, transformedBlockV01, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeV01);

        new_position[0] = position[0] + length[0]/2;
        new_length[0] = length[0] - length[0]/2;
        
        Block4D transformedBlockV11;
        transformedBlockV11.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JV += RDoptimizeTransformStep(inputBlock, transformedBlockV11, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeV11);
        
        new_position[1] = position[1];
        new_length[1] = length[1]/2;
        
        Block4D transformedBlockV10;
        transformedBlockV10.SetDimension(new_length[0], new_length[1], new_length[2], new_length[3]);
        
        JV += RDoptimizeTransformStep(inputBlock, transformedBlockV10, new_position, new_length, mt, entropyCoder, lambda, &partitionCodeV10);
        
        partitionCodeV = new char [2+strlen(partitionCodeV00)+strlen(partitionCodeV01)+strlen(partitionCodeV10)+strlen(partitionCodeV11)];
        strcpy(partitionCodeV, partitionCodeV00);
        strcat(partitionCodeV, partitionCodeV01);
        strcat(partitionCodeV, partitionCodeV11);
        strcat(partitionCodeV, partitionCodeV10);
        
        transformedBlockV.CopySubblockFrom(transformedBlockV00, 0, 0, 0, 0);
        transformedBlockV.CopySubblockFrom(transformedBlockV01, 0, 0, 0, 0, 0, length[1]/2, 0, 0);
        transformedBlockV.CopySubblockFrom(transformedBlockV11, 0, 0, 0, 0, length[0]/2, length[1]/2, 0, 0);
        transformedBlockV.CopySubblockFrom(transformedBlockV10, 0, 0, 0, 0, length[0]/2, 0, 0, 0);
        
        delete [] partitionCodeV00;
        delete [] partitionCodeV01;
        delete [] partitionCodeV10;
        delete [] partitionCodeV11;
        
        transformedBlockV00.SetDimension(0, 0, 0, 0);
        transformedBlockV01.SetDimension(0, 0, 0, 0);
        transformedBlockV11.SetDimension(0, 0, 0, 0);
        transformedBlockV10.SetDimension(0, 0, 0, 0);
        
   }

    //saves the resulting entropyCoder arithmetic model to model_v
    ProbabilityModel *coderModelState_v=NULL;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_v);

    if(J0 > 0) 
        J0 += 1.0*lambda;
    if(JV > 0)
        JV += 2.0*lambda;
    if(JS > 0)
        JS += 2.0*lambda;
    
    //choose the lower cost and returns the corresponding cost,  the partition code and the arithmetic coder model
    //find best J
    int interview_split = 0;
    int intraview_split = 0;
    int no_split = 0;
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
    if((interview_split + intraview_split + no_split) != 1) {
        printf("ERRO: partition fail/n");
        exit(0);
    }
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
        transformedBlock.CopySubblockFrom(transformedBlockV, 0, 0, 0, 0);
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
        transformedBlock.CopySubblockFrom(transformedBlockS, 0, 0, 0, 0);
    }
    if(no_split == 1) {
        optimumJ = J0;   
        char *code = new char[2+strlen(*partitionCode)];
        strcpy(code, *partitionCode);
        flagCode[0] = NOSPLITFLAG;
        strcat(code, flagCode);
        delete(*partitionCode);
        *partitionCode = code;
        entropyCoder.SetOptimizerProbabilisticModelState(coderModelState_0);
        //mPartitionData.CopySubblockFrom(block_0, 0, 0, 0, 0, position[0], position[1], position[2], position[3]);
        transformedBlock.CopySubblockFrom(block_0, 0, 0, 0, 0);
        
    }
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
    
    block_0.SetDimension(0,0,0,0);
    transformedBlockS.SetDimension(0,0,0,0);
    transformedBlockV.SetDimension(0,0,0,0);

    //return optimum J
    return(optimumJ);
    
}
void TransformPartition :: EncodePartition_(Hierarchical4DEncoder&entropyCoder, double lambda){
    
    double scaledLambda = lambda;
    std::array<int64_t,4> length;
    for(int i = 0; i < 4; ++i){
        length[i] = mPartitionData_.data.size(i);
        scaledLambda*=length[i];
    }
    std::array<int64_t,4> position = {0,0,0,0};    
    mPartitionCodeIndex = 0;
      
    entropyCoder.EncodeInteger(entropyCoder.mInferiorBitPlane, MINIMUM_BITPLANE_PRECISION);

    EncodePartitionStep_(position, length, entropyCoder, scaledLambda);
}
void TransformPartition :: EncodePartition(Hierarchical4DEncoder &entropyCoder, double lambda) {
    
    double scaledLambda = mPartitionData.mlength_t*mPartitionData.mlength_s;
    scaledLambda *= lambda*mPartitionData.mlength_v*mPartitionData.mlength_u;
    
    mPartitionCodeIndex = 0;
    
    int position[4];
    position[0] = 0;
    position[1] = 0;
    position[2] = 0;
    position[3] = 0;

    int length[4];
    length[0] = mPartitionData.mlength_t;
    length[1] = mPartitionData.mlength_s;
    length[2] = mPartitionData.mlength_v;
    length[3] = mPartitionData.mlength_u;
        
    entropyCoder.EncodeInteger(entropyCoder.mInferiorBitPlane, MINIMUM_BITPLANE_PRECISION);

    EncodePartitionStep(position, length, entropyCoder, scaledLambda);
        
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

        entropyCoder.mSubbandLF_ = Block4D_(length);
        entropyCoder.mSubbandLF_.CopySubblockFrom(mPartitionData_, position,{0,0,0,0});
        //std::cout<<"Size: "<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"First Coefficent BEING Compressed:"<<entropyCoder.mSubbandLF_.data[0][0][0][0].item()<<std::endl;
        //std::cout<<"Is data contiguous?"<<entropyCoder.mSubbandLF_.data.is_contiguous()<<std::endl;
         int fCoeff = entropyCoder.mSubbandLF_.data[0][0][0][0].item<int>();
        //std::cout<<entropyCoder.mSubbandLF_.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

        // if (abs(fCoeff) < 1e5){
        //     std::cout<<"("<<position[2]<<","<<position[3]<<") "<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"First Coefficent:"<<fCoeff<<" Second (?) Coefficient "<<entropyCoder.mSubbandLF_.data[0][0][0][1].item<int>()<<std::endl;
        //     std::cout<<"d = "<<mSsiBuffer[mSsiBufferIndex-1].getDisparity()<<std::endl;
        // }
        //std::cout<<"SGT Mean = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).mean().item()<<std::endl;
        //std::cout<<"SGT STD = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).std().item()<<std::endl;
        //std::cout<<"SGT Mean = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).mean({2,3},false,at::kDouble)<<std::endl;

        entropyCoder.EncodeSubblock_(lambda);
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

void TransformPartition :: EncodePartitionStep(int *position, int *length, Hierarchical4DEncoder &entropyCoder, double lambda) {
    
    if(mPartitionCode[mPartitionCodeIndex] == NOSPLITFLAG) {
      
        mPartitionCodeIndex++;
                
        entropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
        
        entropyCoder.mSubbandLF.SetDimension(length[0], length[1], length[2], length[3]);
        entropyCoder.mSubbandLF.CopySubblockFrom(mPartitionData, position[0], position[1], position[2], position[3]);
        entropyCoder.EncodeSubblock(lambda);
        
        return;
    }
    if(mPartitionCode[mPartitionCodeIndex] == INTRAVIEWSPLITFLAG) {
        
        mPartitionCodeIndex++;
        
        entropyCoder.EncodePartitionFlag(INTRAVIEWSPLITFLAGSYMBOL);
        
        int new_position[4], new_length[4];
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0];
        new_length[1] = length[1];
        new_length[2] = length[2]/2;
        new_length[3] = length[3]/2;
        
        //Encode four spatial subblocks 
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
        
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);
        return;
    }
    if(mPartitionCode[mPartitionCodeIndex] == INTERVIEWSPLITFLAG) {
        
        mPartitionCodeIndex++;
        
        entropyCoder.EncodePartitionFlag(INTERVIEWSPLITFLAGSYMBOL);
        
        int new_position[4], new_length[4];
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0]/2;
        new_length[1] = length[1]/2;
        new_length[2] = length[2];
        new_length[3] = length[3];
        
        //Encode four view subblocks 
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);
        //optimize partition for Block_V returning JV, the transformed Block_V, partitionCode_S and arithmetic_model_S

        new_position[1] = position[1] + length[1]/2;
        new_length[1] = length[1] - length[1]/2;
        
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);

        new_position[0] = position[0] + length[0]/2;
        new_length[0] = length[0] - length[0]/2;
        
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);
        
        new_position[1] = position[1];
        new_length[1] = length[1]/2;
        
        EncodePartitionStep(new_position, new_length, entropyCoder, lambda);
        return;
    }
}


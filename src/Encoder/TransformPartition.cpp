#include "TransformPartition.h"

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
    
    printf("mPartitionCode = %s\n", mPartitionCode);    
    printf("mInferiorBitPlane = %d\n", entropyCoder.mInferiorBitPlane);    
}

double TransformPartition :: RDoptimizeTransformStep(Block4D &inputBlock, Block4D &transformedBlock, int *position, int *length, MultiscaleTransform &mt, Hierarchical4DEncoder &entropyCoder, double lambda, char **partitionCode) {
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
   
    //copy the transformed input block to entropyCoder.mSubbandLF    
    entropyCoder.mSubbandLF.SetDimension(block_0.mlength_t, block_0.mlength_s, block_0.mlength_v, block_0.mlength_u);
    entropyCoder.mSubbandLF.CopySubblockFrom(block_0, 0, 0, 0, 0);
    double Energy;
    if(mEvaluateOptimumBitPlane == 1) {
        entropyCoder.mInferiorBitPlane = entropyCoder.OptimumBitplane(lambda);
        entropyCoder.LoadOptimizerState();
        mEvaluateOptimumBitPlane = 0;
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


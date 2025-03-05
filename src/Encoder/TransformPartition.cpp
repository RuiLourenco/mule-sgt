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
        transformGain  *= sqrt(mPartitionData_.size[i]/length[i]);
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
    mLambda = scaledLambda;

    std::array<int64_t,4> position = {0,0,0,0};
    entropyCoder.LoadOptimizerState();

    Block4D_ transformedBlock(length);
    transformedBlock.emptyTransform();

    mLagrangianCost = RDoptimizeTransformStep_(inputBlock, transformedBlock, position, length, entropyCoder, mSsiBuffer, &mPartitionCode);
    //std::cout<<"optimized!"<<std::endl;
    this->costImage = mLagrangianCost*at::ones({length[0],length[1],length[2],length[3]},at::kDouble);
    //std::cout<<"mLagrangianCost = "<<mLagrangianCost<<std::endl;
        
    mPartitionData_ = transformedBlock;
    entropyCoder.LoadOptimizerState();
    printf(" Full PartitionCode = %s\n", mPartitionCode);    
    //printf("mInferiorBitPlane = %d\n", entropyCoder.mInferiorBitPlane);
    //std::cout<<"Full Number of Compressed Blocks"<<mSsiBuffer.size()<<std::endl;

}

double TransformPartition :: EvaluatePartition_(Block4D_ &block_0, Hierarchical4DEncoder &entropyCoder, double currGain , double angle){
    //partitionCodeS handles splitting in the spatial dimension, partitionCodeV handles splitting in the view dimension.
    char *partitionCodeS=NULL;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    //begin = std::chrono::steady_clock::now();
    block_0.ssi = SgtSideInfo(angle,angle,mDisparityRange);
    block_0.ssi.estimateRhos(block_0,3000);
    //end = std::chrono::steady_clock::now();
    //std::cout << "Side Info Calc = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    //begin  = std::chrono::steady_clock::now();
    block_0.sgtTransform(currGain);
    //end = std::chrono::steady_clock::now();
      //  std::cout << "Transform Calc = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    //std::cout<<": ";
    
    //begin  = std::chrono::steady_clock::now();
    SgtSideInfo ssi0 = block_0.ssi; 
    entropyCoder.mSubbandLF_ = block_0;
    //end = std::chrono::steady_clock::now();
    //std::cout << "Copy Block Compute = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;


    double Energy;
    double rate = 0;
    double distortion = 0;


    if(mEvaluateOptimumBitPlane == 1){
        entropyCoder.mInferiorBitPlane = entropyCoder.OptimumBitplaneFaster_(mLambda);
        entropyCoder.LoadOptimizerState();
        mEvaluateOptimumBitPlane = 0;
        //std::cout<<"MBP : "<<entropyCoder.mInferiorBitPlane<<std::endl;
    }
    if(entropyCoder.mSegmentationTreeCodeBuffer != NULL){
        delete [] entropyCoder.mSegmentationTreeCodeBuffer;
    }
    entropyCoder.mSegmentationTreeCodeBuffer = new char [2];
    strcpy(entropyCoder.mSegmentationTreeCodeBuffer,"");
    //begin  = std::chrono::steady_clock::now();
    std::array<int64_t,4> lengthTransform = {entropyCoder.mSubbandLF_.data.size(0), entropyCoder.mSubbandLF_.data.size(1), entropyCoder.mSubbandLF_.data.size(2), entropyCoder.mSubbandLF_.data.size(3)};
    //std::cout<<entropyCoder.mSuperiorBitPlane<<std::endl;
    double J0 = entropyCoder.RdOptimizeHexadecaTree_({0, 0, 0, 0}, lengthTransform, mLambda,entropyCoder.mSuperiorBitPlane, &entropyCoder.mSegmentationTreeCodeBuffer, Energy,rate,distortion);
    int RHO_PRECISION = ssi0.getRhoPrecision();
    int DISP_PRECISION = ssi0.getAnglePrecision();
    J0 += RHO_PRECISION*4*mLambda + DISP_PRECISION*mLambda;
    //end = std::chrono::steady_clock::now();
    //std::cout << "Encoding Optimization = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    block_0.ssi.print();
    //std::cout<<"Angle: "<< angle<<", Rate: "<<rate<<", Distortion: "<<distortion<<", J0: "<<J0<<std::endl<<std::endl;
    return J0;
}

double TransformPartition :: RDoptimizeTransformStep_(Block4D_ &inputBlock, Block4D_ &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length, Hierarchical4DEncoder &entropyCoder, std::vector<SgtSideInfo>& currSsiBuffer,char **partitionCode) {
    
    ProbabilityModel *currentCoderModelState;
    entropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    
    //partitionCodeS handles splitting in the spatial dimension, partitionCodeV handles splitting in the view dimension.
    char *partitionCodeS=NULL;
    
    std::vector<SgtSideInfo> ssiBufferS;
    Block4D_ block_0 = inputBlock.copySubblock(length,position);
    //Block4D_ block_0(length);
    //block_0.CopySubblockFrom(inputBlock,position,{0,0,0,0});
    Block4D_ blockOrig = block_0;
    Block4D_ temp_block_0 = block_0;

    double currGain = totalTransformGain(length);
    double J0 = std::numeric_limits<double>::max();
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    double minAngle = angleRange[0];
    double maxAngle = angleRange[1];
    double angleStep = SgtSideInfo::PRECISION_ANGLE;
    //if((maxAngle-minAngle)/angleStep != std::floor((maxAngle-minAngle)/angleStep)) maxAngle += angleStep;
    std::cout<<"Angle Range: "<<minAngle<<" "<<maxAngle<<std::endl;
    int count = 0;
    for (double angle = minAngle; angle <=maxAngle; angle+=angleStep){ // Make this better later
        double J0_curr = EvaluatePartition_(temp_block_0,entropyCoder,currGain,angle);
        if (J0_curr < J0){
            J0 = J0_curr;
            block_0 = temp_block_0;
            //std::cout<<"tempBlock: ";
            //tempBlock.ssi.print();
            //block_0.ssi.print();
        }
        temp_block_0 = blockOrig;
        std::cout<<"                       \rAngle Search: "<<count++<<"/"<<trunc((maxAngle-minAngle)/angleStep)<<std::flush;
    }
    std::cout<<std::endl;

    SgtSideInfo ssi0 = block_0.ssi; 
    std::cout<<"Angle CHOSEN: "<<ssi0.getAngleH()<<std::endl;
    ssi0.print();
    //saves the resulting entropyCoder arithmetic model to model_0
    ProbabilityModel *coderModelState_0;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    //std::cout<<"HERE"<<std::endl;

    double JS = -1.0;
    Block4D_ transformedBlockS(length);
    transformedBlockS.emptyTransform();

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
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS00, new_position, new_length, entropyCoder, ssiBufferS00, &partitionCodeS00);
        
        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2; //???? Why?
                 
        Block4D_ transformedBlockS01(new_length);
        transformedBlockS01.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS01, new_position, new_length, entropyCoder,ssiBufferS01, &partitionCodeS01);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        Block4D_ transformedBlockS11(new_length);
        transformedBlockS11.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS11, new_position, new_length,  entropyCoder, ssiBufferS10, &partitionCodeS11);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        Block4D_ transformedBlockS10(new_length);
        transformedBlockS10.emptyTransform();
        
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS10, new_position, new_length, entropyCoder, ssiBufferS11, &partitionCodeS10);
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
        transformedBlockS.sgtDomain = true;
        //std::cout<<transformedBlockS.data.sizes()<<std::endl;        
        
        delete [] partitionCodeS00;
        delete [] partitionCodeS01;
        delete [] partitionCodeS10;
        delete [] partitionCodeS11; 
    }

    ProbabilityModel *coderModelState_s=NULL;
    entropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_s);
    
    //Restores the current arithmetic model using current_model. 
    entropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
   



    if(J0 > 0) 
        J0 += 1.0*mLambda;
    if(JS > 0)
        JS += 2.0*mLambda;
    
    
    
    //choose the lower cost and returns the corresponding cost,  the partition code and the arithmetic coder model
    //find best J
    int interview_split = 0;
    int intraview_split = 0;
    int no_split = 0;
    //std::cout<<length[2]<<"x"<<length[3]<<std::endl;
    //std::cout<<" J0 = "<<J0<<" JV = "<<JV<<" JS = "<<JS<<std::endl;
    
 
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
        entropyCoder.SetOptimizerProbabilisticModelState(coderModelState_s);
        //transformedBlock.CopySubblockFrom(transformedBlockS, {0,0,0,0},{0,0,0,0});
        transformedBlock = transformedBlockS;
        currSsiBuffer.insert(currSsiBuffer.end(),ssiBufferS.begin(), ssiBufferS.end());
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
        //transformedBlock.CopySubblockFrom(block_0, {0,0,0,0},{0,0,0,0});
        transformedBlock = block_0;
        currSsiBuffer.push_back(block_0.ssi);
    }

    if(partitionCodeS != NULL) {
        delete [] partitionCodeS;
    }

    entropyCoder.DeleteProbabilisticModelState(currentCoderModelState);
    entropyCoder.DeleteProbabilisticModelState(coderModelState_0);
    entropyCoder.DeleteProbabilisticModelState(coderModelState_s);

    return(optimumJ);     
}

void TransformPartition :: EncodePartition_(Hierarchical4DEncoder&entropyCoder, double lambda){

    double scaledLambda = lambda;
    std::array<int64_t,4> length;
    for(int i = 0; i < 4; ++i){
        length[i] = mPartitionData_.size[i];
        scaledLambda*=length[i];
    }
    mLambda = scaledLambda;
    //std::cout<<"Partition Data Size: "<<mPartitionData_.size[0]<<" "<<mPartitionData_.size[1]<<" "<<mPartitionData_.size[2]<<" "<<mPartitionData_.size[3]<<std::endl;
    //std::cout<<"Partition Data Transform Size: "<<mPartitionData_.transformSize[0]<<" "<<mPartitionData_.transformSize[1]<<" "<<mPartitionData_.transformSize[2]<<" "<<mPartitionData_.transformSize[3]<<std::endl;
    this->costImage = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->rateImage = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->distortionImage = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->rhoSImage = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->rhoTImage = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->rhoUImage = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->rhoVImage = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->angleImageH = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
    this->angleImageV = torch::zeros({mPartitionData_.size[2],mPartitionData_.size[3]}, torch::kDouble);
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

        std::array<int64_t,4> trueLength = length;
        std::array<int64_t,4> positionTransform = {0,0,position[2]*length[0],position[3]*length[1]};
        std::cout<<" isSgt? "<<entropyCoder.mSubbandLF_.sgtDomain<<std::endl;

        entropyCoder.mSubbandLF_ = mPartitionData_.copySubblock(length,positionTransform);
        
        //entropyCoder.mSubbandLF_ = Block4D_(length);
        //entropyCoder.mSubbandLF_.emptyTransform();
        //entropyCoder.mSubbandLF_.CopySubblockFrom(mPartitionData_, positionTransform,{0,0,0,0});

        //std::cout<<"mSubbandLF_ Size: "<<entropyCoder.mSubbandLF_.data.size(2)<<"x"<<entropyCoder.mSubbandLF_.data.size(3)<<std::endl;
        //std::cout<<"mPartitionData_ Size: "<<mPartitionData_.data.size(2)<<"x"<<mPartitionData_.data.size(3)<<std::endl;
        //std::cout<<"Size: "<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"First Coefficent BEING Compressed:"<<entropyCoder.mSubbandLF_.data[0][0][0][0].item()<<std::endl;
        //std::cout<<"Is data contiguous?"<<entropyCoder.mSubbandLF_.data.is_contiguous()<<std::endl;
        //int fCoeff = entropyCoder.mSubbandLF_.data[0][0][0][0].item<int>();


        //std::cout<<entropyCoder.mSubbandLF_.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

        // if (abs(fCoeff) < 1e5){
        //     std::cout<<"("<<position[2]<<","<<position[3]<<") "<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"Compressed:"<<std::endl<<entropyCoder.mSubbandLF_.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;
        
        //     std::cout<<"d = "<<mSsiBuffer[mSsiBufferIndex-1].getDisparity()<<std::endl;
        // }
        //std::cout<<"SGT Mean = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).mean().item()<<std::endl;
        //std::cout<<"SGT STD = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).std().item()<<std::endl;
        //std::cout<<"SGT Mean = "<<entropyCoder.mSubbandLF_.data.to(at::kDouble).mean({2,3},false,at::kDouble)<<std::endl;
        //std::cout<<this->distortionImage.sizes()<<std::endl;
        entropyCoder.EncodeSubblock_(lambda);
        this->rateImage.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = entropyCoder.mRate*at::ones({trueLength[2],trueLength[3]},at::kDouble);
        
        double weight = totalTransformGain(length);
        double distortion = (double) entropyCoder.mDistortion/(weight*weight);
        this->distortionImage.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = distortion*at::ones({trueLength[2],trueLength[3]},at::kDouble);

        this->rhoSImage.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = mSsiBuffer[mSsiBufferIndex-1].getRhoS()*at::ones({trueLength[2],trueLength[3]},at::kDouble);
        this->rhoTImage.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = mSsiBuffer[mSsiBufferIndex-1].getRhoT()*at::ones({trueLength[2],trueLength[3]},at::kDouble);
        this->rhoUImage.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = mSsiBuffer[mSsiBufferIndex-1].getRhoU()*at::ones({trueLength[2],trueLength[3]},at::kDouble);
        this->rhoVImage.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = mSsiBuffer[mSsiBufferIndex-1].getRhoV()*at::ones({trueLength[2],trueLength[3]},at::kDouble);
        this->angleImageH.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = mSsiBuffer[mSsiBufferIndex-1].getDisparityH()*at::ones({trueLength[2],trueLength[3]},at::kDouble);
        this->angleImageV.index({at::indexing::Slice({position[2],position[2]+trueLength[2]}),at::indexing::Slice({position[3],position[3]+trueLength[3]})}) = mSsiBuffer[mSsiBufferIndex-1].getDisparityV()*at::ones({trueLength[2],trueLength[3]},at::kDouble);
        //std::cout<<"Weight = "<<weight<<std::endl;
        std::cout<<"mRate = "<<entropyCoder.mRate<<" mDistortion: "<<(double) entropyCoder.mDistortion/(weight*weight)<<std::endl;

        //std::cout<<"Position: = "<<position[0]<<","<<position[1]<<","<<position[2]/9<<","<<position[3]/9<<" Length = "<<length[0]+8<<","<<length[1]+8<<","<<length[2]/9<<","<<length[3]/9<<" "<<(entropyCoder.currCost/(double)size)<< std::endl;

        return;
    }
    if(mPartitionCode[mPartitionCodeIndex] == INTRAVIEWSPLITFLAG) {
        
        mPartitionCodeIndex++;
        
        entropyCoder.EncodePartitionFlag(INTRAVIEWSPLITFLAGSYMBOL);
        
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


#include "Encoder/TransformPartition.h"
#include <chrono>


/*******************************************************************************/
/*                      TransformPartition class methods                    */
/*******************************************************************************/

TransformPartition :: TransformPartition(void) {
    mPartitionCode = NULL;
    
    //mUseSameBitPlane = 1;
}
TransformPartition :: TransformPartition(std::array<int64_t,4> minLength, Hierarchical4DEncoder entropyCoder,std::array<double,2> disparityRange, double transformGain)
    :mEntropyCoder(entropyCoder), mDisparityRange(disparityRange), mGain(transformGain) {
    mPartitionCode = NULL;
    mlength_t_min = minLength[0];
    mlength_s_min = minLength[1];
    mlength_v_min = minLength[2];
    mlength_u_min = minLength[3];
}
TransformPartition :: ~TransformPartition(void) {
    if(mPartitionCode != NULL)
        delete [] mPartitionCode;
}
double TransformPartition :: totalTransformGain(void){
    //length must be the length of the block in the spatial domain
    // double transformGain = 1;
    // for(int i = 0; i < 4; i++){
    //     transformGain*=length[i]/sqrt(length[i]);
    //     transformGain  *= sqrt(mPartitionData_.size[i]/length[i]);
    //     std::cout<<transformGain<<" "<< length[i]/sqrt(length[i]) << " "<< sqrt(mPartitionData_.size[i]/length[i])<<std::endl;
    // } 

    return sqrt(mPartitionData_.size[0]*mPartitionData_.size[1]*mPartitionData_.size[2]*mPartitionData_.size[3]);

}
void TransformPartition :: RDoptimizeTransform_(Block4D_ &inputBlock, double lambda){
    
    mEntropyCoder.RestartProbabilisticModel();
    inputBlock.data = inputBlock.data.contiguous();
    if(!mSsiBuffer.empty()) mSsiBuffer.clear();
    if(!mCuiBuffer.empty()) mCuiBuffer.clear();

    mCodingUnitIndex = 0;
    if(mPartitionCode != NULL)
        delete [] mPartitionCode;
    mPartitionCode = new char [1];
    mPartitionCode[0] = 0;          //initializes the partition code string as the null string
    mEvaluateOptimumBitPlane = 1;
    //std::array<int64_t,4> length = {inputBlock.data.size(0),inputBlock.data.size(1),inputBlock.data.size(2),inputBlock.data.size(3)};
    mPartitionData_ = Block4D_(inputBlock.size,inputBlock.lightFieldPosition,inputBlock.lightField);
    double scaledLambda = lambda;
    for (int i = 0; i < 4; i++){
        scaledLambda *= inputBlock.size[i];
    }
    mLambda = scaledLambda;
    mEntropyCoder.LoadOptimizerState();

    Block4D_ transformedBlock(inputBlock.size,inputBlock.lightFieldPosition,inputBlock.lightField);
    transformedBlock.emptyTransform();
    //std::cout<<"Transformed Block Pre Size: "<<transformedBlock.size[0]<<" "<<transformedBlock.size[1]<<" "<<transformedBlock.size[2]<<" "<<transformedBlock.size[3]<<std::endl;
    //std::cout<<"Transformed Block Pre Transform Size: "<<transformedBlock.transformSize[0]<<" "<<transformedBlock.transformSize[1]<<" "<<transformedBlock.transformSize[2]<<" "<<transformedBlock.transformSize[3]<<std::endl;
    
    mLagrangianCost = RDoptimizeTransformStep_(inputBlock, transformedBlock, {0,0,0,0}, inputBlock.size, mSsiBuffer,mCuiBuffer, &mPartitionCode);
   
    mPartitionData_ = transformedBlock;
    //std::cout<<"Transformed Block Size: "<<mPartitionData_.size[0]<<" "<<mPartitionData_.size[1]<<" "<<mPartitionData_.size[2]<<" "<<mPartitionData_.size[3]<<std::endl;
    //std::cout<<"Transformed Block Transform Size: "<<mPartitionData_.transformSize[0]<<" "<<mPartitionData_.transformSize[1]<<" "<<mPartitionData_.transformSize[2]<<" "<<mPartitionData_.transformSize[3]<<std::endl;
    mEntropyCoder.LoadOptimizerState();
    printf(" Full PartitionCode = %s\n", mPartitionCode);    
    //printf("mInferiorBitPlane = %d\n", mEntropyCoder.mInferiorBitPlane);
    //std::cout<<"Full Number of Compressed Blocks"<<mSsiBuffer.size()<<std::endl;

}



double TransformPartition :: EvaluatePartition_(Block4D_ &block_0, double currGain , double angleV, double angleH, ProbabilityModel *coderModelState){
    
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    //double angle = (angleV + angleH)/2;
    block_0.ssi = SgtSideInfo(angleV,angleH,mDisparityRange);
    block_0.ssi.estimateRhos(block_0,3000);

    block_0.sgtTransform(currGain);
    
    
    
    SgtSideInfo ssi0 = block_0.ssi; 
    mEntropyCoder.mSubbandLF_ = block_0;


    double Energy;
    double rate = 0;
    double distortion = 0;


    if(mEvaluateOptimumBitPlane == 1){
        mEntropyCoder.mInferiorBitPlane = mEntropyCoder.OptimumBitplaneFaster_(mLambda);
        mEntropyCoder.LoadOptimizerState();
        mEvaluateOptimumBitPlane = 0;
        //std::cout<<"MBP : "<<mEntropyCoder.mInferiorBitPlane<<std::endl;
    }
    if(mEntropyCoder.mSegmentationTreeCodeBuffer != NULL){
        delete [] mEntropyCoder.mSegmentationTreeCodeBuffer;
    }
    mEntropyCoder.mSegmentationTreeCodeBuffer = new char [2];
    strcpy(mEntropyCoder.mSegmentationTreeCodeBuffer,"");
    //begin  = std::chrono::steady_clock::now();
    std::array<int64_t,4> lengthTransform = {mEntropyCoder.mSubbandLF_.data.size(0), mEntropyCoder.mSubbandLF_.data.size(1), mEntropyCoder.mSubbandLF_.data.size(2), mEntropyCoder.mSubbandLF_.data.size(3)};
    //std::cout<<mEntropyCoder.mSuperiorBitPlane<<std::endl;
    double J0 = mEntropyCoder.RdOptimizeHexadecaTree_({0, 0, 0, 0}, lengthTransform, mLambda,mEntropyCoder.mSuperiorBitPlane, &mEntropyCoder.mSegmentationTreeCodeBuffer, Energy,rate,distortion);
    int RHO_PRECISION = ssi0.getRhoPrecision();
    int DISP_PRECISION = ssi0.getAnglePrecision();
    J0 += RHO_PRECISION*4*mLambda + DISP_PRECISION*mLambda;

    double weight = totalTransformGain();
    distortion = distortion/(block_0.size[0]*block_0.size[1]*block_0.size[2]*block_0.size[3]);
    rate = rate/(block_0.size[0]*block_0.size[1]*block_0.size[2]*block_0.size[3]);
    distortion = (double) distortion/(weight*weight);
    distortion = 10 * log10((1024*1024)/distortion);
    //if(block_0.size[2] == 32 && angleH == 26)std::cout<<"Angle: ("<< angleH<<","<<angleV<<") Rate: "<<rate<<" bpp, Distortion: "<<distortion<<" dB, J0: "<<J0<<std::endl<<std::endl;
    return J0;
}

double TransformPartition :: RDtestStructureTensor(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    double J0 = std::numeric_limits<double>::max();
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Evaluate Structure Tensor
    std::array<double,2> angles = blockOrig.computeAnglesFromStructureTensor(mDisparityRange);
    std::array<double,3> anglesToTest = {angles[0],angles[1],(angles[0]+angles[1])/2};
    for (int i = 0; i < 3; i++){
        ProbabilityModel *modelStateCurr;
        mEntropyCoder.GetOptimizerProbabilisticModelState(&modelStateCurr);
        double J0_curr = EvaluatePartition_(temp_block_0,currGain,anglesToTest[i],anglesToTest[i],modelStateCurr);
        if(i == 0) cui0.setStructureTensorHorizontal({anglesToTest[i],J0_curr});
        if(i == 1) cui0.setStructureTensorVertical({anglesToTest[i],J0_curr});
        if(i == 2) cui0.setStructureTensorAverage({anglesToTest[i],J0_curr});
        if (J0_curr < J0){
            J0 = J0_curr;
            block_0 = temp_block_0;
            mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);

            if(i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_HORIZONTAL);
            if(i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_VERTICAL);
            if(i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_AVERAGE);
        }
        mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
        temp_block_0 = blockOrig;
    }

    return J0;
}
double TransformPartition :: RDtestCovariance(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    double J0 = std::numeric_limits<double>::max();
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Evaluate Structure Tensor
    std::array<double,2> angles;
    angles[0] = blockOrig.getOrientationFromCovariance(0.01,mDisparityRange,true);
    angles[1] = blockOrig.getOrientationFromCovariance(0.01,mDisparityRange,false);
    std::array<double,3> anglesToTest = {angles[0],angles[1],(angles[0]+angles[1])/2};
    for (int i = 0; i < 3; i++){
        ProbabilityModel *modelStateCurr;
        mEntropyCoder.GetOptimizerProbabilisticModelState(&modelStateCurr);
        double J0_curr = EvaluatePartition_(temp_block_0,currGain,anglesToTest[i],anglesToTest[i],modelStateCurr);
        // if(i == 0) cui0.setStructureTensorHorizontal({anglesToTest[i],J0_curr});
        // if(i == 1) cui0.setStructureTensorVertical({anglesToTest[i],J0_curr});
        // if(i == 2) cui0.setStructureTensorAverage({anglesToTest[i],J0_curr});
        if (J0_curr < J0){
            J0 = J0_curr;
            block_0 = temp_block_0;
            mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);

            // if(i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_HORIZONTAL);
            // if(i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_VERTICAL);
            // if(i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_AVERAGE);
        }
        mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
        temp_block_0 = blockOrig;
    }

    return J0;
}

double TransformPartition :: RDtestLogdet(Block4D_& block_0,  CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    double J0 = std::numeric_limits<double>::max();
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Evaluate Logdet

    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    double minAngle = angleRange[0];
    double maxAngle = angleRange[1];
    double angleStep = 1;
    std::array<double,2> logdetAngles = blockOrig.logDetAngleEstimation(angleStep,mDisparityRange);
    std::array<double,3> anglesToTest = {logdetAngles[0],logdetAngles[1],(logdetAngles[0]+logdetAngles[1])/2};

    for (int i = 0; i < 3; i++){
        ProbabilityModel *modelStateCurr;
        mEntropyCoder.GetOptimizerProbabilisticModelState(&modelStateCurr);
        double J0_curr = EvaluatePartition_(temp_block_0,currGain,anglesToTest[i],anglesToTest[i],modelStateCurr);

        if(i == 0) cui0.setLogdetHorizontal({anglesToTest[i],J0_curr});
        if(i == 1) cui0.setLogdetVertical({anglesToTest[i],J0_curr});
        if(i == 2) cui0.setLogdetAverage({anglesToTest[i],J0_curr});
        if (J0_curr < J0){
            J0 = J0_curr;
            block_0 = temp_block_0;
            mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
            if(i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_HORIZONTAL);
            if(i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_VERTICAL);
            if(i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_AVERAGE);
        }
        
        mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
        temp_block_0 = blockOrig;

    }

    return J0;
}
double TransformPartition :: RDtestGridSearch(double angleStep,std::array<double,2> angleRange, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    double J0 = std::numeric_limits<double>::max();
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Evaluate GS

    double minAngle = angleRange[0];
    double maxAngle = angleRange[1];
    int count = 0;
    for (double angle = minAngle; angle <=maxAngle; angle+=angleStep){ 
        ProbabilityModel *modelStateCurr;
        mEntropyCoder.GetOptimizerProbabilisticModelState(&modelStateCurr);
        double J0_curr = EvaluatePartition_(temp_block_0,currGain,angle,angle,modelStateCurr);
        cui0.addGridSearchAngle(angle,J0_curr);
        if (J0_curr < J0){
            J0 = J0_curr;
            block_0 = temp_block_0;
            mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
            cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);
            //std::cout<<"tempBlock: ";
            //tempBlock.ssi.print();
            //block_0.ssi.print();
        }
        mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
        temp_block_0 = blockOrig;
        //std::cout<<"                       \rAngle Search: "<<count++<<"/"<<trunc((maxAngle-minAngle)/angleStep)<<std::flush;
    }

    return J0;
}
double TransformPartition :: RDrefineStructureTensor(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();


    //Evaluate Structure Tensor
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModel *tempModelState;
    
    double J = RDtestStructureTensor(blockTemp,cui0,currGain,&tempModelState);
    
    J0 = J;
    block_0 = blockTemp;
    mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);
    mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    blockTemp = blockOrig;

    double angle = block_0.ssi.getAngleH();

    //Grid Search Refinement
    std::array<double,2> angleRange = {angle-10,angle+10};

    J = RDtestGridSearch(refinementPrecision,angleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    return J0;
}
double TransformPartition :: RDtestAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();


    //Evaluate Structure Tensor
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModel *tempModelState;
    
    double J = RDtestStructureTensor(blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    blockTemp = blockOrig;
     //Evaluate Covariance
    J = RDtestCovariance(blockTemp,cui0,currGain,&tempModelState);
     if(J < J0){
         J0 = J;
         block_0 = blockTemp;
         mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);
 
         mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
     }
 
     mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
     blockTemp = blockOrig;
    //Evaluate Logdet
    J = RDtestLogdet(blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    blockTemp = blockOrig;

    //Evaluate Grid Search
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);

    J = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    return J0;
}
double TransformPartition :: RDrefineLogdet(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    double currGain = totalTransformGain();
    
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModel *tempModelState;
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();
    //Evaluate Logdet
    //Evaluate Logdet
    double J = RDtestLogdet(blockTemp,cui0,currGain,&tempModelState);
    J0 = J;
    block_0 = blockTemp;
    mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);
    mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    
    blockTemp = blockOrig;
    //Grid Search Refinement
    double angle = block_0.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};

    J = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,refinementAngleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);


    return J0;
}

double TransformPartition :: RDrefineGridSearch(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    double currGain = totalTransformGain();
    
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModel *tempModelState;
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();
    //Evaluate Grid Search
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);

    double J = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,&tempModelState);
    
    J0 = J;
    block_0 = blockTemp;
    mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);
    mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    blockTemp = blockOrig;
    //Grid Search Refinement
    double angle = block_0.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};

    J = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,refinementAngleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);


    return J0;
}


double TransformPartition :: RDrefineAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();


    //Evaluate Structure Tensor
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModel *tempModelState;
    double J;
    J = RDtestStructureTensor(blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    blockTemp = blockOrig;
    //Evaluate Logdet
    J = RDtestLogdet(blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    blockTemp = blockOrig;

    //Evaluate Grid Search
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);

    J = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);
        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    blockTemp = blockOrig;
    //Grid Search Refinement
    double angle = block_0.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};

    J = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,refinementAngleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);


    return J0;
}

double TransformPartition :: RDoptimizeTransformStep_(Block4D_ &inputBlock, Block4D_ &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length, std::vector<SgtSideInfo>& currSsiBuffer,std::vector<CodingUnitInfo>& currCuiBuffer,char **partitionCode) {

    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    
    //partitionCodeS handles splitting in the spatial dimension, partitionCodeV handles splitting in the view dimension.
    char *partitionCodeS=NULL;
    std::array<int64_t,4> lightFieldPosition = inputBlock.lightFieldPosition;
    for(int i = 0; i < 4; i++){
        lightFieldPosition[i] += position[i];
    }
    
    std::vector<SgtSideInfo> ssiBufferS;
    std::vector<CodingUnitInfo> cuiBufferS;
    Block4D_ block_0 = inputBlock.copySubblock(length,position);
    CodingUnitInfo cui0(block_0.size,block_0.lightFieldPosition);

    //std::cout<<"Light Field Position: "<<block_0.lightFieldPosition[0]<<" "<<block_0.lightFieldPosition[1]<<" "<<block_0.lightFieldPosition[2]<<" "<<block_0.lightFieldPosition[3]<<std::endl;
    //std::cout<<"cui0 Light Field Position: "<<cui0.getLightFieldPosition()[0]<<" "<<cui0.getLightFieldPosition()[1]<<" "<<cui0.getLightFieldPosition()[2]<<" "<<cui0.getLightFieldPosition()[3]<<std::endl;
    //std::cout<<"Position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
    //std::cout<<"Light Field Position: "<<block_0.lightFieldPosition<<std::endl;

    //Block4D_ block_0(length);
    //block_0.CopySubblockFrom(inputBlock,position,{0,0,0,0});
    Block4D_ blockOrig = block_0;
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *coderModelState_0;
    //double J0 = RDrefineAllAngleHeuristics(block_0,cui0,&coderModelState_0);
    double J0 = RDtestAllAngleHeuristics(block_0,cui0,&coderModelState_0);
    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    // double J0 = std::numeric_limits<double>::max();
    // ProbabilityModel *coderModelState_0;


    // //Evaluate Structure Tensor
    // Block4D_ blockTemp = block_0;

    // ProbabilityModel *tempModelState;
    
    // double J = RDtestStructureTensor(blockTemp,cui0,currGain,&tempModelState);
    // if(J < J0){
    //     J0 = J;

    //     mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

    //     mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    // }

    // mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    // blockTemp = block_0;
    // //Evaluate Logdet
    // J = RDtestLogdet(blockTemp,cui0,currGain,&tempModelState);
    // if(J < J0){
    //     J0 = J;

    //     mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

    //     mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    // }

    // mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    // blockTemp = block_0;

    // //Evaluate Grid Search
    // J = RDtestGridSearch(1,blockTemp,cui0,currGain,&tempModelState);
    // if(J < J0){
    //     J0 = J;

    //     mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);

    //     mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    // }


    // std::array<double,2> angles = blockOrig.computeAnglesFromStructureTensor(mDisparityRange);
    // //std::cout<<"Angles: "<<angles[0]<<" "<<angles[1]<<std::endl;
    // std::array<double,3> anglesToTest = {angles[0],angles[1],(angles[0]+angles[1])/2};
    // for (int i = 0; i < 3; i++){
    //     ProbabilityModel *modelStateCurr;
    //     mEntropyCoder.GetOptimizerProbabilisticModelState(&modelStateCurr);
    //     double J0_curr = EvaluatePartition_(temp_block_0,currGain,anglesToTest[i],anglesToTest[i],modelStateCurr);
    //     if(i == 0) cui0.setStructureTensorHorizontal({anglesToTest[i],J0_curr});
    //     if(i == 1) cui0.setStructureTensorVertical({anglesToTest[i],J0_curr});
    //     if(i == 2) cui0.setStructureTensorAverage({anglesToTest[i],J0_curr});
    //     if (J0_curr < J0){
    //         J0 = J0_curr;
    //         block_0 = temp_block_0;
    //         mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    //         if(i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_HORIZONTAL);
    //         if(i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_VERTICAL);
    //         if(i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_AVERAGE);
    //     }
    //     mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    //     temp_block_0 = blockOrig;
    // }
    
    
    // //Evaluate Logdet

    // std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    // double minAngle = angleRange[0];
    // double maxAngle = angleRange[1];
    // double angleStep = 1;
    // std::array<double,2> logdetAngles = blockOrig.logDetAngleEstimation(angleStep,mDisparityRange);
    // std::array<double,3> anglesToTest = {logdetAngles[0],logdetAngles[1],(logdetAngles[0]+logdetAngles[1])/2};

    // for (int i = 0; i < 3; i++){
    //     ProbabilityModel *modelStateCurr;
    //     mEntropyCoder.GetOptimizerProbabilisticModelState(&modelStateCurr);
    //     double J0_curr = EvaluatePartition_(temp_block_0,currGain,anglesToTest[i],anglesToTest[i],modelStateCurr);

    //     if(i == 0) cui0.setLogdetHorizontal({anglesToTest[i],J0_curr});
    //     if(i == 1) cui0.setLogdetVertical({anglesToTest[i],J0_curr});
    //     if(i == 2) cui0.setLogdetAverage({anglesToTest[i],J0_curr});
    //     if (J0_curr < J0){
    //         J0 = J0_curr;
    //         block_0 = temp_block_0;
    //         mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    //         if(i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_HORIZONTAL);
    //         if(i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_VERTICAL);
    //         if(i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_AVERAGE);
    //     }
    //     mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);

    //     temp_block_0 = blockOrig;

    // }

    //double angleStep = SgtSideInfo::PRECISION_ANGLE;
    //if((maxAngle-minAngle)/angleStep != std::floor((maxAngle-minAngle)/angleStep)) maxAngle += angleStep;
    //std::cout<<"Angle Range: "<<minAngle<<" "<<maxAngle<<std::endl;
    // int count = 0;
    // for (double angle = minAngle; angle <=maxAngle; angle+=angleStep){ // Make this better later
    //     ProbabilityModel *modelStateCurr;
    //     mEntropyCoder.GetOptimizerProbabilisticModelState(&modelStateCurr);
    //     double J0_curr = EvaluatePartition_(temp_block_0,currGain,angle,angle,modelStateCurr);
    //     cui0.addGridSearchAngle(angle,J0_curr);
    //     if (J0_curr < J0){
    //         J0 = J0_curr;
    //         block_0 = temp_block_0;
    //         mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_0);
    //         cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);
    //         //std::cout<<"tempBlock: ";
    //         //tempBlock.ssi.print();
    //         //block_0.ssi.print();
    //     }
    //     mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    //     temp_block_0 = blockOrig;
    //     //std::cout<<"                       \rAngle Search: "<<count++<<"/"<<trunc((maxAngle-minAngle)/angleStep)<<std::flush;
    
    // }
    // //std::cout<<std::endl;
 



    SgtSideInfo ssi0 = block_0.ssi; 
    
    cui0.setSgtSideInfo(ssi0);
    
    if(length[2] == 64) std::cout<<"Angle CHOSEN: "<<ssi0.getAngleH()<<" "<<ssi0.getAngleV()<<std::endl;
    //ssi0.print();
    //saves the resulting entropyCoder arithmetic model to model_0
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    //std::cout<<"HERE"<<std::endl;

    double JS = -1.0;
    Block4D_ transformedBlockS(length,lightFieldPosition,inputBlock.lightField);
    transformedBlockS.emptyTransform();

    //If you can split more in the spatial dimension
    if((length[3] >= 2*mlength_u_min)&&(length[2] >= 2*mlength_v_min)) {
        JS = 0.0;
        std::vector<SgtSideInfo> ssiBufferS00, ssiBufferS01, ssiBufferS10, ssiBufferS11;
        std::vector<CodingUnitInfo> cuiBufferS00, cuiBufferS01, cuiBufferS10, cuiBufferS11;
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
        Block4D_ transformedBlockS00(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS00.emptyTransform();

        
        //Need to see what this is actually doing...
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS00, new_position, new_length, ssiBufferS00,cuiBufferS00, &partitionCodeS00);
        
        new_position[3] = position[3] + length[3]/2;
        //new_lightField_position[3] = lightFieldPosition[3] + new_position[3];
        new_length[3] = length[3] - length[3]/2; 
                 
        Block4D_ transformedBlockS01(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS01.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS01, new_position, new_length,ssiBufferS01,cuiBufferS01, &partitionCodeS01);

        new_position[2] = position[2] + length[2]/2;
        //new_lightField_position[2] = lightFieldPosition[2] + new_position[2];

        new_length[2] = length[2] - length[2]/2;
        
        Block4D_ transformedBlockS11(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS11.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS11, new_position, new_length, ssiBufferS10,cuiBufferS10, &partitionCodeS11);
        
        new_position[3] = position[3];
        new_lightField_position[3] = lightFieldPosition[2] + new_position[3];
        
        new_length[3] = length[3]/2;
        
        Block4D_ transformedBlockS10(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS10.emptyTransform();
        
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS10, new_position, new_length, ssiBufferS11,cuiBufferS11, &partitionCodeS10);
        //concatenates side info buffers
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS00.begin(), ssiBufferS00.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS01.begin(), ssiBufferS01.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS10.begin(), ssiBufferS10.end());
        ssiBufferS.insert(ssiBufferS.end(), ssiBufferS11.begin(), ssiBufferS11.end());
        
        cuiBufferS.insert(cuiBufferS.end(), cuiBufferS00.begin(), cuiBufferS00.end());
        cuiBufferS.insert(cuiBufferS.end(), cuiBufferS01.begin(), cuiBufferS01.end());
        cuiBufferS.insert(cuiBufferS.end(), cuiBufferS10.begin(), cuiBufferS10.end());
        cuiBufferS.insert(cuiBufferS.end(), cuiBufferS11.begin(), cuiBufferS11.end());
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
    mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_s);
    
    //Restores the current arithmetic model using current_model. 
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);

    //std::cout<<"lambda: "<<mLambda<<std::endl;
   



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
    //if(length[2] == 32)std::cout<<" J0 = "<<J0<<" JS = "<<JS<<std::endl;
    
 
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
        transformedBlock = transformedBlockS;
        currSsiBuffer.insert(currSsiBuffer.end(),ssiBufferS.begin(), ssiBufferS.end());
        currCuiBuffer.insert(currCuiBuffer.end(),cuiBufferS.begin(), cuiBufferS.end());

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
        transformedBlock = block_0;
        currSsiBuffer.push_back(block_0.ssi);
        currCuiBuffer.push_back(cui0);
    }

    if(partitionCodeS != NULL) {
        delete [] partitionCodeS;
    }

    mEntropyCoder.DeleteProbabilisticModelState(currentCoderModelState);
    mEntropyCoder.DeleteProbabilisticModelState(coderModelState_0);
    mEntropyCoder.DeleteProbabilisticModelState(coderModelState_s);

    return(optimumJ);     
}

void TransformPartition :: EncodePartition_(double lambda){

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
    //std::cout<<"Minimum Bit Plane: "<<mEntropyCoder.mInferiorBitPlane<<std::endl;


    EncodePartitionStep_(position, length, scaledLambda);
}

void TransformPartition :: EncodePartitionStep_(std::array<int64_t,4> position, std::array<int64_t,4>length,  double lambda) {
    //std::cout<<mPartitionCode[mPartitionCodeIndex]<<" "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
    if(mPartitionCode[mPartitionCodeIndex] == NOSPLITFLAG) {
      
        mPartitionCodeIndex++;
        mEntropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
        // std::cout<< "RhoS = "<<mSsiBuffer[mSsiBufferIndex].getRhoS()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoSCode()<<std::endl;
        // std::cout<< "RhoT = "<<mSsiBuffer[mSsiBufferIndex].getRhoT()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoTCode()<<std::endl;
        // std::cout<< "RhoU = "<<mSsiBuffer[mSsiBufferIndex].getRhoU()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoUCode()<<std::endl;
        // std::cout<< "RhoV = "<<mSsiBuffer[mSsiBufferIndex].getRhoV()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getRhoVCode()<<std::endl;
        // std::cout<<"Disparity = "<<mSsiBuffer[mSsiBufferIndex].getDisparity()<<" Code: "<<mSsiBuffer[mSsiBufferIndex].getDCode()<<std::endl;
        //std::cout<<"Ssi Buffer Size: "<<mSsiBuffer.size()<<std::endl;
        //std::cout<<"Buffer Size: "<<mCuiBuffer.size()<<std::endl;
        //std::cout<<"LightFieldPositionOfEncodedBlock: "<<mCuiBuffer[mCodingUnitIndex].getLightFieldPosition()[0]<<" "<<mCuiBuffer[mCodingUnitIndex].getLightFieldPosition()[1]<<" "<<mCuiBuffer[mCodingUnitIndex].getLightFieldPosition()[2]<<" "<<mCuiBuffer[mCodingUnitIndex].getLightFieldPosition()[3]<<std::endl;
        
        mSsiBuffer[mCodingUnitIndex].print();
        mCuiBuffer[mCodingUnitIndex].getSgtSideInfo().print();
        

        mEntropyCoder.EncodeSSI_(mSsiBuffer[mCodingUnitIndex]);
        //std::cout<<"Length:"<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"Position:"<<position[0]<<"x"<<position[1]<<"x"<<position[2]<<"x"<<position[3]<<std::endl;
        std::array<int64_t,4> trueLength = length;
        //std::array<int64_t,4> positionTransform = {0,0,position[2]*length[0],position[3]*length[1]};


        mEntropyCoder.mSubbandLF_ = mPartitionData_.copySubblock(length,position);
        mEntropyCoder.EncodeSubblock_(lambda);
        
        double weight = totalTransformGain();
        double distortion = (double) mEntropyCoder.mDistortion/(weight*weight);
        //std::cout<<"mRate = "<<mEntropyCoder.mRate<<" mDistortion: "<<(double) mEntropyCoder.mDistortion/(weight*weight)<<std::endl;
        distortion = 10 * log10((1024*1024)/distortion);
        mCuiBuffer[mCodingUnitIndex].setPSNR(distortion);
        mCuiBuffer[mCodingUnitIndex].setRate(mEntropyCoder.mRate);
        mCodingPartitionInfo.appendCodingUnitInfo(mCuiBuffer[mCodingUnitIndex]);
        //std::cout<<"stA: "<<mCuiBuffer[mCodingUnitIndex].getStructureTensorAverageAngle()<<std::endl;
        //std::cout<<"ldA: "<<mCuiBuffer[mCodingUnitIndex].getLogdetAverageAngle()<<std::endl;

        mCodingUnitIndex++;

        //std::cout<<"Weight = "<<weight<<std::endl;

        //std::cout<<"Position: = "<<position[0]<<","<<position[1]<<","<<position[2]/9<<","<<position[3]/9<<" Length = "<<length[0]+8<<","<<length[1]+8<<","<<length[2]/9<<","<<length[3]/9<<" "<<(mEntropyCoder.currCost/(double)size)<< std::endl;

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
        EncodePartitionStep_(new_position, new_length, lambda);

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
        
        EncodePartitionStep_(new_position, new_length, lambda);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        EncodePartitionStep_(new_position, new_length, lambda);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        EncodePartitionStep_(new_position, new_length, lambda);
        return;
    }
}


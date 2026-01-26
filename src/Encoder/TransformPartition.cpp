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
    //length must be the length of the block in the spatial domain
    // double transformGain = 1;
    // for(int i = 0; i < 4; i++){
    //     transformGain*=length[i]/sqrt(length[i]);
    //     transformGain  *= sqrt(mPartitionData_.size[i]/length[i]);
    //     std::cout<<transformGain<<" "<< length[i]/sqrt(length[i]) << " "<< sqrt(mPartitionData_.size[i]/length[i])<<std::endl;
    // } 

    return mGain*sqrt(mPartitionData_.size[0]*mPartitionData_.size[1]*mPartitionData_.size[2]*mPartitionData_.size[3]);

}
void TransformPartition :: RDoptimizeTransform_(Block4D_ &inputBlock, double lambda){
    std::cout<<"Starting RDoptimizeTransform with lambda: " << lambda << std::endl;
    mEntropyCoder.RestartProbabilisticModel();
    for(int i = 0; i < m_encoder_pool.size(); i++) {
        m_encoder_pool[i]->RestartProbabilisticModel();
    }

    std::cout<<"I have a feeling we've double freed something"<<std::endl;
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
        std::cout<<"mPartitionData_ valid position size:"<<mPartitionData_.validPositions.valid_positions_v.size(0) << std::endl;
        std::cout<<"mPartitionData_ includes invalids:"<<mPartitionData_.includesInvalidCorners << std::endl;
    double scaledLambda = lambda;
    for (int i = 0; i < 4; i++){
        scaledLambda *= inputBlock.size[i];
    }
    mLambda = scaledLambda;
    mEntropyCoder.LoadOptimizerState();

    Block4D_ transformedBlock(inputBlock.size,inputBlock.lightFieldPosition,inputBlock.lightField);
    std::cout<<"transformedBlock valid position size:"<<transformedBlock.validPositions.valid_positions_v.size(0) << std::endl;
    std::cout<<"transformedBlock includes invalids:"<<transformedBlock.includesInvalidCorners << std::endl;


    transformedBlock.emptyTransform();
    //std::cout<<"Transformed Block Pre Size: "<<transformedBlock.size[0]<<" "<<transformedBlock.size[1]<<" "<<transformedBlock.size[2]<<" "<<transformedBlock.size[3]<<std::endl;
    //std::cout<<"Transformed Block Pre Transform Size: "<<transformedBlock.transformSize[0]<<" "<<transformedBlock.transformSize[1]<<" "<<transformedBlock.transformSize[2]<<" "<<transformedBlock.transformSize[3]<<std::endl;
    mDepth = 0;
    getOptimalMinimumBitPlane(inputBlock);
    mLagrangianCost = RDoptimizeTransformStep_(inputBlock, transformedBlock, {0,0,0,0}, inputBlock.size, mSsiBuffer,mCuiBuffer, &mPartitionCode);
    //std::cout<<"Lagrangian Cost: "<<mLagrangianCost<<std::endl;
    mPartitionData_ = transformedBlock;
    std::cout<<"mPartitionData_ valid position size after transform:"<<mPartitionData_.validPositions.valid_positions_v.size(0) << std::endl;
    std::cout<<"mPartitionData_ includes invalids:"<<mPartitionData_.includesInvalidCorners << std::endl;

            //std::cout<<"Transformed Block Size: "<<transformedBlock.size[0]<<" "<<transformedBlock.size[1]<<" "<<transformedBlock.size[2]<<" "<<transformedBlock.size[3]<<std::endl;
    //std::cout<<"Transformed Block Size: "<<mPartitionData_.size[0]<<" "<<mPartitionData_.size[1]<<" "<<mPartitionData_.size[2]<<" "<<mPartitionData_.size[3]<<std::endl;
    //std::cout<<"Transformed Block Transform Size: "<<mPartitionData_.transformSize[0]<<" "<<mPartitionData_.transformSize[1]<<" "<<mPartitionData_.transformSize[2]<<" "<<mPartitionData_.transformSize[3]<<std::endl;
    mEntropyCoder.LoadOptimizerState();
    printf(" Full PartitionCode = %s\n", mPartitionCode);    
    //printf("mInferiorBitPlane = %d\n", mEntropyCoder.mInferiorBitPlane);
    //std::cout<<"Full Number of Compressed Blocks"<<mSsiBuffer.size()<<std::endl;

}

void TransformPartition :: getOptimalMinimumBitPlane(Block4D_& inputBlock){
    // This function is used to find the optimal minimum bit plane for the input block.
    // It sets the mInferiorBitPlane of the encoder to the optimal value.
    Block4D_ block_0 = inputBlock.clone();
    block_0.ssi = SgtSideInfo(0,0,mDisparityRange);
    block_0.sgtTransform(this->totalTransformGain());

    mEntropyCoder.mSubbandLF_ = block_0;
    mEntropyCoder.RestartProbabilisticModel();
    mEntropyCoder.mInferiorBitPlane = mEntropyCoder.OptimumBitplaneFaster_(mLambda);
    mEntropyCoder.LoadOptimizerState();
    for(int i = 0; i < m_encoder_pool.size(); i++) {
        m_encoder_pool[i]->mInferiorBitPlane = mEntropyCoder.mInferiorBitPlane;
    }
}

double TransformPartition :: EvaluatePartitionArbitraryRho(Hierarchical4DEncoder& encoder, Block4D_ &block_0, double currGain , double angle, double rhoAngle, double rhoSpace){
    
    block_0.ssi = SgtSideInfo(angle,angle,mDisparityRange);
    block_0.ssi.setAngularRhos(rhoAngle,rhoAngle);
    block_0.ssi.setSpatialRhos(rhoSpace,rhoSpace);
    //std::cout<<"Evaluating Partition Fixed Rho: "<<angle<<" "<<angle<<" "<<block_0.ssi.getAngleH()<<" "<<block_0.ssi.getAngleV()<<std::endl;
    return EvaluatePartition_(encoder,block_0, currGain, angle, angle);
}
double TransformPartition :: EvaluatePartitionFixedRho(Hierarchical4DEncoder& encoder, Block4D_ &block_0, double currGain , double angleV, double angleH){
    
    block_0.ssi = SgtSideInfo(angleV,angleH,mDisparityRange);
    //std::cout<<"Evaluating Partition Fixed Rho: "<<angleV<<" "<<angleH<<" "<<block_0.ssi.getAngleH()<<" "<<block_0.ssi.getAngleV()<<std::endl;
    return EvaluatePartition_(encoder,block_0, currGain, angleV, angleH);
}
double TransformPartition :: EvaluatePartitionLSRho(Hierarchical4DEncoder& encoder,Block4D_ &block_0, double currGain , double angleV, double angleH){
    block_0.ssi = SgtSideInfo(angleV,angleH,mDisparityRange);
    block_0.ssi.estimateRhos(block_0,-1);
    return EvaluatePartition_(encoder,block_0, currGain, angleV, angleH);
}

double TransformPartition :: EvaluatePartition_(Hierarchical4DEncoder& encoder, Block4D_ &block_0, double currGain , double angleV, double angleH){
    
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    //double angle = (angleV + angleH)/2;
    //block_0.ssi = SgtSideInfo(angleV,angleH,mDisparityRange);
    //block_0.ssi.estimateRhos(block_0,3000);
    std::chrono::steady_clock::time_point sTranform = std::chrono::steady_clock::now();
    block_0.sgtTransform(currGain);

    //std::cout<<"After Transform Pointer: " <<block_0.data.data_ptr<int>() <<" in thread: "<<omp_get_thread_num()<<std::endl;

    std::chrono::steady_clock::time_point fTransform = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> eTransform = fTransform - sTranform;
    //std::cout << "Transform Time: " << eTransform.count() << " ms" << std::endl;
    
    
    
    SgtSideInfo ssi0 = block_0.ssi; 

    encoder.mSubbandLF_ = block_0;
    //std::cout<<"After Transform Pointer: " <<block_0.data.data_ptr<int>() <<" in thread: "<<omp_get_thread_num()<<std::endl;


    double Energy;
    double rate = 0;
    double distortion = 0;

    //encoder.mInferiorBitPlane = 14; // Set the inferior bit plane to a default value 
    // if(mEvaluateOptimumBitPlane == 1){
    //     encoder.mInferiorBitPlane = encoder.OptimumBitplaneFaster_(mLambda);
    //     encoder.LoadOptimizerState();
    //     mEvaluateOptimumBitPlane = 0;
    //     //std::cout<<"MBP : "<<mEntropyCoder.mInferiorBitPlane<<std::endl;
    // }
   
    
    //begin  = std::chrono::steady_clock::now();
    std::array<int64_t,4> lengthTransform = {encoder.mSubbandLF_.data.size(0), encoder.mSubbandLF_.data.size(1), encoder.mSubbandLF_.data.size(2), encoder.mSubbandLF_.data.size(3)};
    // //std::cout<<mEntropyCoder.mSuperiorBitPlane<<std::endl;
    // std::cout<<"OPTIMIZING HEXADECA TREE THE NEW WAY"<<std::endl;
    // std::cout<<"_____________________________________________________"<<std::endl;
    std::chrono::steady_clock::time_point sEncode = std::chrono::steady_clock::now();
    double J0 = encoder.build_optimal_tree_from_pool(lengthTransform,{0,0,0,0}, encoder.mSuperiorBitPlane, mLambda);
    std::chrono::steady_clock::time_point fEncode = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> eEncode= fEncode- sEncode;
    //std::cout << "Encoding Time: " << eEncode.count() << " ms" << std::endl;
    //std::cout<<mEntropyCoder.mSuperiorBitPlane<<std::endl;
    // std::cout<<"_____________________________________________________"<<std::endl;
    // std::cout<<"OPTIMIZING HEXADECA TREE THE OLD WAY"<<std::endl;
    // std::cout<<"_____________________________________________________"<<std::endl;
    //HexResult res = mEntropyCoder.RdOptimizeHexadecaTree_({0, 0, 0, 0}, lengthTransform, mLambda,mEntropyCoder.mSuperiorBitPlane, Energy,rate,distortion);
    
    //double J0 = res.cost;

    //std::cout<<"J0: "<<J0<<" J0_new: "<<J0<<std::endl<<std::endl;
    //mEntropyCoder.mSegmentationTreeCodeBuffer = res.codeStream;
    //std::cout<<res.codeStream<<std::endl;
    //std::cout<<"OPTIMIZED HEXADECA TREE"<<std::endl;

    int RHO_PRECISION = ssi0.getRhoPrecision();
    int DISP_PRECISION = ssi0.getAnglePrecision();
    J0 += RHO_PRECISION*4*mLambda + DISP_PRECISION*mLambda;

    double weight = totalTransformGain();
    distortion = distortion/(block_0.size[0]*block_0.size[1]*block_0.size[2]*block_0.size[3]);
    rate = rate/(block_0.size[0]*block_0.size[1]*block_0.size[2]*block_0.size[3]);
    distortion = (double) distortion/(weight*weight);
    distortion = 10 * log10((1024*1024)/distortion);
    //std::cout<<"EVALUATED ANGLE "<<angleV<<std::endl;

    //if(block_0.size[2] == 32 && angleH == 26)std::cout<<"Angle: ("<< angleH<<","<<angleV<<") Rate: "<<rate<<" bpp, Distortion: "<<distortion<<" dB, J0: "<<J0<<std::endl<<std::endl;
    return J0;
}

double TransformPartition :: RDtestAngle(double angle,Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Evaluate Structure Tensor
    ProbabilityModel *modelStateCurr;
    double J0 = EvaluatePartitionFixedRho(mEntropyCoder,temp_block_0,currGain,angle,angle);
    block_0 = temp_block_0;
    mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);

    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    delete[] currentCoderModelState;

    return J0;
}
double TransformPartition :: RDtestZero(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    cui0.setAngleHeuristicUsed(AngleHeuristic::ZERO);
    return RDtestAngle(0,block_0,cui0,currGain, coderModelState_0);
}
double TransformPartition :: RDgridSearchAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockTemp = block_0.clone();
    double J0;
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    ProbabilityModel *tempModelState;
    J0 = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,&tempModelState);
    delete[] tempModelState;
    //J0 = RDtestStructureTensor(blockTemp,cui0,currGain,coderModelState_0);
    double angle = blockTemp.ssi.getAngleH();
    blockTemp = block_0.clone();
    //double angle = 45;
    J0 = parallelRhoSearch(false,-1, angle, block_0, cui0, currGain, coderModelState_0);
    //J0 = parallelRhoSearch(true,blockTemp.ssi.getRhoS(), angle, block_0, cui0, currGain, coderModelState_0);
    return J0;
}
double TransformPartition :: RDtestStructureTensorAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
     ProbabilityModel *tempModelState;
    
    Block4D_ blockTemp = block_0.clone();
    double J0;
    J0 = RDtestStructureTensor(blockTemp,cui0,currGain,&tempModelState);
    delete[] tempModelState;
    double angle = blockTemp.ssi.getAngleH();
    if (blockTemp.ssi.getAngleH() != blockTemp.ssi.getAngleV()){
        std::cerr << "ERROR: THINGS ARE NOT AS THEY SEEM! COMPUTATIONS HAVE BEEN MADE I DID NOT IMPLEMENT! ZOMBIES ABOUND!" <<std::endl;
        std::exit(3);
    }
    blockTemp = block_0.clone();
    //double angle = 45;
    J0 = parallelRhoSearch(false,-1, angle, block_0, cui0, currGain, coderModelState_0);
    
    if (block_0.ssi.getAngleH() != block_0.ssi.getAngleV()){
        std::cerr << "ERROR:  ZOMBIES ABOUND! THINGS ARE NOT AS THEY SEEM! COMPUTATIONS HAVE BEEN MADE I DID NOT IMPLEMENT!" <<std::endl;
        std::cerr << block_0.ssi.getAngleH()<<" != "<<block_0.ssi.getAngleV() << " Both should be -> "<<angle<<std::endl;
        std::exit(3); 
    }

    //J0 = parallelRhoSearch(true,blockTemp.ssi.getRhoS(), angle, block_0, cui0, currGain, coderModelState_0);
    return J0;
}
double TransformPartition :: RDtestStructureTensor(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);


    //std::cout<<"TESTING  zero"<<std::endl;

    double J0 = RDtestZero(block_0,cui0,currGain,coderModelState_0);

    
    //std::cout<<"TESTed zero. J0 = "<< J0<<std::endl;

    //std::cout<<"TESTING  STRUCTURE TENSOR BlockOrig Device:"<< blockOrig.data.device()<<std::endl;
    
    //Evaluate Structure Tensor
    std::array<double,2> angles = blockOrig.computeAnglesFromStructureTensor(mDisparityRange);
    std::array<double,3> anglesToTest = {angles[0],angles[1],(angles[0]+angles[1])/2};
    //std::cout<<"Angles to test: "<<anglesToTest[0]<<" "<<anglesToTest[1]<<" "<<anglesToTest[2]<<std::endl;
    for (int i = 0; i < 3; i++){
        double J0_curr = EvaluatePartitionFixedRho(mEntropyCoder,temp_block_0,currGain,anglesToTest[i],anglesToTest[i]);
        //std::cout<<"Before: "<<temp_block_0.ssi.getAngleH()<<" "<<temp_block_0.ssi.getAngleV()<<std::endl;

        //std::cout<<i<<" - "<<anglesToTest[i]<<": "<<J0_curr<<std::endl;
       // std::cout<<temp_block_0.ssi.getAngleH()<<" "<<temp_block_0.ssi.getAngleV()<<std::endl;

        if(i == 0) cui0.setStructureTensorHorizontal({temp_block_0.ssi.getAngleH(),J0_curr});
        if(i == 1) cui0.setStructureTensorVertical({temp_block_0.ssi.getAngleH(),J0_curr});
        if(i == 2) cui0.setStructureTensorAverage({temp_block_0.ssi.getAngleH(),J0_curr});
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
    delete[] currentCoderModelState;
    //std::cout<<"TESTed STRUCTURE TENSOR. J0 = "<< J0<<std::endl;

    return J0;
}
double TransformPartition :: RDtestCovariance(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    //std::cout<<"TESTING  Covariance BlockOrig Device:"<< blockOrig.data.device()<<std::endl;

    double J0 = std::numeric_limits<double>::max();
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Evaluate Structure Tensor
    std::array<double,2> angles;
    angles[0] = blockOrig.getOrientationFromCovariance(1,mDisparityRange,true);
    angles[1] = blockOrig.getOrientationFromCovariance(1,mDisparityRange,false);
    std::array<double,3> anglesToTest = {angles[0],angles[1],(angles[0]+angles[1])/2};
    //std::cout<<"Angles to test: "<<anglesToTest[0]<<" "<<anglesToTest[1]<<" "<<anglesToTest[2]<<std::endl;
    for (int i = 0; i < 3; i++){
        double J0_curr = EvaluatePartitionFixedRho(mEntropyCoder,temp_block_0,currGain,anglesToTest[i],anglesToTest[i]);
        if(i == 0) cui0.setCovarianceHorizontal({temp_block_0.ssi.getAngleH(),J0_curr});
        if(i == 1) cui0.setCovarianceVertical({temp_block_0.ssi.getAngleH(),J0_curr});
        if(i == 2) cui0.setCovarianceAverage({temp_block_0.ssi.getAngleH(),J0_curr});
        if (J0_curr < J0){
            J0 = J0_curr;
            block_0 = temp_block_0;
            mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);

            if(i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::COVARIANCE_HORIZONTAL);
            if(i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::COVARIANCE_VERTICAL);
            if(i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::COVARIANCE_AVERAGE);
        }
        mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
        temp_block_0 = blockOrig;
    }
    delete[] currentCoderModelState;
    //std::cout<<"TESTed Covariance. J0 = "<< J0<<std::endl;


    return J0;
}

double TransformPartition :: RDtestLogdet(Block4D_& block_0,  CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    ProbabilityModel *currentCoderModelState;
    double J0 = std::numeric_limits<double>::max();
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    //Evaluate Logdet
    //std::cout<<"TESTING  LogDet Divergence BlockOrig Device:"<< blockOrig.data.device()<<std::endl;

    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    double minAngle = angleRange[0];
    double maxAngle = angleRange[1];
    double angleStep = 1;
    std::array<double,2> logdetAngles = blockOrig.logDetAngleEstimation(angleStep,mDisparityRange);
    std::array<double,3> anglesToTest = {logdetAngles[0],logdetAngles[1],(logdetAngles[0]+logdetAngles[1])/2};

    for (int i = 0; i < 3; i++){

        double J0_curr = EvaluatePartitionFixedRho(mEntropyCoder,temp_block_0,currGain,anglesToTest[i],anglesToTest[i]);
        //std::cout<<i<<" - "<<anglesToTest[i]<<": "<<J0_curr<<std::endl;
        //std::cout<<temp_block_0.ssi.getAngleH()<<" "<<temp_block_0.ssi.getAngleV()<<std::endl;
        if(i == 0) cui0.setLogdetHorizontal({temp_block_0.ssi.getAngleH(),J0_curr});
        if(i == 1) cui0.setLogdetVertical({temp_block_0.ssi.getAngleH(),J0_curr});
        if(i == 2) cui0.setLogdetAverage({temp_block_0.ssi.getAngleH(),J0_curr});
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
    delete[] currentCoderModelState;
    //std::cout<<"TESTed LogDetDivergence. J0 = "<< J0<<std::endl;

    return J0;
}
// double TransformPartition :: RDtestGridSearch(double angleStep,std::array<double,2> angleRange, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){
//     std::chrono::steady_clock::time_point begin;
//     std::chrono::steady_clock::time_point end;
//     begin = std::chrono::steady_clock::now();
//     Block4D_ blockOrig = block_0.clone();
//     Block4D_ temp_block_0 = block_0;
//     ProbabilityModel *currentCoderModelState;
//     double J0 = std::numeric_limits<double>::max();
//     mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
//     //Evaluate GS

//     double minAngle = angleRange[0];
//     double maxAngle = angleRange[1];
//     int count = 0;
//     for (double angle = minAngle; angle <=maxAngle; angle+=angleStep){ 
//         double J0_curr = EvaluatePartitionFixedRho(mEntropyCoder,temp_block_0,currGain,angle,angle);
//         //std::cout<<angle<<": "<<J0_curr<<std::endl;
//         //std::cout<<temp_block_0.ssi.getAngleH()<<" "<<temp_block_0.ssi.getAngleV()<<std::endl;
//         cui0.addGridSearchAngle(angle,J0_curr);
//         if (J0_curr < J0){
//             J0 = J0_curr;
//             block_0 = temp_block_0;
//             mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
//             cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);
            
//             //std::cout<<"tempBlock: ";
//             //tempBlock.ssi.print();
//             //block_0.ssi.print();
//         }
//         mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
//         temp_block_0 = blockOrig;

//         //std::cout<<"                       \rAngle Search: "<<count++<<"/"<<trunc((maxAngle-minAngle)/angleStep)<<std::flush;
//     }
//     //std::cout<<"TESTed GridSearch. J0 = "<< J0<<" "<<block_0.ssi.getAngleV()<<std::endl;
    
//     delete[] currentCoderModelState;
//     //std::cout<<"TESTed NonFixedRhos. J0 = "<< J0<<" "<<block_0.ssi.getAngleV()<<std::endl;
//             end = std::chrono::steady_clock::now();
//         std::chrono::duration<double, std::milli> elapsed = end - begin;
//         std::cout << "Grid Search Time: " << elapsed.count() << " ms" << std::endl;

//     return J0;

// }
double TransformPartition::RefineGridSearchAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0) {
    // This function refines the grid search and rho search for the given block.
    // It first performs a grid search to find the best angle, then refines the rhos.
    Block4D_ blockTemp = block_0.clone();
    double J0;
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    ProbabilityModel *tempModelState;
    J0 = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,&tempModelState);
    delete[] tempModelState;
    double angle = blockTemp.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-0.9,angle+0.9};
    blockTemp = block_0.clone();
    double J = RDtestGridSearch(0.1,refinementAngleRange,blockTemp,cui0,currGain,&tempModelState);  
    angle = blockTemp.ssi.getAngleH();
    delete[] tempModelState;


    //J0 = RDtestStructureTensor(blockTemp,cui0,currGain,coderModelState_0);
    //double angle = blockTemp.ssi.getAngleH();
    //double angle = 45;
    J0 = parallelRhoSearch(false,-1, angle, block_0, cui0, currGain, coderModelState_0);
    //J0 = parallelRhoSearch(true,blockTemp.ssi.getRhoS(), angle, block_0, cui0, currGain, coderModelState_0);
    return J0;
}
double TransformPartition::RefineStructureTensorAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0) {
    // This function refines the grid search and rho search for the given block.
    // It first performs a grid search to find the best angle, then refines the rhos.
    Block4D_ blockTemp = block_0.clone();
    double J0;
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    ProbabilityModel *tempModelState;
    //J0 = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,&tempModelState);
    J0 = RDtestStructureTensor(blockTemp,cui0,currGain,&tempModelState);
    delete[] tempModelState;
    double angle = blockTemp.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};
    blockTemp = block_0.clone();
    double J = RDtestGridSearch(0.1,refinementAngleRange,blockTemp,cui0,currGain,&tempModelState);  
    angle = blockTemp.ssi.getAngleH();
    delete[] tempModelState;


    //J0 = RDtestStructureTensor(blockTemp,cui0,currGain,coderModelState_0);
    //double angle = blockTemp.ssi.getAngleH();
    //double angle = 45;
    J0 = parallelRhoSearch(false,-1, angle, block_0, cui0, currGain, coderModelState_0);
    //J0 = parallelRhoSearch(true,blockTemp.ssi.getRhoS(), angle, block_0, cui0, currGain, coderModelState_0);
    return J0;
}
double TransformPartition::RDtestGridSearch(double angleStep, std::array<double, 2> angleRange, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0) {
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
    begin = std::chrono::steady_clock::now();
    //std::cout<<"Starting Grid Search with angleStep: " << angleStep << " and angleRange: [" << angleRange[0] << ", " << angleRange[1] << "]" << std::endl;

    Block4D_ blockOrig = block_0.clone();


    double minAngle = angleRange[0];
    double maxAngle = angleRange[1];
    // Ensure numSteps is not negative if angleRange is invalid
    if (minAngle > maxAngle) {
        // Handle error case or return a default value
        return std::numeric_limits<double>::max();
    }
       // A small value to counteract floating point inaccuracies.
    const double epsilon = 1e-9;
    // The resilient calculation
    int numSteps = static_cast<int>(floor((maxAngle - minAngle) / angleStep + epsilon)) + 1;

    // --- Phase 1: Allocate Storage for ALL Iterations ---
    // This is the significant memory allocation you requested.
    std::vector<double> all_J_values(numSteps);
    std::vector<double> all_angles(numSteps);
    std::vector<Block4D_> all_blocks(numSteps);
    std::vector<ProbabilityModel*> all_models(numSteps);

    for (int i = 0; i < numSteps; ++i) {
        // This is now guaranteed to be safe.
        all_blocks[i] = blockOrig.clone();
        //sanity check for all blocks:
        int* useless_ptr = all_blocks[i].data.data_ptr<int>();
        
        // We can also pre-calculate the angles here.
        all_angles[i] = minAngle + i * angleStep;
    }
    // --- Phase 2: Map (Parallel Evaluation) ---
    // This loop has no inter-thread communication or locking. Each iteration is fully independent.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numSteps; ++i) {
        int thread_id = omp_get_thread_num();
        auto& localEncoderPtr = m_encoder_pool[thread_id];
        //std::cout<<"WE ARE STARTING " <<omp_get_thread_num()<<std::endl;
        //std::cout<< "Angle Search: " << i << "/" << numSteps << "\r" << std::flush;
        Block4D_& current_block = all_blocks[i];
        double angle = all_angles[i];

        //std::cout<<"WE RUN CRITICAL!" <<omp_get_thread_num()<<std::endl;

        ProbabilityModel* loopIterationModelState;
        localEncoderPtr->GetOptimizerProbabilisticModelState(&loopIterationModelState);
        //std::cout<<"Evaluation INCOMING "<<omp_get_thread_num()<<std::endl;

        // Evaluate the cost for the current angle.
        double J0_curr = EvaluatePartitionFixedRho(*localEncoderPtr,current_block, currGain, angle, angle);
        //std::cout<<"Evaluation Successful "<<omp_get_thread_num()<<std::endl;
        // Store the complete result of this iteration without any comparisons.
        all_angles[i] = angle;
        all_J_values[i] = J0_curr;
        //all_blocks[i] = temp_block;
        all_models[i] = loopIterationModelState; // Store the pointer; will be managed later.
    } // --- End of parallel region ---

    //std::cout<<std::endl<<"We Finished it"<<std::endl;
    // --- Phase 3: Reduce (Serial Selection) ---
    // This section is executed by a single thread after the parallel work is done.
    
    // First, find the index of the best result.
    auto min_iterator = std::min_element(all_J_values.begin(), all_J_values.end());
    int best_index = std::distance(all_J_values.begin(), min_iterator);

    double J0 = all_J_values[best_index];
    
    // Set the final output parameters from the winning iteration.
    block_0 = all_blocks[best_index];
    //std::cout<<"is sgt domain: "<<block_0.sgtDomain<<std::endl;

    *coderModelState_0 = all_models[best_index]; // Transfer ownership of the winning model state.
    cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);

    // Populate the GridSearchAngle info in cui0 and clean up memory.
    for (int i = 0; i < numSteps; ++i) {
        cui0.addGridSearchAngle(all_angles[i], all_J_values[i]);
        
        // CRITICAL: Clean up all model states that were not chosen.
        // The winning model's ownership was transferred, so we must not delete it.
        if (i != best_index) {
            delete[] all_models[i];
        }
    }

    // --- Final LS Rho Evaluation ---
    // This final check remains serial, using the best result from the grid search.
    // mEntropyCoder.SetOptimizerProbabilisticModelState(initialCoderModelState);
    // Block4D_ temp_block_ls = block_0; // Use the best block from grid search
    // double J_ls = EvaluatePartitionLSRho(temp_block_ls, currGain, block_0.ssi.getAngleV(), block_0.ssi.getAngleH());

    // if (J_ls < J0) {
    //     J0 = J_ls;
    //     block_0 = temp_block_ls;
    //     // If this is better, we need to get its corresponding model state.
    //     // We must also de-allocate the previous best model from the grid search.
    //     delete[] *coderModelState_0; 
    //     mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    //     cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);
    // }
    


    return J0;
}

double TransformPartition :: parallelRhoSearch(bool searchSpace, double fixedRho,double angle, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0){

    double defaultSpaceRho = 0.99;
    double defaultAngularRho = 0.99999;
        int num_points = 16; // Number of points to sample in the rho space

    if (fixedRho < 0){
        if(searchSpace){
            fixedRho = defaultAngularRho;
        }else{
            fixedRho = defaultSpaceRho;
        }
    } 

    //std::vector<double> all_rhos = {0.6,0.7,0.8,0.9,0.95,0.99,0.995,0.999, 0.9995, 0.9999,0.99995,0.99999};
    std::vector<double> all_rhos = {};
    double minimumCovariance = 1e-2;
    double max_corr_distance = block_0.size[2]+abs(block_0.size[0] * tan(angle * M_PI / 180.0)); // Calculate the maximum size based on the angle 
    double rho_min = std::max(SgtSideInfo::MIN_RHO,std::pow(minimumCovariance, 1.0 / max_corr_distance)); // Calculate rho_min based on the maximum correlation distance
    double delta_max = 1 - rho_min;
    double delta_min = 1e-5;
    
    // std::cout<<"Maximum Correlation Distance: "<<max_corr_distance<<std::endl;
    // std::cout<<"delta_min: "<<delta_min<<", delta_max: "<<delta_max<<std::endl;
    //std::cout<<"Rhos: ";
    for (int i = 0; i < num_points; i++){
        double k = static_cast<double>(i) / (num_points - 1);
        double delta = delta_min * std::pow(delta_max / delta_min, k);
        //std::cout<<1-delta<<" ";
        all_rhos.push_back(1-delta);
    }
    //std::cout<<std::endl;

    int numSteps = all_rhos.size();

    // --- Phase 1: Allocate Storage for ALL Iterations ---
    // This is the significant memory allocation you requested.
    std::vector<double> all_J_values(numSteps);
    std::vector<Block4D_> all_blocks(numSteps);
    std::vector<ProbabilityModel*> all_models(numSteps);
    Block4D_ blockOrig = block_0.clone();

    for (int i = 0; i < numSteps; ++i) {
        // This is now guaranteed to be safe.
        all_blocks[i] = blockOrig.clone();
    }

    // --- Phase 2: Map (Parallel Evaluation) ---
    // This loop has no inter-thread communication or locking. Each iteration is fully independent.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numSteps; ++i) {
        int thread_id = omp_get_thread_num();
        auto& localEncoderPtr = m_encoder_pool[thread_id];
        //std::cout<<"WE ARE STARTING " <<omp_get_thread_num()<<std::endl;
        //std::cout<< "Angle Search: " << i << "/" << numSteps << "\r" << std::flush;
        Block4D_& current_block = all_blocks[i];
        double rho_space;
        double rho_angle;
        if(searchSpace){
            rho_space = all_rhos[i]; // Varying rho space for each iteration
            rho_angle = fixedRho; // Fixed rho angle for all iterations
        }else{
            // If not searching space, we search angle, so we fix the rho space and vary the angle.
            // This is the opposite of the previous case.
            rho_angle = all_rhos[i]; // Varying rho angle for each iteration
            rho_space = fixedRho; // Fixed rho space for all iterations
        }

        //std::cout<<"WE RUN CRITICAL!" <<omp_get_thread_num()<<std::endl;

        ProbabilityModel* loopIterationModelState;
        localEncoderPtr->GetOptimizerProbabilisticModelState(&loopIterationModelState);
        //std::cout<<"Evaluation INCOMING "<<omp_get_thread_num()<<std::endl;

        // Evaluate the cost for the current angle. encoder, Block4D_ &block_0, double currGain , double angle, double rhoAngle, double rhoSpace
        double J0_curr = EvaluatePartitionArbitraryRho(*localEncoderPtr,current_block, currGain, angle, rho_angle, rho_space);
        //std::cout<<"Evaluation Successful "<<omp_get_thread_num()<<std::endl;
        // Store the complete result of this iteration without any comparisons.
        all_J_values[i] = J0_curr;
        //all_blocks[i] = temp_block;
        all_models[i] = loopIterationModelState; // Store the pointer; will be managed later.
    } // --- End of parallel region ---

    //std::cout<<std::endl<<"We Finished it"<<std::endl;
    // --- Phase 3: Reduce (Serial Selection) ---
    // This section is executed by a single thread after the parallel work is done.
    
    // First, find the index of the best result.
    auto min_iterator = std::min_element(all_J_values.begin(), all_J_values.end());
    int best_index = std::distance(all_J_values.begin(), min_iterator);

    double J0 = all_J_values[best_index];
    
    // Set the final output parameters from the winning iteration.
    block_0 = all_blocks[best_index];
    //std::cout<<"is sgt domain: "<<block_0.sgtDomain<<std::endl;

    *coderModelState_0 = all_models[best_index]; // Transfer ownership of the winning model state.
    cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);

    //clean up memory
    for (int i = 0; i < numSteps; ++i) {        
        // CRITICAL: Clean up all model states that were not chosen.
        // The winning model's ownership was transferred, so we must not delete it.
        if (i != best_index) {
            delete[] all_models[i];
        }
    }

    return J0;

}

double TransformPartition :: RDrefineStructureTensor(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){

    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();


    //Evaluate Structure Tensor
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    
    double J = RDtestStructureTensor(blockTemp,cui0,currGain,coderModelState_0);
    
    J0 = J;
    block_0 = blockTemp;
    blockTemp = blockOrig;

    double angle = block_0.ssi.getAngleH();

    //Grid Search Refinement
    std::array<double,2> angleRange = {angle-10,angle+10};
    ProbabilityModel *tempModelState;
    J = RDtestGridSearch(refinementPrecision,angleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        coderModelState_0 = &tempModelState;
    }else{
        delete[] tempModelState;
    }
    return J0;
}

double TransformPartition :: RDrefineCovariance(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
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
    double J = RDtestCovariance(blockTemp,cui0,currGain,coderModelState_0);
    
    J0 = J;
    block_0 = blockTemp;
    blockTemp = blockOrig;

    double angle = block_0.ssi.getAngleH();

    //Grid Search Refinement
    std::array<double,2> angleRange = {angle-10,angle+10};

    J = RDtestGridSearch(refinementPrecision,angleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        coderModelState_0 = &tempModelState;
    }else{
        delete[] tempModelState;
    }
    return J0;

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

    J0 = RDtestStructureTensor(block_0,cui0,currGain,coderModelState_0);

    //std::cout<<"After structure tensor: "<<blockTemp.ssi.getAngleH()<<" SUCCESS!! ";
    for (int i = 0; i < 4;i++){
        //std::cout<<block_0.data.size(i)<<"x";
    }
    //std::cout<<std::endl;
    //std::cout<<"ST: "<<blockTemp.ssi.getAngleH()<<std::endl;
     //Evaluate Covariance
    double J = RDtestCovariance(blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        coderModelState_0 = &tempModelState;
    }else{
        delete[] tempModelState;
    }
    //std::cout<<"After Covariance: "<<blockTemp.ssi.getAngleH()<<" SUCCESS!! ";
    for (int i = 0; i < 4;i++){
        //std::cout<<block_0.data.size(i)<<"x";
    }
    //std::cout<<std::endl;
     //std::cout<<"COV: "<<blockTemp.ssi.getAngleH()<<std::endl;
     blockTemp = blockOrig;
    //Evaluate Logdet
    J = RDtestLogdet(blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        coderModelState_0 = &tempModelState;
    }else{
        delete[] tempModelState;
    }
    //std::cout<<"LogDet: "<<blockTemp.ssi.getAngleH()<<std::endl;
    //std::cout<<"After LogDet: "<<blockTemp.ssi.getAngleH()<<" SUCCESS!! ";
    for (int i = 0; i < 4;i++){
        //std::cout<<block_0.data.size(i)<<"x";
    }
    //std::cout<<std::endl;
    blockTemp = blockOrig;

    //Evaluate Grid Search
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);

    J = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,&tempModelState);
    //std::cout<<"Grid Search: "<<blockTemp.ssi.getAngleH()<<" SUCCESS!!"<<std::endl;
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        coderModelState_0 = &tempModelState;
    }else{
        delete[] tempModelState;
    }

    //std::cout<<"After Grid Search: "<<blockTemp.ssi.getAngleH()<<" SUCCESS!! ";
    for (int i = 0; i < 4;i++){
        //std::cout<<block_0.data.size(i)<<"x";
    }
    //std::cout<<std::endl;
    blockTemp = blockOrig;

    J =  EvaluatePartitionLSRho(mEntropyCoder,blockTemp,currGain,block_0.ssi.getAngleV(),block_0.ssi.getAngleH());
    //std::cout<<"ls try: "<<blockTemp.ssi.getRhoS()<<" SUCCESS!!"<<std::endl;
    if(J < J0){


        J0 = J;
        block_0 = blockTemp;
        delete[] *coderModelState_0;
        mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    }
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    delete[] currentCoderModelState;
    //std::cout<<"After LS Refinement: "<<blockTemp.ssi.getAngleH()<<" SUCCESS!! ";
    for (int i = 0; i < 4;i++){
        //std::cout<<block_0.data.size(i)<<"x";
    }
    //std::cout<<std::endl;

    //std::cout<<"GridSearch: "<<blockTemp.ssi.getAngleH()<<std::endl;
    return J0;
}
double TransformPartition :: RDrefineLogdet(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){

    double currGain = totalTransformGain();
    
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModel *tempModelState;
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();
    //Evaluate Logdet
    //Evaluate Logdet
    double J = RDtestLogdet(block_0,cui0,currGain,coderModelState_0);
    J0 = J;

    //Grid Search Refinement
    double angle = block_0.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};

    J = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,refinementAngleRange,blockTemp,cui0,currGain,&tempModelState);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        coderModelState_0 = &tempModelState;
    }else{
        delete[] tempModelState;
    }


    return J0;
}

double TransformPartition :: RDStructureTensorOrLogdet(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
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
    double J;
    if(block_0.size[2] > 8 && block_0.lightFieldPosition[2] > 0 && block_0.lightFieldPosition[2] < block_0.lightField->data.size(2)-block_0.size[2] && block_0. lightFieldPosition[3] > 0 && block_0.lightFieldPosition[3] < block_0.lightField->data.size(3)-block_0.size[3]){
        J = RDtestStructureTensor(blockTemp,cui0,currGain,&tempModelState);
        //std::cout<<"ST: "<<block_0.size[2] << block_0.lightFieldPosition[2]<<"x"<<block_0.lightFieldPosition[3]<<std::endl;
    }else{
        J = RDtestLogdet(blockTemp,cui0,currGain,&tempModelState);
        //std::cout<<"LogDet: "<<block_0.size[2] << " "<<block_0.lightFieldPosition[2]<<"x"<<block_0.lightFieldPosition[3]<<std::endl;

    }

    J0 = J;
    block_0 = blockTemp;
    mEntropyCoder.SetOptimizerProbabilisticModelState(tempModelState);
    mEntropyCoder.GetOptimizerProbabilisticModelState(coderModelState_0);
    
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    
    blockTemp = blockOrig;
   
    return J0;
}

double TransformPartition :: RDrefineGridSearch(Block4D_& block_0,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0){
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

    J = RDtestGridSearch(10,angleRange,blockTemp,cui0,currGain,&tempModelState);
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
    std::array<double,2> refinementAngleRange = {angle-9,angle+9};

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
    //std::cout<<"-3"<<std::endl;

    ProbabilityModel *currentCoderModelState;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&currentCoderModelState);
    for(int i = 0; i < m_encoder_pool.size(); i++){
        m_encoder_pool[i]->SetOptimizerProbabilisticModelState(currentCoderModelState);
    }   
    
    //partitionCodeS handles splitting in the spatial dimension, partitionCodeV handles splitting in the view dimension.
    char *partitionCodeS=NULL;
    //std::cout<<"-3"<<std::endl;

    std::array<int64_t,4> lightFieldPosition = inputBlock.lightFieldPosition;
    for(int i = 0; i < 4; i++){
        lightFieldPosition[i] += position[i]; 
    }
    //std::cout<<"-2"<<std::endl;

    std::vector<SgtSideInfo> ssiBufferS;
    std::vector<CodingUnitInfo> cuiBufferS;

    //std::cout<<"-1"<<std::endl;

    Block4D_ block_0 = inputBlock.copySubblock(length,position);
    CodingUnitInfo cui0(block_0.size,block_0.lightFieldPosition);
    //std::cout<<"-1"<<std::endl;

    //std::cout<<"Light Field Position: "<<block_0.lightFieldPosition[0]<<" "<<block_0.lightFieldPosition[1]<<" "<<block_0.lightFieldPosition[2]<<" "<<block_0.lightFieldPosition[3]<<std::endl;
    //std::cout<<"cui0 Light Field Position: "<<cui0.getLightFieldPosition()[0]<<" "<<cui0.getLightFieldPosition()[1]<<" "<<cui0.getLightFieldPosition()[2]<<" "<<cui0.getLightFieldPosition()[3]<<std::endl;
    //std::cout<<"Position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
    //std::cout<<"Light Field Position: "<<block_0.lightFieldPosition<<std::endl;

    //Block4D_ block_0(length);
    //block_0.CopySubblockFrom(inputBlock,position,{0,0,0,0});
    //std::cout<<"-1"<<std::endl;

    Block4D_ blockOrig = block_0;
    Block4D_ temp_block_0 = block_0;
    //std::cout<<"-0"<<std::endl;

    ProbabilityModel *coderModelState_0;
    double currGain = totalTransformGain();
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    //std::cout<<"-0"<<std::endl;

    // double J0 = RDtestCovariance(block_0,cui0,totalTransformGain(),&coderModelState_0);
    //double J0 = RDtestAngle(1,block_0,cui0,totalTransformGain(),&coderModelState_0);
    // double J0 = RDtestStructureTensor(block_0,cui0,totalTransformGain(),&coderModelState_0);
    // double J0 = RDgridSearchAndRhos(block_0,cui0,totalTransformGain(),&coderModelState_0);
    //double J0 = RDtestStructureTensorAndRhos(block_0,cui0,totalTransformGain(),&coderModelState_0);
    //double J0 = RefineGridSearchAndRhos(block_0,cui0,totalTransformGain(),&coderModelState_0);
    double J0 = RefineStructureTensorAndRhos(block_0,cui0,totalTransformGain(),&coderModelState_0);
    

    //double J0 = RDrefineCovariance(block_0,1,cui0, &coderModelState_0);
    //double J0 = RDtestLogdet(block_0,cui0,totalTransformGain(),&coderModelState_0);

    //double J0 = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,angleRange,block_0,cui0,currGain,&coderModelState_0);



    //double J0 = RDStructureTensorOrLogdet(block_0,cui0,&coderModelState_0);
    //double J0 = RDrefineStructureTensor(block_0,1,cui0,&coderModelState_0);
    //double J0 = RDrefineAllAngleHeuristics(block_0,cui0,&coderModelState_0);
    //std::cout<<"STARTED TESTING"<<std::endl;

    //double J0 = RDtestAllAngleHeuristics(block_0,cui0,&coderModelState_0);
    //double J0 = RDrefineGridSearch(block_0,cui0,&coderModelState_0);

    //std::cout<<"FINISHED TESTING WITH J0: "<<J0<<std::endl;

    SgtSideInfo ssi0 = block_0.ssi; 


    
    cui0.setSgtSideInfo(ssi0);
    //std::cout<<"1"<<std::endl;


    // if(lightFieldPosition[2] ==  512 && lightFieldPosition[3] == 432){
    //     if(length[2] == 16){
    //         std::cout<<"Angle CHOSEN: "<<ssi0.getAngleH()<<" "<<ssi0.getAngleV()<<std::endl;
    //         std::cout<<"Light Field Position: "<<lightFieldPosition[2]<<"x"<<lightFieldPosition[3]<<std::endl;
    //         ssi0.print();
    //     }
    // }
    
    //if(length[2] == 64) std::cout<<"Angle CHOSEN: "<<ssi0.getAngleH()<<" "<<ssi0.getAngleV()<<std::endl;
    //ssi0.print();
    //saves the resulting entropyCoder arithmetic model to model_0
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);
    //std::cout<<"HERE"<<std::endl;
    //std::cout<<"1"<<std::endl;

    double JS = -1.0;
    Block4D_ transformedBlockS(length,lightFieldPosition,inputBlock.lightField);
    transformedBlockS.emptyTransform();
    //std::cout<<"1"<<std::endl;

    //If you can split more in the spatial dimension
    if((length[3] >= 2*mlength_u_min)&&(length[2] >= 2*mlength_v_min)) {
        mDepth++;
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
        //std::cout<<"A"<<std::endl;

        new_position[3] = position[3] + length[3]/2;
        //new_lightField_position[3] = lightFieldPosition[3] + new_position[3];
        new_length[3] = length[3] - length[3]/2; 
                 
        Block4D_ transformedBlockS01(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS01.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS01, new_position, new_length,ssiBufferS01,cuiBufferS01, &partitionCodeS01);
        //std::cout<<"B"<<std::endl;

        new_position[2] = position[2] + length[2]/2;
        //new_lightField_position[2] = lightFieldPosition[2] + new_position[2];

        new_length[2] = length[2] - length[2]/2;
        
        Block4D_ transformedBlockS11(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS11.emptyTransform();
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS11, new_position, new_length, ssiBufferS10,cuiBufferS10, &partitionCodeS11);
        //std::cout<<"C"<<std::endl;

        new_position[3] = position[3];
        new_lightField_position[3] = lightFieldPosition[2] + new_position[3];
        
        new_length[3] = length[3]/2;
        
        Block4D_ transformedBlockS10(new_length,new_lightField_position,inputBlock.lightField);
        transformedBlockS10.emptyTransform();
        
        
        JS += RDoptimizeTransformStep_(inputBlock, transformedBlockS10, new_position, new_length, ssiBufferS11,cuiBufferS11, &partitionCodeS10);
        //std::cout<<"D"<<std::endl;

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
        //std::cout<<"E"<<std::endl;

        // for (int i = 0; i < 4; i++) std::cout<<transformedBlockS00.data.size(i)<<"x";
        // std::cout<<std::endl;
        // for (int i = 0; i < 4; i++) std::cout<<transformedBlockS01.data.size(i)<<"x";
        // std::cout<<std::endl;
        // for (int i = 0; i < 4; i++) std::cout<<transformedBlockS10.data.size(i)<<"x";
        // std::cout<<std::endl;
        // for (int i = 0; i < 4; i++) std::cout<<transformedBlockS11.data.size(i)<<"x";
        //std::cout<<std::endl;

        transformedBlockS = Block4D_(transformedBlockS00,transformedBlockS01,transformedBlockS10,transformedBlockS11,false);
        //std::cout<<"F"<<std::endl;

        transformedBlockS.sgtDomain = true;
        //std::cout<<transformedBlockS.data.sizes()<<std::endl;        
        
        delete [] partitionCodeS00;
        delete [] partitionCodeS01;
        delete [] partitionCodeS10;
        delete [] partitionCodeS11; 
    }
    //std::cout<<"2"<<std::endl;

    ProbabilityModel *coderModelState_s=NULL;
    mEntropyCoder.GetOptimizerProbabilisticModelState(&coderModelState_s);
    
    //Restores the current arithmetic model using current_model. 
    mEntropyCoder.SetOptimizerProbabilisticModelState(currentCoderModelState);

    //std::cout<<"lambda: "<<mLambda<<std::endl;
   
    //std::cout<<"2"<<std::endl;



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
    
    //std::cout<<"2"<<std::endl;

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
        transformedBlock = block_0.clone();
        currSsiBuffer.push_back(block_0.ssi);
        currCuiBuffer.push_back(cui0);
    }

    if(partitionCodeS != NULL) {
        delete [] partitionCodeS;
    }

    mEntropyCoder.DeleteProbabilisticModelState(currentCoderModelState);
    mEntropyCoder.DeleteProbabilisticModelState(coderModelState_0);
    mEntropyCoder.DeleteProbabilisticModelState(coderModelState_s);
    //std::cout<<"3"<<std::endl;

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

    std::cout<<"first few elements: "<<mPartitionData_.data.index({at::indexing::Slice(),at::indexing::Slice(),0,at::indexing::Slice(0,10)})<<std::endl;
    EncodePartitionStep_(position, length, scaledLambda);
}

void TransformPartition :: EncodePartitionStep_(std::array<int64_t,4> position, std::array<int64_t,4>length,  double lambda) {
    //std::cout<<mPartitionCode[mPartitionCodeIndex]<<" "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
    if(mPartitionCode[mPartitionCodeIndex] == NOSPLITFLAG) {
        std::cout<<"Size: "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        std::cout<<"Position: "<<position[0]<<"x"<<position[1]<<"x"<<position[2]<<"x"<<position[3]<<std::endl;
        mPartitionCodeIndex++;
        mEntropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
        
        mSsiBuffer[mCodingUnitIndex].print();
        //mCuiBuffer[mCodingUnitIndex].getSgtSideInfo().print();
        //std::cout<<"1"<<std::endl;

        mEntropyCoder.EncodeSSI_(mSsiBuffer[mCodingUnitIndex]);
                //std::cout<<"2"<<std::endl;

        //std::cout<<"Length:"<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"Position:"<<position[0]<<"x"<<position[1]<<"x"<<position[2]<<"x"<<position[3]<<std::endl;
        //std::array<int64_t,4> positionTransform = {0,0,position[2]*length[0],position[3]*length[1]};
        
        //std::cout<<"3"<<std::endl;

        mEntropyCoder.mSubbandLF_ = mPartitionData_.copySubblock(length,position);
        //if(length[3] == 32) std::cout<<mEntropyCoder.mSubbandLF_.validPositions.valid_positions_h % length[3];

        std::array<int64_t,4> trueLength = {mEntropyCoder.mSubbandLF_.data.size(0), mEntropyCoder.mSubbandLF_.data.size(1), mEntropyCoder.mSubbandLF_.data.size(2), mEntropyCoder.mSubbandLF_.data.size(3)};
        //std::cout<<"first few elements block: "<<mPartitionData_.data.index({at::indexing::Slice(0),at::indexing::Slice(0),0,at::indexing::Slice(0,3)})<<std::endl;

        //std::cout<<"first few elements subblock: "<<mEntropyCoder.mSubbandLF_.data.index({at::indexing::Slice(0),at::indexing::Slice(0),0,at::indexing::Slice(0,3)})<<std::endl;

        std::cout<<"mPartitionData_.size: "<<mPartitionData_.data.size(0)<<"x"<<mPartitionData_.data.size(1)<<"x"<<mPartitionData_.data.size(2)<<"x"<<mPartitionData_.data.size(3)<<std::endl;
        std::cout<<"mEntropyCoder.mSubbandLF_.size: "<<mEntropyCoder.mSubbandLF_.data.size(0)<<"x"<<mEntropyCoder.mSubbandLF_.data.size(1)<<"x"<<mEntropyCoder.mSubbandLF_.data.size(2)<<"x"<<mEntropyCoder.mSubbandLF_.data.size(3)<<std::endl;
        std::cout<<"length: "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        std::cout<<"trueLength: "<<trueLength[0]<<"x"<<trueLength[1]<<"x"<<trueLength[2]<<"x"<<trueLength[3]<<std::endl;
        std::cout<<"lf position: "<<mEntropyCoder.mSubbandLF_.lightFieldPosition[0]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[1]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[2]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[3]<<std::endl;
        //std::cout<<"4"<<std::endl;
        if(trueLength[2] * trueLength[3] > 0) mEntropyCoder.encodeSubblockFromPool(trueLength, {0,0,0,0},mEntropyCoder.mSuperiorBitPlane,mLambda);
        //std::cout<<"5"<<std::endl;

        //mEntropyCoder.EncodeSubblock_(lambda);
        
        
        //std::cout<<"Light Field Position: "<<mEntropyCoder.mSubbandLF_.lightFieldPosition[2]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[3]<<std::endl;

        // if(mEntropyCoder.mSubbandLF_.lightFieldPosition[2] <=  512 && mEntropyCoder.mSubbandLF_.lightFieldPosition[3] <= 432){
        //     //if(length[2] == 16){
        //         std::cout<<"ENCODED"<<std::endl;
        //         std::cout<<"Light Field Position: "<<mEntropyCoder.mSubbandLF_.lightFieldPosition[2]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[3]<<std::endl;
        //         mSsiBuffer[mCodingUnitIndex].print();
        //     //}
        // }
        
        double weight = totalTransformGain();
        double distortion = (double) mEntropyCoder.mDistortion/(weight*weight);
        mCodingPartitionInfo.incrementTotalDistortion(distortion);
        double mse = distortion/(length[0]*length[1]*length[2]*length[3]);
        mCodingPartitionInfo.incrementTotalSize(mEntropyCoder.mRate * (length[0]*length[1]*length[2]*length[3]));
        //std::cout<<mCodingPartitionInfo.getTotalDistortion()<<" "<<mCodingPartitionInfo.getTotalSize()<<std::endl;
        //std::cout<<"mRate = "<<mEntropyCoder.mRate<<" mDistortion: "<<(double) mEntropyCoder.mDistortion/(weight*weight)<<std::endl;
        double psnr = 10 * log10((1024*1024)/mse);
        mCuiBuffer[mCodingUnitIndex].setPSNR(psnr);
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


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
    //std::cout<<"Using "<<m_encoder_pool.size()<<" threads for encoding."<<std::endl;
    //std::cout<<"MBP : "<<mEntropyCoder.mInferiorBitPlane<<std::endl;

    mPartitionCode = "";
    mlength_t_min = minLength[0];
    mlength_s_min = minLength[1];
    mlength_v_min = minLength[2];
    mlength_u_min = minLength[3];

    //std::cout<<"We gucci"<<std::endl;
    
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
    // No-op since mPartitionCode is a std::string
}
double TransformPartition :: totalTransformGain(void){
    return mGain*sqrt(mMaxSize[0]*mMaxSize[1]*mMaxSize[2]*mMaxSize[3]);

}
void TransformPartition :: RDoptimizeTransform_(Block4D_ &inputBlock, double lambda){
    mEntropyCoder.RestartProbabilisticModel();
    mEntropyCoder.LoadOptimizerState();
    for(int i = 0; i < m_encoder_pool.size(); i++) {
        m_encoder_pool[i]->RestartProbabilisticModel();
        m_encoder_pool[i]->LoadOptimizerState();
    }
    this->mMaxSize = inputBlock.size;
    inputBlock.data = inputBlock.data.contiguous();
    if(!mSsiBuffer.empty()) mSsiBuffer.clear();
    if(!mCuiBuffer.empty()) mCuiBuffer.clear();

    mCodingUnitIndex = 0;
    mPartitionCode.clear();
    mEvaluateOptimumBitPlane = 1;
    mPartitionData_ = Block4D_(inputBlock.size,inputBlock.lightFieldPosition,inputBlock.lightField);
    double scaledLambda = lambda;
    for (int i = 0; i < 4; i++){
        scaledLambda *= inputBlock.size[i];
    }
    mLambda = scaledLambda;
    // 4.5. Calculate the optimal minimum bitplane for this LightField
    getOptimalMinimumBitPlane(inputBlock);

    // 5. The Main Optimization Loop
    // We create an empty collage to hold the "winning" partition structure
    BlockCollage finalCollage;
    
    // Kick off the recursion starting at (0,0,0,0) with the full block size
    mLagrangianCost = RDoptimizeTransformStep(
        inputBlock, 
        finalCollage, 
        {0, 0, 0, 0}, 
        inputBlock.size, 
        mPartitionCode
    );

    // 6. Store the Result
    // Instead of mPartitionData_ = block, we store the winning collage
    mPartitionCollage = std::move(finalCollage);
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

double TransformPartition::EvaluatePartitionArbitraryRho(
    Hierarchical4DEncoder& encoder, 
    Block4D_ &block_0, 
    double currGain, 
    double angle, 
    double rhoAngle, 
    double rhoSpace,
    ProbabilityModelCollection& outModel // <-- Added collection
){
    block_0.ssi = SgtSideInfo(angle, angle, mDisparityRange);
    block_0.ssi.setAngularRhos(rhoAngle, rhoAngle);
    block_0.ssi.setSpatialRhos(rhoSpace, rhoSpace);
    
    // Pass the collection directly into the sandbox
    return EvaluatePartition_(encoder, block_0, currGain, outModel);
}

double TransformPartition::EvaluatePartitionFixedRho(
    Hierarchical4DEncoder& encoder, 
    Block4D_ &block_0, 
    double currGain, 
    double angleV, 
    double angleH,
    ProbabilityModelCollection& outModel // <-- Added collection
){
    block_0.ssi = SgtSideInfo(angleV, angleH, mDisparityRange);
    
    // Pass the collection directly into the sandbox
    return EvaluatePartition_(encoder, block_0, currGain, outModel);
}

double TransformPartition::EvaluatePartitionLSRho(
    Hierarchical4DEncoder& encoder,
    Block4D_ &block_0, 
    double currGain, 
    double angleV, 
    double angleH,
    ProbabilityModelCollection& outModel // <-- Added collection
){
    block_0.ssi = SgtSideInfo(angleV, angleH, mDisparityRange);
    block_0.ssi.estimateRhos(block_0, -1);
    
    // Pass the collection directly into the sandbox
    return EvaluatePartition_(encoder, block_0, currGain, outModel);
}
double TransformPartition::EvaluatePartition_(
    Hierarchical4DEncoder& encoder, 
    Block4D_ &block_0, 
    double currGain, 
    ProbabilityModelCollection& outModel_0 // Pass in a collection to hold the result
) {
    // 1. THE SANDBOX ENTRANCE: Snapshot the encoder's original state
    ProbabilityModelCollection initialState = encoder.GetOptimizerSnapshot();

    // 2. RUN THE SIMULATION: Perform transform and tree building 
    // (This mutates the encoder's internal probability models)
    block_0.sgtTransform(currGain);
    SgtSideInfo ssi0 = block_0.ssi; 
    encoder.mSubbandLF_ = block_0;

    std::array<int64_t,4> lengthTransform = {
        encoder.mSubbandLF_.data.size(0), 
        encoder.mSubbandLF_.data.size(1), 
        encoder.mSubbandLF_.data.size(2), 
        encoder.mSubbandLF_.data.size(3)
    };

    double J0 = encoder.build_optimal_tree_from_pool(
        lengthTransform, 
        {0,0,0,0}, 
        encoder.mSuperiorBitPlane, 
        mLambda
    );

    int RHO_PRECISION = block_0.ssi.getRhoPrecision();
    int DISP_PRECISION = block_0.ssi.getAnglePrecision();
    J0 += RHO_PRECISION * 4 * mLambda + DISP_PRECISION * mLambda;

    // 3. CAPTURE THE RESULTS: Hand the mutated state back to the caller
    outModel_0 = encoder.GetOptimizerSnapshot();

    // 4. THE SANDBOX EXIT: Restore the encoder to its pristine initial state
    encoder.RestoreOptimizerState(initialState);

    return J0;
}

double TransformPartition::RDtestAngle(double angle, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    // The wrapper handles the sandboxing. We just pass outModel directly to it!
    double J0 = EvaluatePartitionFixedRho(mEntropyCoder, block_0, currGain, angle, angle, outModel);
    return J0;
}

double TransformPartition::RDtestZero(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    cui0.setAngleHeuristicUsed(AngleHeuristic::ZERO);
    return RDtestAngle(0, block_0, cui0, currGain, outModel);
}

double TransformPartition::RDgridSearchAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    Block4D_ blockTemp = block_0.clone();
    std::array<double, 2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    
    ProbabilityModelCollection tempModel; // No new/delete!
    double J0 = RDtestGridSearch(1, angleRange, blockTemp, cui0, currGain, tempModel);
    
    double angle = blockTemp.ssi.getAngleH();
    
    J0 = parallelRhoSearch(false, -1, angle, block_0, cui0, currGain, outModel);
    return J0;
}

double TransformPartition::RDtestStructureTensorAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    Block4D_ blockTemp = block_0.clone();
    ProbabilityModelCollection tempModel;
    
    double J0 = RDtestStructureTensor(blockTemp, cui0, currGain, tempModel);
    
    if (blockTemp.ssi.getAngleH() != blockTemp.ssi.getAngleV()){
        std::cerr << "ERROR: THINGS ARE NOT AS THEY SEEM! ZOMBIES ABOUND!" << std::endl;
        std::exit(3);
    }
    
    double angle = blockTemp.ssi.getAngleH();
    blockTemp = block_0.clone();
    
    J0 = parallelRhoSearch(false, -1, angle, block_0, cui0, currGain, outModel);
    
    if (block_0.ssi.getAngleH() != block_0.ssi.getAngleV()){
        std::cerr << "ERROR: ZOMBIES ABOUND!" << std::endl;
        std::exit(3); 
    }
    return J0;
}
double TransformPartition::RDtestStructureTensor(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;

    double J0 = RDtestZero(block_0, cui0, currGain, outModel);
    
    std::array<double, 2> angles = blockOrig.computeAnglesFromStructureTensor(mDisparityRange);
    std::array<double, 3> anglesToTest = {angles[0], angles[1], (angles[0]+angles[1])/2};
    
    ProbabilityModelCollection currModel;
    
    for (int i = 0; i < 3; i++) {
        double J0_curr = EvaluatePartitionFixedRho(mEntropyCoder, temp_block_0, currGain, anglesToTest[i], anglesToTest[i], currModel);
        
        if(i == 0) cui0.setStructureTensorHorizontal({temp_block_0.ssi.getAngleH(), J0_curr});
        if(i == 1) cui0.setStructureTensorVertical({temp_block_0.ssi.getAngleH(), J0_curr});
        if(i == 2) cui0.setStructureTensorAverage({temp_block_0.ssi.getAngleH(), J0_curr});
        
        if (J0_curr < J0) {
            J0 = J0_curr;
            block_0 = temp_block_0;
            outModel = currModel; // Just a simple struct copy!
            
            if(i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_HORIZONTAL);
            if(i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_VERTICAL);
            if(i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::STRUCTURE_TENSOR_AVERAGE);
        }
        temp_block_0 = blockOrig;
    }
    return J0;
}

double TransformPartition::RDtestCovariance(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;

    double J0 = std::numeric_limits<double>::max();
    
    // Evaluate Covariance
    std::array<double, 2> angles;
    angles[0] = blockOrig.getOrientationFromCovariance(1, mDisparityRange, true);
    angles[1] = blockOrig.getOrientationFromCovariance(1, mDisparityRange, false);
    std::array<double, 3> anglesToTest = {angles[0], angles[1], (angles[0] + angles[1]) / 2};
    
    ProbabilityModelCollection currModel;

    for (int i = 0; i < 3; i++) {
        double J0_curr = EvaluatePartitionFixedRho(mEntropyCoder, temp_block_0, currGain, anglesToTest[i], anglesToTest[i], currModel);
        
        if (i == 0) cui0.setCovarianceHorizontal({temp_block_0.ssi.getAngleH(), J0_curr});
        if (i == 1) cui0.setCovarianceVertical({temp_block_0.ssi.getAngleH(), J0_curr});
        if (i == 2) cui0.setCovarianceAverage({temp_block_0.ssi.getAngleH(), J0_curr});
        
        if (J0_curr < J0) {
            J0 = J0_curr;
            block_0 = temp_block_0;
            outModel = currModel; // Simple struct copy saves the state

            if (i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::COVARIANCE_HORIZONTAL);
            if (i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::COVARIANCE_VERTICAL);
            if (i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::COVARIANCE_AVERAGE);
        }
        temp_block_0 = blockOrig; // Reset block for the next iteration
    }

    return J0;
}


double TransformPartition::RDtestLogdet(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    Block4D_ blockOrig = block_0.clone();
    Block4D_ temp_block_0 = block_0;
    
    double J0 = std::numeric_limits<double>::max();
    
    // Evaluate Logdet
    double angleStep = 1;
    std::array<double, 2> logdetAngles = blockOrig.logDetAngleEstimation(angleStep, mDisparityRange);
    std::array<double, 3> anglesToTest = {logdetAngles[0], logdetAngles[1], (logdetAngles[0] + logdetAngles[1]) / 2};

    ProbabilityModelCollection currModel;

    for (int i = 0; i < 3; i++) {
        double J0_curr = EvaluatePartitionFixedRho(mEntropyCoder, temp_block_0, currGain, anglesToTest[i], anglesToTest[i], currModel);
        
        if (i == 0) cui0.setLogdetHorizontal({temp_block_0.ssi.getAngleH(), J0_curr});
        if (i == 1) cui0.setLogdetVertical({temp_block_0.ssi.getAngleH(), J0_curr});
        if (i == 2) cui0.setLogdetAverage({temp_block_0.ssi.getAngleH(), J0_curr});
        
        if (J0_curr < J0) {
            J0 = J0_curr;
            block_0 = temp_block_0;
            outModel = currModel; // Simple struct copy saves the state
            
            if (i == 0) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_HORIZONTAL);
            if (i == 1) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_VERTICAL);
            if (i == 2) cui0.setAngleHeuristicUsed(AngleHeuristic::LOGDET_AVERAGE);
        }
        
        temp_block_0 = blockOrig; // Reset block for the next iteration
    }

    return J0;
}


double TransformPartition::RefineGridSearchAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    // This function refines the grid search and rho search for the given block.
    // It first performs a grid search to find the best angle, then refines the rhos.
    Block4D_ blockTemp = block_0.clone();
    double J0;
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    ProbabilityModelCollection tempModel;
    J0 = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,tempModel);
    
    double angle = blockTemp.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-0.9,angle+0.9};
    blockTemp = block_0.clone();
    double J = RDtestGridSearch(0.1,refinementAngleRange,blockTemp,cui0,currGain,tempModel);  
    angle = blockTemp.ssi.getAngleH();

    //J0 = RDtestStructureTensor(blockTemp,cui0,currGain,coderModelState_0);
    //double angle = blockTemp.ssi.getAngleH();
    //double angle = 45;
    J0 = parallelRhoSearch(false,-1, angle, block_0, cui0, currGain, outModel);
    //J0 = parallelRhoSearch(true,blockTemp.ssi.getRhoS(), angle, block_0, cui0, currGain, coderModelState_0);
    return J0;
}
void TransformPartition::CommitOptimizerState(const ProbabilityModelCollection& winningState) {
    // 1. Update the master timeline
    mEntropyCoder.RestoreOptimizerState(winningState);
    
    // 2. Broadcast the master timeline to the entire thread pool
    for (int i = 0; i < m_encoder_pool.size(); ++i) {
        m_encoder_pool[i]->RestoreOptimizerState(winningState);
    }
}
double TransformPartition::RDtestGridSearch(double angleStep, std::array<double, 2> angleRange, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    Block4D_ blockOrig = block_0.clone();
    double minAngle = angleRange[0];
    double maxAngle = angleRange[1];
    
    if (minAngle > maxAngle) return std::numeric_limits<double>::max();
    
    const double epsilon = 1e-9;
    int numSteps = static_cast<int>(floor((maxAngle - minAngle) / angleStep + epsilon)) + 1;

    std::vector<double> all_J_values(numSteps);
    std::vector<double> all_angles(numSteps);
    std::vector<Block4D_> all_blocks(numSteps);
    // Look at this! Just a vector of structs. Automatically initialized safely.
    std::vector<ProbabilityModelCollection> all_models(numSteps); 

    for (int i = 0; i < numSteps; ++i) {
        all_blocks[i] = blockOrig.clone();
        all_angles[i] = minAngle + i * angleStep;
    }

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numSteps; ++i) {
        int thread_id = omp_get_thread_num();
        auto& localEncoderPtr = m_encoder_pool[thread_id];
        Block4D_& current_block = all_blocks[i];
        double angle = all_angles[i];

        // Evaluate puts the result straight into our vector slot
        all_J_values[i] = EvaluatePartitionFixedRho(*localEncoderPtr, current_block, currGain, angle, angle, all_models[i]);
    }

    // --- Phase 3: Reduce ---
    auto min_iterator = std::min_element(all_J_values.begin(), all_J_values.end());
    int best_index = std::distance(all_J_values.begin(), min_iterator);

    double J0 = all_J_values[best_index];
    block_0 = all_blocks[best_index];
    
    // Just copy the winning struct to the output reference!
    outModel = all_models[best_index]; 
    cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);

    for (int i = 0; i < numSteps; ++i) {
        cui0.addGridSearchAngle(all_angles[i], all_J_values[i]);
    }
    // No delete[] loop needed! The vector cleans itself up entirely automatically.
    
    return J0;
}



double TransformPartition::parallelRhoSearch(
    bool searchSpace, 
    double fixedRho, 
    double angle, 
    Block4D_& block_0, 
    CodingUnitInfo& cui0, 
    double currGain, 
    ProbabilityModelCollection& outModel // <-- 1. Replaced the double pointer
) {
    double defaultSpaceRho = 0.99;
    double defaultAngularRho = 0.99999;
    int num_points = 16; // Number of points to sample in the rho space

    if (fixedRho < 0) {
        if (searchSpace) {
            fixedRho = defaultAngularRho;
        } else {
            fixedRho = defaultSpaceRho;
        }
    } 

    std::vector<double> all_rhos = {};
    double minimumCovariance = 1e-2;
    double max_corr_distance = block_0.size[2] + abs(block_0.size[0] * tan(angle * M_PI / 180.0)); 
    double rho_min = std::max(SgtSideInfo::MIN_RHO, std::pow(minimumCovariance, 1.0 / max_corr_distance)); 
    double delta_max = 1 - rho_min;
    double delta_min = 1e-5;
    
    for (int i = 0; i < num_points; i++) {
        double k = static_cast<double>(i) / (num_points - 1);
        double delta = delta_min * std::pow(delta_max / delta_min, k);
        all_rhos.push_back(1 - delta);
    }

    int numSteps = all_rhos.size();

    // --- Phase 1: Allocate Storage for ALL Iterations ---
    std::vector<double> all_J_values(numSteps);
    std::vector<Block4D_> all_blocks(numSteps);
    
    // 2. Safely initialize a vector of our new collections. 
    // They are fully zeroed and ready to capture the thread states!
    std::vector<ProbabilityModelCollection> all_models(numSteps); 
    
    Block4D_ blockOrig = block_0.clone();

    for (int i = 0; i < numSteps; ++i) {
        all_blocks[i] = blockOrig.clone();
    }

    // --- Phase 2: Map (Parallel Evaluation) ---
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numSteps; ++i) {
        int thread_id = omp_get_thread_num();
        auto& localEncoderPtr = m_encoder_pool[thread_id];
        
        Block4D_& current_block = all_blocks[i];
        double rho_space;
        double rho_angle;
        
        if (searchSpace) {
            rho_space = all_rhos[i];
            rho_angle = fixedRho;
        } else {
            rho_angle = all_rhos[i];
            rho_space = fixedRho;
        }

        // 3. Evaluate the cost and capture the resulting models directly into this thread's designated slot!
        // Because of the sandbox in EvaluatePartition_, *localEncoderPtr is left completely unharmed.
        double J0_curr = EvaluatePartitionArbitraryRho(
            *localEncoderPtr, 
            current_block, 
            currGain, 
            angle, 
            rho_angle, 
            rho_space, 
            all_models[i] // <-- Passes the exact struct we want populated
        );
        
        // Store the result
        all_J_values[i] = J0_curr;
    } // --- End of parallel region ---

    // --- Phase 3: Reduce (Serial Selection) ---
    auto min_iterator = std::min_element(all_J_values.begin(), all_J_values.end());
    int best_index = std::distance(all_J_values.begin(), min_iterator);

    double J0 = all_J_values[best_index];
    
    // Set the final output parameters from the winning iteration
    block_0 = all_blocks[best_index];
    
    // 4. Copy the winning state out to the caller
    outModel = all_models[best_index]; 
    
    cui0.setAngleHeuristicUsed(AngleHeuristic::GRID_SEARCH);

    // 5. No cleanup loop needed here. 
    // The vector of ProbabilityModelCollections goes out of scope and cleans itself perfectly.

    return J0;
}


double TransformPartition::RefineStructureTensorAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel) {
    // This function refines the grid search and rho search for the given block.
    // It first performs a grid search to find the best angle, then refines the rhos.
    Block4D_ blockTemp = block_0.clone();
    double J0;
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    ProbabilityModelCollection tempModel;
    //J0 = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,tempModel);
    J0 = RDtestStructureTensor(blockTemp,cui0,currGain,tempModel);
    
    double angle = blockTemp.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};
    blockTemp = block_0.clone();
    double J = RDtestGridSearch(0.1,refinementAngleRange,blockTemp,cui0,currGain,tempModel);  
    angle = blockTemp.ssi.getAngleH();


    //J0 = RDtestStructureTensor(blockTemp,cui0,currGain,coderModelState_0);
    //double angle = blockTemp.ssi.getAngleH();
    //double angle = 45;
    J0 = parallelRhoSearch(false,-1, angle, block_0, cui0, currGain, outModel);
    //J0 = parallelRhoSearch(true,blockTemp.ssi.getRhoS(), angle, block_0, cui0, currGain, coderModelState_0);
    return J0;
}
double TransformPartition :: RDrefineStructureTensor(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModelCollection& outModel){

    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();


    //Evaluate Structure Tensor
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    
    double J = RDtestStructureTensor(blockTemp,cui0,currGain,outModel);
    
    J0 = J;
    block_0 = blockTemp;
    blockTemp = blockOrig;

    double angle = block_0.ssi.getAngleH();

    //Grid Search Refinement
    std::array<double,2> angleRange = {angle-10,angle+10};
    ProbabilityModelCollection tempModel;
    J = RDtestGridSearch(refinementPrecision,angleRange,blockTemp,cui0,currGain,tempModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }
    return J0;
}

double TransformPartition :: RDrefineCovariance(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModelCollection& outModel){
    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();


    //Evaluate Structure Tensor
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModelCollection tempModel;
    double J = RDtestCovariance(blockTemp,cui0,currGain,outModel);
    
    J0 = J;
    block_0 = blockTemp;
    blockTemp = blockOrig;

    double angle = block_0.ssi.getAngleH();

    //Grid Search Refinement
    std::array<double,2> angleRange = {angle-10,angle+10};

    J = RDtestGridSearch(refinementPrecision,angleRange,blockTemp,cui0,currGain,tempModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }
    return J0;
}
double TransformPartition::RDtestAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModelCollection& outModel) {
    double currGain = totalTransformGain();
    double J0 = std::numeric_limits<double>::max();

    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;
    ProbabilityModelCollection tempModel; // Local buffer for tests

    // 1. Structure Tensor
    J0 = RDtestStructureTensor(block_0, cui0, currGain, outModel);
    
    // 2. Covariance
    blockTemp = blockOrig;
    double J = RDtestCovariance(blockTemp, cui0, currGain, tempModel);
    if(J < J0) {
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    // 3. Logdet
    blockTemp = blockOrig;
    J = RDtestLogdet(blockTemp, cui0, currGain, tempModel);
    if(J < J0) {
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    // 4. Grid Search
    blockTemp = blockOrig;
    std::array<double, 2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);
    J = RDtestGridSearch(1, angleRange, blockTemp, cui0, currGain, tempModel);
    if(J < J0) {
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    // 5. LS Rho
    blockTemp = blockOrig;
    J = EvaluatePartitionLSRho(mEntropyCoder, blockTemp, currGain, block_0.ssi.getAngleV(), block_0.ssi.getAngleH(), tempModel);
    if(J < J0) {
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    // No state to restore, no pointers to delete. 
    return J0;
}
double TransformPartition :: RDrefineLogdet(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModelCollection& outModel){

    double currGain = totalTransformGain();
    
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModelCollection tempModel;
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();
    //Evaluate Logdet
    //Evaluate Logdet
    double J = RDtestLogdet(block_0,cui0,currGain,outModel);
    J0 = J;

    //Grid Search Refinement
    double angle = block_0.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};

    J = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,refinementAngleRange,blockTemp,cui0,currGain,tempModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }


    return J0;
}

double TransformPartition :: RDStructureTensorOrLogdet(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModelCollection& outModel){
    double currGain = totalTransformGain();
    
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;
    double J0 = std::numeric_limits<double>::max();
    //Evaluate Logdet
    //Evaluate Logdet
    double J;
    if(block_0.size[2] > 8 && block_0.lightFieldPosition[2] > 0 && block_0.lightFieldPosition[2] < block_0.lightField->data.size(2)-block_0.size[2] && block_0. lightFieldPosition[3] > 0 && block_0.lightFieldPosition[3] < block_0.lightField->data.size(3)-block_0.size[3]){
        J = RDtestStructureTensor(blockTemp,cui0,currGain,outModel);
        //std::cout<<"ST: "<<block_0.size[2] << block_0.lightFieldPosition[2]<<"x"<<block_0.lightFieldPosition[3]<<std::endl;
    }else{
        J = RDtestLogdet(blockTemp,cui0,currGain,outModel);
        //std::cout<<"LogDet: "<<block_0.size[2] << " "<<block_0.lightFieldPosition[2]<<"x"<<block_0.lightFieldPosition[3]<<std::endl;

    }

    J0 = J;
    block_0 = blockTemp;
    
    blockTemp = blockOrig;
   
    return J0;
}

double TransformPartition :: RDrefineGridSearch(Block4D_& block_0,CodingUnitInfo& cui0, ProbabilityModelCollection& outModel){
    double currGain = totalTransformGain();
    
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModelCollection tempModel;
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();
    //Evaluate Grid Search
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);

    double J = RDtestGridSearch(1,angleRange,blockTemp,cui0,currGain,outModel);
    
    J0 = J;
    block_0 = blockTemp;
    
    blockTemp = blockOrig;
    //Grid Search Refinement
    double angle = block_0.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-1,angle+1};

    J = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,refinementAngleRange,blockTemp,cui0,currGain,tempModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    return J0;
}


double TransformPartition :: RDrefineAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModelCollection& outModel){
    double currGain = totalTransformGain();
    //std::cout<<"length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<"Curr Gain: "<<currGain<<std::endl;

    double J0 = std::numeric_limits<double>::max();


    //Evaluate Structure Tensor
    Block4D_ blockOrig = block_0.clone();
    Block4D_ blockTemp = block_0;

    ProbabilityModelCollection tempModel;
    double J;
    J = RDtestStructureTensor(blockTemp,cui0,currGain,outModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
    }

    blockTemp = blockOrig;
    //Evaluate Logdet
    J = RDtestLogdet(blockTemp,cui0,currGain,tempModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    blockTemp = blockOrig;

    //Evaluate Grid Search
    std::array<double,2> angleRange = SgtSideInfo::angleRangeFromDispRange(mDisparityRange);

    J = RDtestGridSearch(10,angleRange,blockTemp,cui0,currGain,tempModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    blockTemp = blockOrig;
    //Grid Search Refinement
    double angle = block_0.ssi.getAngleH();
    std::array<double,2> refinementAngleRange = {angle-9,angle+9};

    J = RDtestGridSearch(SgtSideInfo::PRECISION_ANGLE,refinementAngleRange,blockTemp,cui0,currGain,tempModel);
    if(J < J0){
        J0 = J;
        block_0 = blockTemp;
        outModel = tempModel;
    }

    return J0;
}

std::array<int64_t,4> TransformPartition :: adjustLFPositionForSubblock(std::array<int64_t,4> parentLFPos, std::array<int64_t,4> subblockOffset) const{
    // This function adjusts the light field position for a given subblock based on its offset from the parent block.
    std::array<int64_t,4> adjustedPos;
    for(int i = 0; i < 4; i++) {
        adjustedPos[i] = parentLFPos[i] + subblockOffset[i];
    }
    return adjustedPos;
}

double TransformPartition::RDoptimizeTransformStep(const Block4D_ &inputBlock, BlockCollage &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length, std::string& partitionCode) {
    
    std::array<int64_t,4> lightFieldPosition = adjustLFPositionForSubblock(inputBlock.lightFieldPosition, position);
    
    // 1. Snapshot the original state before we test the recursive split branch
    ProbabilityModelCollection originalState = mEntropyCoder.GetOptimizerSnapshot();
    
    Block4D_ block_0 = inputBlock.copySubblock(length, position);

    double currGain = totalTransformGain();
    
    // 2. Evaluate NO SPLIT (J0) 
    // We pass our state collection directly in. Thanks to the sandbox, mEntropyCoder remains untouched!
    ProbabilityModelCollection state0;
    CodingUnitInfo cui0(block_0.size,block_0.lightFieldPosition); 
    
    double J0 = RefineStructureTensorAndRhos(block_0, cui0, currGain, state0);

    // 3. Evaluate SPLIT (JS)
    double JS = -1.0;
    std::string partitionCodeS;
    BlockCollage transformedBlockS;
    ProbabilityModelCollection stateS;

    if ((length[3] >= 2 * mlength_u_min) && (length[2] >= 2 * mlength_v_min)) {
        // splitInFour will recursively call RDoptimizeTransformStep and mutate mEntropyCoder
        JS = splitInFour(inputBlock, position, length, transformedBlockS, partitionCodeS);
        
        // Capture the resulting state of the split branch
        stateS = mEntropyCoder.GetOptimizerSnapshot();
        
        // Restore the pristine original state before we make our final decision
        mEntropyCoder.RestoreOptimizerState(originalState);
    }

    // 4. Add Flag Costs
    if (J0 > 0) 
        J0 += 1.0 * mLambda;
    if (JS > 0)
        JS += 2.0 * mLambda;
    
    // 5. Decide the Winner
    bool intraview_split = (JS >= 0 && JS < J0);
    double optimumJ = 0;

    if (intraview_split) {
        optimumJ = JS;
        
        // Append flag and the sub-partition codes to the master string
        partitionCode += (char)INTRAVIEWSPLITFLAG;
        partitionCode += partitionCodeS; 
        
        // Apply the winning Split state to the encoder
        CommitOptimizerState(stateS);
        
        transformedBlock = std::move(transformedBlockS);
    } else {
        optimumJ = J0;   
        
        partitionCode += (char)NOSPLITFLAG;
        
        // Apply the winning No-Split state to the encoder
        CommitOptimizerState(state0);
        
        transformedBlock = BlockCollage(std::move(block_0));

    }  
    
    return optimumJ;     
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

    //std::cout<<"first few elements: "<<mPartitionData_.data.index({at::indexing::Slice(),at::indexing::Slice(),0,at::indexing::Slice(0,10)})<<std::endl;
    EncodePartitionStep_(position, length, scaledLambda);
}

void TransformPartition :: EncodePartitionStep_(std::array<int64_t,4> position, std::array<int64_t,4>length,  double lambda) {
    //std::cout<<mPartitionCode[mPartitionCodeIndex]<<" "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
    if(mPartitionCode[mPartitionCodeIndex] == NOSPLITFLAG) {
        std::cout<<"Size: "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        std::cout<<"Position: "<<position[0]<<"x"<<position[1]<<"x"<<position[2]<<"x"<<position[3]<<std::endl;

#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] mPartitionCodeIndex: " << mPartitionCodeIndex << ", mCodingUnitIndex: " << mCodingUnitIndex << std::endl;
#endif

        mPartitionCodeIndex++;

#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Encoding partition flag" << std::endl;
#endif
        mEntropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);
        
#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Accessing mSsiBuffer[" << mCodingUnitIndex << "]. Size: " << mSsiBuffer.size() << std::endl;
#endif
        mSsiBuffer[mCodingUnitIndex].print();
        //mCuiBuffer[mCodingUnitIndex].getSgtSideInfo().print();
        //std::cout<<"1"<<std::endl;

#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Encoding SSI" << std::endl;
#endif
        mEntropyCoder.EncodeSSI_(mSsiBuffer[mCodingUnitIndex]);
                //std::cout<<"2"<<std::endl;

        //std::cout<<"Length:"<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        //std::cout<<"Position:"<<position[0]<<"x"<<position[1]<<"x"<<position[2]<<"x"<<position[3]<<std::endl;
        //std::array<int64_t,4> positionTransform = {0,0,position[2]*length[0],position[3]*length[1]};
        
        //std::cout<<"3"<<std::endl;

#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Copying subblock" << std::endl;
#endif
        mEntropyCoder.mSubbandLF_ = mPartitionData_.copySubblock(length,position);
        //if(length[3] == 32) std::cout<<mEntropyCoder.mSubbandLF_.validPositions.valid_positions_h % length[3];

#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Initializing trueLength" << std::endl;
#endif
        std::array<int64_t,4> trueLength = {mEntropyCoder.mSubbandLF_.data.size(0), mEntropyCoder.mSubbandLF_.data.size(1), mEntropyCoder.mSubbandLF_.data.size(2), mEntropyCoder.mSubbandLF_.data.size(3)};
        //std::cout<<"first few elements block: "<<mPartitionData_.data.index({at::indexing::Slice(0),at::indexing::Slice(0),0,at::indexing::Slice(0,3)})<<std::endl;

        //std::cout<<"first few elements subblock: "<<mEntropyCoder.mSubbandLF_.data.index({at::indexing::Slice(0),at::indexing::Slice(0),0,at::indexing::Slice(0,3)})<<std::endl;

#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Printing sizes" << std::endl;
#endif
        std::cout<<"mPartitionData_.size: "<<mPartitionData_.data.size(0)<<"x"<<mPartitionData_.data.size(1)<<"x"<<mPartitionData_.data.size(2)<<"x"<<mPartitionData_.data.size(3)<<std::endl;
        std::cout<<"mEntropyCoder.mSubbandLF_.size: "<<mEntropyCoder.mSubbandLF_.data.size(0)<<"x"<<mEntropyCoder.mSubbandLF_.data.size(1)<<"x"<<mEntropyCoder.mSubbandLF_.data.size(2)<<"x"<<mEntropyCoder.mSubbandLF_.data.size(3)<<std::endl;
        std::cout<<"length: "<<length[0]<<"x"<<length[1]<<"x"<<length[2]<<"x"<<length[3]<<std::endl;
        std::cout<<"trueLength: "<<trueLength[0]<<"x"<<trueLength[1]<<"x"<<trueLength[2]<<"x"<<trueLength[3]<<std::endl;
        std::cout<<"lf position: "<<mEntropyCoder.mSubbandLF_.lightFieldPosition[0]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[1]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[2]<<"x"<<mEntropyCoder.mSubbandLF_.lightFieldPosition[3]<<std::endl;
        //std::cout<<"4"<<std::endl;
#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] encodingSubblockFromPool" << std::endl;
#endif
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
        
#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Incrementing distortion and size" << std::endl;
#endif
        double weight = totalTransformGain();
        double distortion = (double) mEntropyCoder.mDistortion/(weight*weight);
        mCodingPartitionInfo.incrementTotalDistortion(distortion);
        double mse = distortion/(length[0]*length[1]*length[2]*length[3]);
        mCodingPartitionInfo.incrementTotalSize(mEntropyCoder.mRate * (length[0]*length[1]*length[2]*length[3]));
        //std::cout<<mCodingPartitionInfo.getTotalDistortion()<<" "<<mCodingPartitionInfo.getTotalSize()<<std::endl;
        //std::cout<<"mRate = "<<mEntropyCoder.mRate<<" mDistortion: "<<(double) mEntropyCoder.mDistortion/(weight*weight)<<std::endl;
#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Accessing mCuiBuffer[" << mCodingUnitIndex << "]. Size: " << mCuiBuffer.size() << std::endl;
#endif
        double psnr = 10 * log10((1024*1024)/mse);
        mCuiBuffer[mCodingUnitIndex].setPSNR(psnr);
        mCuiBuffer[mCodingUnitIndex].setRate(mEntropyCoder.mRate);
#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Appending CodingUnitInfo" << std::endl;
#endif
        mCodingPartitionInfo.appendCodingUnitInfo(mCuiBuffer[mCodingUnitIndex]);
        //std::cout<<"stA: "<<mCuiBuffer[mCodingUnitIndex].getStructureTensorAverageAngle()<<std::endl;
        //std::cout<<"ldA: "<<mCuiBuffer[mCodingUnitIndex].getLogdetAverageAngle()<<std::endl;

        mCodingUnitIndex++;

#ifdef DEBUG_TRANSFORM_PARTITION
        std::cout << "[DEBUG_TP] Finished NOSPLITFLAG block" << std::endl;
#endif
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

double TransformPartition::solveQuadrant(
    const Block4D_& inputBlock,
    int64_t y_off, int64_t x_off, 
    int64_t h, int64_t w,
    const std::array<int64_t, 4>& parentPos,
    const std::array<int64_t, 4>& parentLen,
    BlockCollage& outCollage, 
    std::string& outCode) 
{
    // Create new coordinates based on parent + offsets
    std::array<int64_t, 4> sub_pos = parentPos;
    std::array<int64_t, 4> sub_len = parentLen;

    sub_pos[2] += y_off; sub_pos[3] += x_off;
    sub_len[2] = h;      sub_len[3] = w;

    // Recurse back into the optimizer
    // Assuming RDoptimizeTransformStep is the entry point that calls splitAndOptimize
    return RDoptimizeTransformStep(inputBlock, outCollage, sub_pos, sub_len, outCode);
}
double TransformPartition::splitInFour(
    const Block4D_& inputBlock,
    const std::array<int64_t, 4>& pos,
    const std::array<int64_t, 4>& len,
    BlockCollage& outCollage,
    std::string& outCode) 
{
    double totalJS = 0.0;

    // 1. Calculate split dimensions (handling odd sizes)
    int64_t h_top    = len[2] / 2;
    int64_t h_bottom = len[2] - h_top;
    int64_t w_left   = len[3] / 2;
    int64_t w_right  = len[3] - w_left;

    // 2. Prepare storage for sub-results
    std::array<BlockCollage, 4> subCollages;
    std::string pc00, pc01, pc10, pc11;

    // 3. Solve each quadrant
    // Indices: 0=TL, 1=TR, 2=BL, 3=BR (matches your verifyBlockSize logic)
    totalJS += solveQuadrant(inputBlock, 0,     0,      h_top,    w_left,  pos, len, subCollages[0], pc00);
    totalJS += solveQuadrant(inputBlock, 0,     w_left, h_top,    w_right, pos, len, subCollages[1], pc01);
    totalJS += solveQuadrant(inputBlock, h_top, 0,      h_bottom, w_left,  pos, len, subCollages[3], pc10);
    totalJS += solveQuadrant(inputBlock, h_top, w_left, h_bottom, w_right, pos, len, subCollages[2], pc11);

    // 4. Set Output: Partition Code 
    outCode = pc00 + pc01 + pc11 + pc10;

    // 5. Set Output: The Collage (Uses the move-based array constructor we wrote earlier)
    outCollage = BlockCollage(std::move(subCollages));

    return totalJS;
}


void TransformPartition::EncodePartition() {
    // 1. Reset state
    size_t codeIdx = 0;
    size_t blockIdx = 0;
    
    // 2. Encode the initial bitplane precision (legacy)
    mEntropyCoder.EncodeInteger(mEntropyCoder.mInferiorBitPlane, MINIMUM_BITPLANE_PRECISION);
    // 3. Start the recursive walkthrough
    mPartitionCollage.logBlockSizes();
    EncodeStep_Recursive(mPartitionCollage, mPartitionCode, codeIdx, blockIdx);
}


void TransformPartition::EncodeStep_Recursive(const BlockCollage& collage, const std::string& code, size_t& codeIdx, size_t& blockIdx) {
    char flag = code[codeIdx++];

    if (flag == NOSPLITFLAG) {
        // --- LEAF NODE ---
        mEntropyCoder.EncodePartitionFlag(NOSPLITFLAGSYMBOL);

 

        // MAGIC: We don't calculate position or length. 
        // We just take the next block that the RDO logic put into the vector!
        const Block4D_& currentBlock = collage.getBlock(blockIdx++);
        currentBlock.ssi.print();
        mEntropyCoder.EncodeSSI_(currentBlock.ssi);


        // Set the entropy coder's active subband
        mEntropyCoder.mSubbandLF_ = currentBlock;
        std::cout<<"CurrentBlock Size = "<< currentBlock.size[0] << " " << currentBlock.size[1] << " " << currentBlock.size[2] << " " << currentBlock.size[3]<<std::endl;
        std::cout<<"CurrentBlock Transform Size = "<< currentBlock.transformSize[0] << " " <<currentBlock.transformSize[1] << " " <<currentBlock.transformSize[2] << " " <<currentBlock.transformSize[3]<<std::endl;
        std::cout<<"CurrentBlock Data Size = "<< currentBlock.data.size(0)<<" "<< currentBlock.data.size(1)<<" "<< currentBlock.data.size(2)<<" "<< currentBlock.data.size(3)<<std::endl;

        // Use the block's internal size (no more guessing)
        if (currentBlock.transformSize[2] * currentBlock.transformSize[3] > 0) {
            
            // Check if the tensor data itself is entirely composed of zeros (a black image)
            // currentBlock.data.any() is false if all elements are exactly zero.
            std::cout<<currentBlock.data.index({at::indexing::Slice(),at::indexing::Slice(),0,at::indexing::Slice(0,10)});
            if (!currentBlock.data.any().item<bool>()) {
                std::cout << "WARNING: Compressing a completely black (all zeros) block!" << std::endl;
                // Add any logic here if you want to skip compressing this zero block
            }

            mEntropyCoder.encodeSubblockFromPool(
                currentBlock.transformSize, 
                {0,0,0,0}, 
                mEntropyCoder.mSuperiorBitPlane, 
                mLambda
            );
        }
    } 
    else if (flag == INTRAVIEWSPLITFLAG) {
        // --- SPLIT NODE ---
        mEntropyCoder.EncodePartitionFlag(INTRAVIEWSPLITFLAGSYMBOL);

        // Recurse in the same order as your RDO (TL, TR, BR, BL)
        EncodeStep_Recursive(collage, code, codeIdx, blockIdx); // TL
        EncodeStep_Recursive(collage, code, codeIdx, blockIdx); // TR
        EncodeStep_Recursive(collage, code, codeIdx, blockIdx); // BR
        EncodeStep_Recursive(collage, code, codeIdx, blockIdx); // BL
    }
}
#include "OldDCT/MultiscaleTransform.h"
#include "Encoder/Hierarchical4DEncoder.h"
#include <math.h>
#include <string.h>
#include <string>
#include <vector>
#include "LightField/Block4D_.h"
#include "LightField/BlockCollage.h"
#include "DebugTools/CodingUnitInfo.h"
#include "DebugTools/CodingPartitionInfo.h"


#ifndef TRANSFORMPARTITION_H
#define TRANSFORMPARTITION_H

#define NOSPLITFLAG 'T'
#define INTRAVIEWSPLITFLAG 'S'
#define INTERVIEWSPLITFLAG 'V'
#define NOSPLITFLAGSYMBOL 0
#define INTRAVIEWSPLITFLAGSYMBOL 1
#define INTERVIEWSPLITFLAGSYMBOL 2
#define MINIMUM_BITPLANE_PRECISION 5

class MultiScaleTransfrom;

class TransformPartition {
     void create_encoder_pool(size_t num_threads, 
                                                                  size_t height, 
                                                                  size_t width);
    std::vector<std::unique_ptr<Hierarchical4DEncoder>> m_encoder_pool;
    void getOptimalMinimumBitPlane(Block4D_& inputBlock);
public:  
    // --- THE FIX: MAKE THE MANAGER CLASS NON-COPYABLE/MOVABLE ---
    // Because this class owns a pool of non-copyable encoders,
    // the class itself cannot be safely copied or moved.
    TransformPartition(const TransformPartition&) = delete;
    TransformPartition& operator=(const TransformPartition&) = delete;
    TransformPartition(TransformPartition&&) = delete;
    TransformPartition& operator=(TransformPartition&&) = delete;
    std::array<double,2> mDisparityRange;
    std::vector<SgtSideInfo> mSsiBuffer;
    std::vector<CodingUnitInfo> mCuiBuffer;
    CodingPartitionInfo mCodingPartitionInfo;
    Hierarchical4DEncoder& mEntropyCoder;
    int mDepth = 0;           /*!< Current depth in the partition tree */

    


    double mGain = 1;
    double mLambda = 0;
    double totalTransformGain(void);
    int mCodingUnitIndex = 0;
    std::string mPartitionCode;       // Was char*
    int mPartitionCodeIndex;            /*!< Scan index for the partition tree code string */
    double mLagrangianCost;             /*!< Lagrangian cost of the chosen partition */
    int mEvaluateOptimumBitPlane;       /*!< Toggles the optimum bit plane evaluation procedure on and off */
    Block4D mPartitionData;             /*!< DCT of all subblocks of the partition */
    Block4D_ mPartitionData_;
    Block4D_ mInputBlock;
    int mSpectralComponent = 0;
    std::array<int64_t,4> mMaxSize;
    BlockCollage mPartitionCollage;
    int mlength_t_min, mlength_s_min;   /*!< minimum subblock size at directions t, s */
    int mlength_v_min, mlength_u_min;   /*!< minimum subblock size at directions v, u */
    TransformPartition(void);
    TransformPartition(std::array<int64_t,4> minLength, Hierarchical4DEncoder& entropyCoder,std::array<double,2> disparityRange, double transformGain);
    ~TransformPartition(void);
    void RDoptimizeTransform_(Block4D_ &inputBlock, double lambda);
    double RDoptimizeTransformStep(
        const Block4D_ &inputBlock,
        BlockCollage &transformedBlock,
        std::array<int64_t,4> position,
        std::array<int64_t,4> length,
        std::string& partitionCode
    );    
    double getSSIBitCost(const SgtSideInfo& ssi);
    double GetExactIntegerCost(int integerValue, int precision, ProbabilityModelCollection& trackingModels);
    double GetExactSSIBitCost(const SgtSideInfo& ssi, const ProbabilityModelCollection& baselineState);
    double GetExactPartitionFlagCost(int symbol, const ProbabilityModelCollection& baselineState);
    double solveQuadrant(const Block4D_& inputBlock, int64_t y_off, int64_t x_off, int64_t h, int64_t w, const std::array<int64_t, 4>& parentPos, const std::array<int64_t, 4>& parentLen, BlockCollage& outCollage, std::string& outCode);
    double splitInFour(const Block4D_& inputBlock, const std::array<int64_t, 4>& pos, const std::array<int64_t, 4>& len, BlockCollage& outCollage, std::string& outCode);
    void CommitOptimizerState(const ProbabilityModelCollection& winningState);
    void EncodePartition_( double lambda);
    void EncodePartitionStep_(std::array<int64_t,4> position, std::array<int64_t,4> length, double lambda);
    void EncodePartition();
    void EncodeStep_Recursive(const BlockCollage& collage, const std::string& code, size_t& codeIdx, size_t& blockIdx);
    double EvaluatePartition_(Hierarchical4DEncoder& encoder,Block4D_ &block_0, double currGain , ProbabilityModelCollection& outModel);
    double EvaluatePartitionArbitraryRho(Hierarchical4DEncoder& encoder, Block4D_ &block_0, double currGain, double angle, double rhoAngle, double rhoSpace, ProbabilityModelCollection& outModel);
    double EvaluatePartitionFixedRho(Hierarchical4DEncoder& encoder, Block4D_ &block_0, double currGain, double angleV, double angleH, ProbabilityModelCollection& outModel);
    double EvaluatePartitionLSRho(Hierarchical4DEncoder& encoder, Block4D_ &block_0, double currGain, double angleV, double angleH, ProbabilityModelCollection& outModel);
    double RDtestStructureTensor(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RDtestLogdet(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RDtestGridSearch(double angleStep,std::array<double,2> angleRange, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RDtestAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModelCollection& outModel);
    double RDrefineStructureTensor(Block4D_& block_0, double refinementPrecision, CodingUnitInfo& cui0, ProbabilityModelCollection& outModel);
    double RDrefineAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModelCollection& outModel);
    double RDrefineLogdet(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModelCollection& outModel);
    double RDrefineGridSearch(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModelCollection& outModel);
    double RDtestCovariance(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RDrefineCovariance(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModelCollection& outModel);
    double  RDStructureTensorOrLogdet(Block4D_& block_0,CodingUnitInfo& cui0, ProbabilityModelCollection& outModel);
    double RDtestAngle(double angle,Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RDtestZero(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double parallelRhoSearch(bool searchSpace, double fixedRho, double angle, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double  RDtestStructureTensorAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RDgridSearchAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RefineGridSearchAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    double RefineStructureTensorAndRhos(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModelCollection& outModel);
    std::array<int64_t,4> adjustLFPositionForSubblock(std::array<int64_t,4> parentLFPos, std::array<int64_t,4> subblockOffset) const;
};
   

#endif /* TRANSFORMOPTIMIZATION_H */


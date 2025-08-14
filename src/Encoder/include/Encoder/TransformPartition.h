#include "OldDCT/MultiscaleTransform.h"
#include "Encoder/Hierarchical4DEncoder.h"
#include <math.h>
#include <string.h>
#include <vector>
#include "LightField/Block4D_.h"
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
    char *mPartitionCode;               /*!< String of flags defining the partition tree */
    int mPartitionCodeIndex;            /*!< Scan index for the partition tree code string */
    double mLagrangianCost;             /*!< Lagrangian cost of the chosen partition */
    int mEvaluateOptimumBitPlane;       /*!< Toggles the optimum bit plane evaluation procedure on and off */
    Block4D mPartitionData;             /*!< DCT of all subblocks of the partition */
    Block4D_ mPartitionData_;
    int mlength_t_min, mlength_s_min;   /*!< minimum subblock size at directions t, s */
    int mlength_v_min, mlength_u_min;   /*!< minimum subblock size at directions v, u */
    TransformPartition(void);
    TransformPartition(std::array<int64_t,4> minLength, Hierarchical4DEncoder& entropyCoder,std::array<double,2> disparityRange, double transformGain);
    ~TransformPartition(void);
    void RDoptimizeTransform_(Block4D_ &inputBlock, double lambda);
    double RDoptimizeTransformStep_(Block4D_ &inputBlock, Block4D_ &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length , std::vector<SgtSideInfo>& currSsi,std::vector<CodingUnitInfo>& currCui,char **partitionCode);
    void EncodePartition_( double lambda);
    void EncodePartitionStep_(std::array<int64_t,4> position, std::array<int64_t,4> length, double lambda);
    double EvaluatePartition_(Hierarchical4DEncoder& encoder,Block4D_ &block_0, double currGain , double angleH, double angleV);
    double EvaluatePartitionLSRho(Hierarchical4DEncoder& encoder,Block4D_ &block_0, double currGain , double angleH, double angleV);
    double EvaluatePartitionFixedRho(Hierarchical4DEncoder& encoder,Block4D_ &block_0, double currGain , double angleH, double angleV);
    double RDtestStructureTensor(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0);
    double RDtestLogdet(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0);
    double RDtestGridSearch(double angleStep,std::array<double,2> angleRange, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0);
    double RDtestAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0);
    double RDrefineStructureTensor(Block4D_& block_0, double refinementPrecision, CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0);
    double RDrefineAllAngleHeuristics(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0);
    double RDrefineLogdet(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0);
    double RDrefineGridSearch(Block4D_& block_0, CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0);
    double RDtestCovariance(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0);
    double RDrefineCovariance(Block4D_& block_0, double refinementPrecision,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0);
    double  RDStructureTensorOrLogdet(Block4D_& block_0,CodingUnitInfo& cui0, ProbabilityModel **coderModelState_0);
    double RDtestAngle(double angle,Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0);
    double RDtestZero(Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0);
    double EvaluatePartitionArbitraryRho(Hierarchical4DEncoder& encoder, Block4D_ &block_0, double currGain , double angle, double rhoAngle, double rhoSpace);
    double parallelRhoSearch(bool searchSpace, double fixedRho, double angle, Block4D_& block_0, CodingUnitInfo& cui0, double currGain, ProbabilityModel **coderModelState_0);



};
   

#endif /* TRANSFORMOPTIMIZATION_H */


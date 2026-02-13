#include "Encoder/Hierarchical4DEncoder.h"
#include <math.h>
#include <string.h>
#include <vector>
#include "LightField/Block4D.h"



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
    void getOptimalMinimumBitPlane(Block4D& inputBlock);
public:  
    // --- THE FIX: MAKE THE MANAGER CLASS NON-COPYABLE/MOVABLE ---
    // Because this class owns a pool of non-copyable encoders,
    // the class itself cannot be safely copied or moved.
    TransformPartition(const TransformPartition&) = delete;
    TransformPartition& operator=(const TransformPartition&) = delete;
    TransformPartition(TransformPartition&&) = delete;
    TransformPartition& operator=(TransformPartition&&) = delete;
    std::array<double,2> mDisparityRange;
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
    Block4D mPartitionData_;
    int mlength_t_min, mlength_s_min;   /*!< minimum subblock size at directions t, s */
    int mlength_v_min, mlength_u_min;   /*!< minimum subblock size at directions v, u */
    TransformPartition(void);
    TransformPartition(std::array<int64_t,4> minLength, Hierarchical4DEncoder& entropyCoder,std::array<double,2> disparityRange, double transformGain);
    ~TransformPartition(void);
    void RDoptimizeTransform(Block4D &inputBlock, double lambda);
    double RDoptimizeTransformStep(Block4D &inputBlock, Block4D &transformedBlock, std::array<int64_t,4> position, std::array<int64_t,4> length ,char **partitionCode);
    void EncodePartition( double lambda);
    void EncodePartitionStep(std::array<int64_t,4> position, std::array<int64_t,4> length, double lambda);
    double EvaluatePartition(Hierarchical4DEncoder& encoder,Block4D &block_0, double currGain);
   


};
   

#endif /* TRANSFORMOPTIMIZATION_H */


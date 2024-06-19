#include "MultiscaleTransform.h"
#include "Hierarchical4DEncoder.h"
#include <math.h>
#include <string.h>

#ifndef TRANSFORMPARTITION_H
#define TRANSFORMPARTITION_H

#define NOSPLITFLAG 'T'
#define INTRAVIEWSPLITFLAG 'S'
#define INTERVIEWSPLITFLAG 'V'
#define NOSPLITFLAGSYMBOL 0
#define INTRAVIEWSPLITFLAGSYMBOL 1
#define INTERVIEWSPLITFLAGSYMBOL 2
#define MINIMUM_BITPLANE_PRECISION 5

class TransformPartition {
public:  
    char *mPartitionCode;               /*!< String of flags defining the partition tree */
    int mPartitionCodeIndex;            /*!< Scan index for the partition tree code string */
    double mLagrangianCost;             /*!< Lagrangian cost of the chosen partition */
    int mEvaluateOptimumBitPlane;       /*!< Toggles the optimum bit plane evaluation procedure on and off */
    Block4D mPartitionData;             /*!< DCT of all subblocks of the partition */
    int mlength_t_min, mlength_s_min;   /*!< minimum subblock size at directions t, s */
    int mlength_v_min, mlength_u_min;   /*!< minimum subblock size at directions v, u */
    TransformPartition(void);
    ~TransformPartition(void);
    void RDoptimizeTransform(Block4D &inputBlock, MultiscaleTransform &mt, Hierarchical4DEncoder &entropyCoder, double lambda);
    double RDoptimizeTransformStep(Block4D &inputBlock, Block4D &transformedBlock, int *position, int *length, MultiscaleTransform &mt, Hierarchical4DEncoder &entropyCoder, double lambda, char **partitionCode);
    void EncodePartition(Hierarchical4DEncoder &entropyCoder, double lambda);
    void EncodePartitionStep(int *position, int *length, Hierarchical4DEncoder &entropyCoder, double lambda);
};

#endif /* TRANSFORMOPTIMIZATION_H */


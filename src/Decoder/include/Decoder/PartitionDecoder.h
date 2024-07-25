#include "Decoder/Hierarchical4DDecoder.h"
#include <math.h>
#include <string.h>


#ifndef PARTITIONDECODER_H
#define PARTITIONDECODER_H

#define NOSPLITFLAGSYMBOL 0
#define INTRAVIEWSPLITFLAGSYMBOL 1
#define INTERVIEWSPLITFLAGSYMBOL 2
#define MINIMUM_BITPLANE_PRECISION 5


class PartitionDecoder {
public:  
    int mPartitionCodeMaxLength;        /*!< Maximum length of the partition tree code string */
    int mEvaluateOptimumBitPlane;       /*!< Toggles the optimum bit plane evaluation procedure on and off */
    int mUseSameBitPlane;               /*!< Forces to use the same minimum bitplane for all subblocks */
    Block4D_ mPartitionData;             /*!< DCT of all subblocks of the partition */
    PartitionDecoder(void);
    void DecodePartition(Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange);
    double transformGain(std::array<int64_t,4> length);

    void DecodePartitionStep(std::array<int64_t,4> position, std::array<int64_t,4> length, Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange);
};

#endif /* TRANSFORMOPTIMIZATION_H */


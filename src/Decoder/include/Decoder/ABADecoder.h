#include "ProbabilityModel/ProbabilityModel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef ABADECODER_H
#define ABADECODER_H

#define INTERVAL_PRECISION 16
// The arithmetic decoder legitimately reads a handful of bits past the true end of a
// well-formed, complete bitstream (renormalization look-ahead for the final symbols).
// Tolerate that; anything beyond this many total padded reads means the file is
// genuinely truncated or the decoder has desynced, and we should fail cleanly instead
// of fabricating bits and recursing forever.
#define MAX_TOLERATED_EOF_READS 64
#define MSB_MASK 0x8000
#define SECOND_MSB_MASK 0x4000
#define MAXINT 0xffff
#define RESET_LSB_MASK 0xfffe
#define SET_LSB_MASK 0x0001

/*********************************************************************************************/
/*                                         class ABADecoder                                     */
/*********************************************************************************************/
class ABADecoder {
public:  
  FILE *mInputFilePointer;            /*!< pointer to input file */
  unsigned int mLow;                  /*!< interval lower limit */
  unsigned int mHigh;                 /*!< interval upper limit */
  unsigned int mTag;                  /*!< received tag */
  unsigned char mBitBuffer;           /*!< bit-readable buffer */
  int mNumberOfBitsInBuffer;          /*!< number of valid bits in buffer */
  //unsigned char mLastByte;
  int mNumberOfbitsreadAfterlastBitDecoded;
  int mCumulativeEOFReads;             /*!< total bits read past end-of-file across the whole decode session */
  ABADecoder(void);
  ~ABADecoder(void);
  void InitDecoder(FILE *ifp);
  int DecodeBit(ProbabilityModel &mPmodel);
  void Finish(void);  
  int ReadBitFromFile(void);
};

#endif /* ABADECODER_H */


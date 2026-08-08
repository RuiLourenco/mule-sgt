#include "Decoder/ABADecoder.h"
#include <iostream>
/*********************************************************************************************/
/*                                    class ABADecoder methods                               */
/*********************************************************************************************/
ABADecoder :: ABADecoder(void) {
    mNumberOfBitsInBuffer = 0;
    mInputFilePointer = NULL;
    mCumulativeEOFReads = 0;
}
ABADecoder :: ~ABADecoder(void) {
  
}
void ABADecoder :: InitDecoder(FILE *ifp) {

    mInputFilePointer = ifp;

    mNumberOfBitsInBuffer = 0;
    mCumulativeEOFReads = 0;

    mLow = 0;
    mHigh = MAXINT;
    
    mTag = 0;
    for(int n = 0; n < INTERVAL_PRECISION; n++) {
        mTag = mTag << 1;
        int bit = ReadBitFromFile();
        if(bit == 1)
            mTag = mTag | SET_LSB_MASK;
    }

}
int ABADecoder :: DecodeBit(ProbabilityModel &mPmodel) {
  
    unsigned long int acumFreq_0 = mPmodel.mCumulativeFrequency[0];
    unsigned long int acumFreq_1 = mPmodel.mCumulativeFrequency[1];
    int bitDecoded;
    unsigned long int threshold = (((mTag - mLow + 1) * acumFreq_1 - 1)/(mHigh - mLow + 1));
    unsigned long int length_0 = (((mHigh - mLow + 1) * acumFreq_0)/acumFreq_1);

    if(threshold < acumFreq_0) {
         bitDecoded = 0;
         mHigh = mLow + length_0 - 1;
    }
    else {
         bitDecoded = 1;
         mLow = mLow + length_0;
    }

    mNumberOfbitsreadAfterlastBitDecoded = 0;
    while(((mLow & MSB_MASK) == (mHigh & MSB_MASK)) || ((mLow >= SECOND_MSB_MASK) && (mHigh < (MSB_MASK + SECOND_MSB_MASK)))) {
        
       if((mLow & MSB_MASK) == (mHigh & MSB_MASK)) {
            mLow = mLow << 1;
            mHigh = mHigh << 1;
            mLow = mLow & RESET_LSB_MASK;
            mHigh = mHigh | SET_LSB_MASK;
            mTag = mTag << 1;
            int bit = ReadBitFromFile();
            if(bit == 1)
                mTag = mTag | SET_LSB_MASK;
            else
                mTag = mTag & RESET_LSB_MASK;
        
            mHigh = mHigh & MAXINT;
            mLow = mLow & MAXINT;
            mTag = mTag  & MAXINT;
        }
        if((mLow >= SECOND_MSB_MASK) && (mHigh < (MSB_MASK + SECOND_MSB_MASK))) {
            mLow = mLow << 1;
            mHigh = mHigh << 1;
            mLow = mLow & RESET_LSB_MASK;
            mHigh = mHigh | SET_LSB_MASK;
            mTag = mTag << 1;
            int bit = ReadBitFromFile();
            if(bit == 1)
                mTag = mTag | SET_LSB_MASK;
            else
                mTag = mTag & RESET_LSB_MASK;
                       
            mLow = mLow ^ MSB_MASK;           
            mHigh = mHigh ^ MSB_MASK;
            mTag = mTag ^ MSB_MASK;
            mHigh = mHigh & MAXINT;
            mLow = mLow & MAXINT;
            mTag = mTag  & MAXINT;
        }
    }
    mHigh = mHigh & MAXINT;
    mLow = mLow & MAXINT;
    mTag = mTag  & MAXINT;
    return (bitDecoded);
  
}

void ABADecoder :: Finish(void) {
  
    fseek(mInputFilePointer, -2, SEEK_CUR);
    
}

int ABADecoder :: ReadBitFromFile(void) {
  
    int bit;
    mNumberOfbitsreadAfterlastBitDecoded++;
    
    if (mNumberOfBitsInBuffer == 0) {
        int byteRead = fgetc(mInputFilePointer);
        if (byteRead == EOF) {
            mCumulativeEOFReads += 8;
            if (mCumulativeEOFReads > MAX_TOLERATED_EOF_READS) {
                std::cout << "Error: ABADecoder::ReadBitFromFile read too far past end of file "
                          << "(likely a truncated/incomplete bitstream or a decode desync). "
                          << "Bits read past EOF so far: " << mCumulativeEOFReads
                          << ", bits read after last decoded symbol: " << mNumberOfbitsreadAfterlastBitDecoded << std::endl;
                exit(-58);
            }
            // A well-formed, complete bitstream is not padded, so decoding its final few
            // symbols legitimately reads a little past the real end (renormalization
            // look-ahead). Pad with neutral zero bits rather than fabricating garbage.
            byteRead = 0;
        }
        mBitBuffer = byteRead;
        //mLastByte = mBitBuffer;
        mNumberOfBitsInBuffer = 8;
    }

    bit = mBitBuffer&01;

    mBitBuffer = mBitBuffer >> 1;

    mNumberOfBitsInBuffer--;

    return(bit);  
}

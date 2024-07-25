#include "Decoder/PartitionDecoder.h"
#include "LightField/Block4D_.h"

/*******************************************************************************/
/*                      PartitionDecoder class methods                         */
/*******************************************************************************/

PartitionDecoder :: PartitionDecoder(void) {
    mUseSameBitPlane = 1;
}
   
void PartitionDecoder :: DecodePartition(Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange) {
    
    
    std::array<int64_t,4> position;
    position[0] = 0;
    position[1] = 0;
    position[2] = 0;
    position[3] = 0;
    entropyDecoder.mSkipCount = 0;


    std::array<int64_t,4> length;
    length[0] = mPartitionData.data.size(0);
    length[1] = mPartitionData.data.size(1);
    length[2] = mPartitionData.data.size(2);
    length[3] = mPartitionData.data.size(3);
    entropyDecoder.mSkipMatrix = at::zeros(length);

    
    entropyDecoder.mInferiorBitPlane = entropyDecoder.DecodeInteger(MINIMUM_BITPLANE_PRECISION);

    DecodePartitionStep(position, length, entropyDecoder,disparityRange);
    std::cout<<"Skipped "<<entropyDecoder.mSkipCount<<" blocks"<<std::endl;
    std::cout<<"finished partition decoding"<<std::endl;
}
double PartitionDecoder :: transformGain(std::array<int64_t,4> length){
    
    double transformGain = 1;
    for(int i = 0; i < 4; i++){
        transformGain*=length[i]/sqrt(length[i]);

        transformGain  *= sqrt(mPartitionData.data.size(i)/length[i]);
    }
    return transformGain;

}
void PartitionDecoder :: DecodePartitionStep(std::array<int64_t,4> position, std::array<int64_t,4>length, Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange) {
    //std::cout << "Entered Step with size: ("<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<")"<<std::endl;
    int flagCode = entropyDecoder.DecodePartitionFlag();
    //std::cout<<flagCode<<" "<<std::endl;
    //std::cout<<"("<<position[2]<<","<<position[3]<<") "<<length[2]<<"x"<<length[3]<<std::endl;

    if(flagCode == NOSPLITFLAGSYMBOL) {
        
        SgtSideInfo ssi = entropyDecoder.DecodeSsi(disparityRange);
        entropyDecoder.mSubbandLF = Block4D_(length);
        entropyDecoder.mSubbandLF.Zeros();


        entropyDecoder.DecodeBlock(0, 0, 0, 0, length[0], length[1], length[2], length[3], entropyDecoder.mSuperiorBitPlane); 
        int64_t maxIndex = entropyDecoder.mSubbandLF.data.argmax().item<int64_t>();
        int64_t minIndex = entropyDecoder.mSubbandLF.data.argmin().item<int64_t>();
        int64_t t_min, s_min, u_min, v_min, t_max, s_max, u_max, v_max;
        entropyDecoder.mSubbandLF.CoordPosition(maxIndex, t_max, s_max, u_max, v_max);
        entropyDecoder.mSubbandLF.CoordPosition(minIndex, t_min, s_min, u_min, v_min);
        // std::cout<<entropyDecoder.mSubbandLF.data.argmax() <<" "<<entropyDecoder.mSubbandLF.data.argmin()<<std::endl;
        // std::cout<<"Partition Size: "<<entropyDecoder.mSubbandLF.data.sizes()<<"  "<<entropyDecoder.mSubbandLF.data.is_contiguous()<<std::endl;
        double gain = transformGain(length);        
        entropyDecoder.mSubbandLF.isgtTransform(gain,ssi);
        
        //std::cout<<"Completed Transform"<< " position = (" << position[0] << " " << position[1] << " " << position[2] << " " << position[3] << ")" << std::endl;
        //std::cout<<"Partition Size: "<<mPartitionData.data.sizes()<<std::endl;
        mPartitionData.CopySubblockFrom(entropyDecoder.mSubbandLF, {0, 0, 0, 0}, position);
        //std::cout<<" copied block"<<std::endl;
        return;
    }
    
    if(flagCode == INTRAVIEWSPLITFLAGSYMBOL) {
        
        std::array<int64_t,4> new_position, new_length;
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0];
        new_length[1] = length[1];
        new_length[2] = length[2]/2;
        new_length[3] = length[3]/2;
        
        //Decode four spatial subblocks 
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);

        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        return;
    }
    
    if(flagCode == INTERVIEWSPLITFLAGSYMBOL) {
        
        std::array<int64_t,4> new_position, new_length;
        
        new_position[0] = position[0];
        new_position[1] = position[1];
        new_position[2] = position[2];
        new_position[3] = position[3];
        
        new_length[0] = length[0]/2;
        new_length[1] = length[1]/2;
        new_length[2] = length[2];
        new_length[3] = length[3];
        
        //Decode four view subblocks 
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);

        new_position[1] = position[1] + length[1]/2;
        new_length[1] = length[1] - length[1]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);

        new_position[0] = position[0] + length[0]/2;
        new_length[0] = length[0] - length[0]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        
        new_position[1] = position[1];
        new_length[1] = length[1]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        return;
    }
}

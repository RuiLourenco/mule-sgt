#include "Decoder/PartitionDecoder.h"
#include "LightField/Block4D.h"
#include <vector>

/*******************************************************************************/
/*                      PartitionDecoder class methods                         */
/*******************************************************************************/

PartitionDecoder :: PartitionDecoder(double mGain):mGain(mGain) {
    mUseSameBitPlane = 1;
}

double PartitionDecoder :: calcEntropy(at::Tensor &data) {

    auto [unique,dummy1]  = at::_unique(data,true,false);
    int64_t levels = unique.size(0);
    std::cout<<"Byte number = "<<levels<<std::endl;
    //std::cout<<"Count: "<<std::endl<<unique<<std::endl;
    at::Tensor flat = data.flatten().contiguous();
    int* data_ptr = flat.data_ptr<int>();
    int* unq_ptr = unique.data_ptr<int>();

    //int64_t levels = std::pow(2,byte_number);

    std::vector<int64_t> hc(levels,0);
    std::vector<double> aux(levels);
    //cout<<levels<<" "<<flat.max().item()<<endl;
    double total0 = 0;
    int64_t total_size = flat.numel();
    for(int i = 0; i < total_size; i++){
        int elem = data_ptr[i];
        //std::cout<<"Elem: "<<elem<<std::endl;

        for(int j = 0; j < levels; j++){
            int64_t elem2 = unq_ptr[j];
            //int64_t elem2 = unique[j].item<int>();
            if(elem2 == elem){
                hc[j]++;
                total0++;
                //std::cout<<elem2<<" "<<elem<<std::endl;
            }
        }
    }
    double total = 0;
    //cout<<"levels = "<<levels<<endl;
    double entropy = 0;
    for(int j = 0; j < levels; j++){
        //cout<<j<<" = "<< hc[j]<<" ";
        total += hc[j];
        if(hc[j] != 0){
            double p = (double)hc[j]/(double)(total_size);
            //cout<<p<<" ";
            double logp= log2(p);
            entropy+= -p*logp;
        }
        //cout<<logp<<" ";
        //aux[j] = -p*logp;
    }
    //std::cout<<total0<<" "<<total<<" "<<total_size + levels<<std::endl;
    //std::cout<<"Total Information = "<<(entropy * total)/8<<" bytes"<<std::endl;
    //cout<<endl;
    //double entropy = std::accumulate(aux.begin(),aux.end(),0);  
    //cout<<"entropy = "<<entropy<<endl;    
    //at::Tensor entropy_tensor = torch::zeros({1})+entropy;
    return entropy;  
};
   
void PartitionDecoder :: DecodePartition(Block4D reconstructedBlock, Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange) {
    
    
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


    
    entropyDecoder.mInferiorBitPlane = entropyDecoder.DecodeInteger(MINIMUM_BITPLANE_PRECISION);
    std::cout<<"Minimum Bit Plane: "<<entropyDecoder.mInferiorBitPlane<<std::endl;

    DecodePartitionStep(reconstructedBlock,position, length, entropyDecoder,disparityRange);
    std::cout<<"finished partition decoding"<<std::endl;
}
double PartitionDecoder :: transformGain(std::array<int64_t,4> length){
    
    double transformGain = 1;
    for(int i = 0; i < 4; i++){
        transformGain*=length[i]/sqrt(length[i]);

        transformGain  *= sqrt(mPartitionData.data.size(i)/length[i]);
    }
    return transformGain*mGain;
}
void PartitionDecoder :: DecodePartitionStep(Block4D reconstructedBlock,std::array<int64_t,4> position, std::array<int64_t,4>length, Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange) {
    int flagCode = entropyDecoder.DecodePartitionFlag();
    if(flagCode != NOSPLITFLAGSYMBOL && flagCode != INTRAVIEWSPLITFLAGSYMBOL) {std::cout<<"why? "<<flagCode<<std::endl;exit(-55);return;}
    

    if(flagCode == NOSPLITFLAGSYMBOL) {
        //std::cout<<" Decoding SSI"<<std::endl;
        std::cout<<" decoding subblock of length: "<<length[2]<<" "<<length[3]<<std::endl;
        Block4D reconstructedSubblock = reconstructedBlock.copySubblock(length, position);
        //std::cout<<"Creating block 4D for the entropy decoder"<<std::endl;
        std::array<int64_t,4> newLFPosition = mPartitionData.lightFieldPosition;
        for(int i = 0; i < 4; i++){
            newLFPosition[i] += position[i];
        }
        std::cout<<"Light Field Position: "<<newLFPosition[0]<<" "<<newLFPosition[1]<<" "<<newLFPosition[2]<<" "<<newLFPosition[3]<<std::endl;

        entropyDecoder.mSubbandLF = Block4D(length,newLFPosition,mPartitionData.lightField);

        //std::cout<<"Block Created"<<std::endl;
        std::cout<<"Transform Size = "<<entropyDecoder.mSubbandLF.transformSize[0]<<"x"<<entropyDecoder.mSubbandLF.transformSize[1]<<"x"<<entropyDecoder.mSubbandLF.transformSize[2]<<"x"<<entropyDecoder.mSubbandLF.transformSize[3]<<std::endl;
        //This transforms the underlying data structure from 4D to 2D
        entropyDecoder.mSubbandLF.emptyTransform();

        //We only decode the block if it contains valid coefficients (necessary guard using preslant)
        if(entropyDecoder.mSubbandLF.transformSize[2] * entropyDecoder.mSubbandLF.transformSize[3] > 0){
             entropyDecoder.DecodeBlock(0, 0, 0, 0, entropyDecoder.mSubbandLF.transformSize[0], entropyDecoder.mSubbandLF.transformSize[1], entropyDecoder.mSubbandLF.transformSize[2], entropyDecoder.mSubbandLF.transformSize[3], entropyDecoder.mSuperiorBitPlane); 
        }
        
    

        double gain = transformGain(length);        
        std::cout<<"size before IKLT: "<<entropyDecoder.mSubbandLF.data.sizes()<<std::endl;
        entropyDecoder.mSubbandLF.ikltTransform(reconstructedSubblock,gain);
        mPartitionData.CopySubblockFrom(entropyDecoder.mSubbandLF, {0, 0, 0, 0}, position);
        std::cout<<" copied decoded subblock of length: "<<length[2]<<" "<<length[3]<<std::endl<<std::endl;
        return;
    }
    
    if(flagCode == INTRAVIEWSPLITFLAGSYMBOL) {
        std::cout<<"Intra View Split"<<std::endl;
        
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
        DecodePartitionStep(reconstructedBlock,new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded First Intra View Split Partition"<<std::endl;

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
        
        DecodePartitionStep(reconstructedBlock,new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded Second Intra View Split Partition"<<std::endl;


        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        DecodePartitionStep(reconstructedBlock,new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded Third Intra View Split Partition"<<std::endl;

        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        DecodePartitionStep(reconstructedBlock,new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded Last Intra View Split Partition"<<std::endl;
        return;
    }
    
    
}

#include "Decoder/PartitionDecoder.h"
#include "LightField/Block4D_.h"
#include <vector>

/*******************************************************************************/
/*                      PartitionDecoder class methods                         */
/*******************************************************************************/

PartitionDecoder :: PartitionDecoder(void) {
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
   
void PartitionDecoder :: DecodePartition(Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange) {
    
    
    std::array<int64_t,4> position;
    position[0] = 0;
    position[1] = 0;
    position[2] = 0;
    position[3] = 0;
    entropyDecoder.mSkipCount = 0;

    //std::cout<<"Foofoofoo"<<std::endl;

    std::array<int64_t,4> length;
    length[0] = mPartitionData.data.size(0);
    length[1] = mPartitionData.data.size(1);
    length[2] = mPartitionData.data.size(2);
    length[3] = mPartitionData.data.size(3);
    entropyDecoder.mSkipMatrix = at::zeros({1,1,length[0]*length[2],length[1]*length[3]});

    //std::cout<<"Moofoofoo"<<std::endl;

    
    entropyDecoder.mInferiorBitPlane = entropyDecoder.DecodeInteger(MINIMUM_BITPLANE_PRECISION);
    std::cout<<"Minimum Bit Plane: "<<entropyDecoder.mInferiorBitPlane<<std::endl;

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
    int flagCode = entropyDecoder.DecodePartitionFlag();
    if(flagCode == INTERVIEWSPLITFLAGSYMBOL) {std::cout<<"why?";return;}
    //std::cout << "Entered Step with size: ("<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<")"<<std::endl;
    //std::cout<<"Decoding Partition Flag"<<std::endl;
    //std::cout<<"Partition Flag Decoded: "<<flagCode<<std::endl;


    //std::cout<<flagCode<<" "<<std::endl;
    //std::cout<<"("<<position[2]<<","<<position[3]<<") "<<length[2]<<"x"<<length[3]<<std::endl;
    //std::array<int64_t,4> lengthTransform = {1,1,length[0]*length[2],length[1]*length[3]};

    if(flagCode == NOSPLITFLAGSYMBOL) {
        //std::cout<<" Decoding SSI"<<std::endl;
        SgtSideInfo ssi = entropyDecoder.DecodeSsi(disparityRange);
        ssi.print();
        //std::cout<<"SSI Decoded"<<std::endl;

        //std::cout<<"Creating block 4D for the entropy decoder"<<std::endl;
        entropyDecoder.mSubbandLF = Block4D_(length);

        //std::cout<<"Block Created"<<std::endl;
        //std::cout<<"Transform Size = "<<entropyDecoder.mSubbandLF.transformSize[0]<<"x"<<entropyDecoder.mSubbandLF.transformSize[1]<<"x"<<entropyDecoder.mSubbandLF.transformSize[2]<<"x"<<entropyDecoder.mSubbandLF.transformSize[3]<<std::endl;
        entropyDecoder.mSubbandLF.emptyTransform();

        //std::cout<<"parsing file and decoding hexadeca-tree clustering"<<std::endl;
        entropyDecoder.DecodeBlock(0, 0, 0, 0, entropyDecoder.mSubbandLF.transformSize[0], entropyDecoder.mSubbandLF.transformSize[1], entropyDecoder.mSubbandLF.transformSize[2], entropyDecoder.mSubbandLF.transformSize[3], entropyDecoder.mSuperiorBitPlane); 
        //std::cout<<"Block Parsed"<<std::endl;
        //std::cout<<"Parsed Block:"<<std::endl<<entropyDecoder.mSubbandLF.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

        //int64_t maxIndex = entropyDecoder.mSubbandLF.data.argmax().item<int64_t>();
        //int64_t minIndex = entropyDecoder.mSubbandLF.data.argmin().item<int64_t>();
        // int64_t t_min, s_min, u_min, v_min, t_max, s_max, u_max, v_max;
        // entropyDecoder.mSubbandLF.CoordPosition(maxIndex, t_max, s_max, u_max, v_max);
        // entropyDecoder.mSubbandLF.CoordPosition(minIndex, t_min, s_min, u_min, v_min);
        // std::cout<<entropyDecoder.mSubbandLF.data.argmax() <<" "<<entropyDecoder.mSubbandLF.data.argmin()<<std::endl;
        // std::cout<<"Partition Size: "<<entropyDecoder.mSubbandLF.data.sizes()<<"  "<<entropyDecoder.mSubbandLF.data.is_contiguous()<<std::endl;
        double entropy = calcEntropy(entropyDecoder.mSubbandLF.data);
        this->entropyImage = entropy*at::ones({length[0],length[1],length[2],length[3]},at::kDouble);

        //std::cout<<"ENTROPY = "<<entropy<<std::endl;
        //std::cout<<"ISG Transform"<<std::endl;
        double gain = transformGain(length);        

        entropyDecoder.mSubbandLF.isgtTransform(gain,ssi);
        //std::cout<<"ISG Transform DONE"<<std::endl;


        //std::cout<<"Block Decoded: "<<entropyDecoder.mSubbandLF.data.sizes()<<std::endl;
        //std::cout<<"DecompressedBlock:"<<std::endl<<entropyDecoder.mSubbandLF.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

        
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

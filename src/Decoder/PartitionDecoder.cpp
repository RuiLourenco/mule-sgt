#include "Decoder/PartitionDecoder.h"
#include "LightField/Block4D_.h"
#include <vector>
#include <fstream>

extern std::string g_inputFileName;


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
    return transformGain*mGain;
}
void PartitionDecoder :: DecodePartitionStep(std::array<int64_t,4> position, std::array<int64_t,4>length, Hierarchical4DDecoder &entropyDecoder,std::array<double,2> disparityRange) {
    int flagCode = entropyDecoder.DecodePartitionFlag();
    if(flagCode != NOSPLITFLAGSYMBOL && flagCode != INTRAVIEWSPLITFLAGSYMBOL) {std::cout<<"why? "<<flagCode<<std::endl;exit(-55);return;}
    
    //std::cout << "Entered Step with size: ("<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<")"<<std::endl;
    //std::cout<<"Decoding Partition Flag"<<std::endl;
    //std::cout<<"Partition Flag Decoded: "<<flagCode<<std::endl;


    //std::cout<<flagCode<<" "<<std::endl;
    //std::cout<<"("<<position[2]<<","<<position[3]<<") "<<length[2]<<"x"<<length[3]<<std::endl;
    //std::array<int64_t,4> lengthTransform = {1,1,length[0]*length[2],length[1]*length[3]};

    if(flagCode == NOSPLITFLAGSYMBOL) {
        //std::cout<<" Decoding SSI"<<std::endl;
        std::cout<<" decoding subblock of length: "<<length[2]<<" "<<length[3]<<std::endl;

        SgtSideInfo ssi = entropyDecoder.DecodeSsi(disparityRange);
        ssi.print();
        //std::cout<<"SSI Decoded"<<std::endl;

        //std::cout<<"Creating block 4D for the entropy decoder"<<std::endl;
        std::array<int64_t,4> newLFPosition = mPartitionData.lightFieldPosition;
        for(int i = 0; i < 4; i++){
            newLFPosition[i] += position[i];
        }
        std::cout<<"Light Field Position: "<<newLFPosition[0]<<" "<<newLFPosition[1]<<" "<<newLFPosition[2]<<" "<<newLFPosition[3]<<std::endl;

        entropyDecoder.mSubbandLF = Block4D_(length,newLFPosition,mPartitionData.lightField);

        //std::cout<<"Block Created"<<std::endl;
        std::cout<<"Transform Size = "<<entropyDecoder.mSubbandLF.transformSize[0]<<"x"<<entropyDecoder.mSubbandLF.transformSize[1]<<"x"<<entropyDecoder.mSubbandLF.transformSize[2]<<"x"<<entropyDecoder.mSubbandLF.transformSize[3]<<std::endl;
        entropyDecoder.mSubbandLF.emptyTransform();

        //std::cout<<"parsing file and decoding hexadeca-tree clustering"<<std::endl;
        std::cout<<"entropyDecoderSize: "<<entropyDecoder.mSubbandLF.data.sizes()<<std::endl;
        if(entropyDecoder.mSubbandLF.transformSize[2] * entropyDecoder.mSubbandLF.transformSize[3] > 0) entropyDecoder.DecodeBlock(0, 0, 0, 0, entropyDecoder.mSubbandLF.transformSize[0], entropyDecoder.mSubbandLF.transformSize[1], entropyDecoder.mSubbandLF.transformSize[2], entropyDecoder.mSubbandLF.transformSize[3], entropyDecoder.mSuperiorBitPlane); 
        //std::cout<<"Block Parsed"<<std::endl;
        std::cout<<"Parsed Block:"<<std::endl<<entropyDecoder.mSubbandLF.data.index({0,0,at::indexing::Slice(0,1),at::indexing::Slice(0,3)})<<std::endl;
        std::cout<<"Parsed Block:"<<std::endl<<entropyDecoder.mSubbandLF.data.index({0,0,at::indexing::Slice(0,3),at::indexing::Slice(0,1)})<<std::endl;
        
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
        std::cout<<"size before ISGT: "<<entropyDecoder.mSubbandLF.data.sizes()<<std::endl;
        
        {
            std::string channelStr = "";
            if (mSpectralComponent == 0) channelStr = "Y";
            else if (mSpectralComponent == 1) channelStr = "Cb";
            else if (mSpectralComponent == 2) channelStr = "Cr";
            
            static bool first_time[3] = {true, true, true};
            std::string out_name = g_inputFileName + "_decoder_matrices_" + channelStr + ".txt";
            std::ios_base::openmode mode = first_time[mSpectralComponent] ? std::ios::out : std::ios::app;
            first_time[mSpectralComponent] = false;
            std::ofstream dec_out(out_name, mode);
            
            at::Tensor modelCovMatH = entropyDecoder.mSubbandLF.calcModelCovMatrix(ssi, true);
            at::Tensor modelCovMatV = entropyDecoder.mSubbandLF.calcModelCovMatrix(ssi, false);
            at::Tensor eigValsH, eigValsV;
            at::Tensor sgtMatrixH = entropyDecoder.mSubbandLF.getSgtTransformMatrix(modelCovMatH, true, eigValsH);
            at::Tensor sgtMatrixV = entropyDecoder.mSubbandLF.getSgtTransformMatrix(modelCovMatV, false, eigValsV);
            
            dec_out << "Block Position: " << entropyDecoder.mSubbandLF.lightFieldPosition[0] << " " << entropyDecoder.mSubbandLF.lightFieldPosition[1] << " " << entropyDecoder.mSubbandLF.lightFieldPosition[2] << " " << entropyDecoder.mSubbandLF.lightFieldPosition[3] << "\n";
            dec_out << "Block Size: " << entropyDecoder.mSubbandLF.size[0] << " " << entropyDecoder.mSubbandLF.size[1] << " " << entropyDecoder.mSubbandLF.size[2] << " " << entropyDecoder.mSubbandLF.size[3] << "\n";
            
            dec_out << "SSI RhoS: " << ssi.getRhoS() << " RhoT: " << ssi.getRhoT() 
                    << " RhoU: " << ssi.getRhoU() << " RhoV: " << ssi.getRhoV() << "\n";
            dec_out << "SSI AngleV: " << ssi.getAngleV() << " AngleH: " << ssi.getAngleH() << "\n";
            
            int rowsH = std::min<int>(10, sgtMatrixH.size(0));
            int colsH = std::min<int>(10, sgtMatrixH.size(1));
            dec_out << "sgtMatrixH (top " << rowsH << "x" << colsH << "):\n" << sgtMatrixH.index({at::indexing::Slice(0, rowsH), at::indexing::Slice(0, colsH)}) << "\n";
            
            int rowsV = std::min<int>(10, sgtMatrixV.size(0));
            int colsV = std::min<int>(10, sgtMatrixV.size(1));
            dec_out << "sgtMatrixV (top " << rowsV << "x" << colsV << "):\n" << sgtMatrixV.index({at::indexing::Slice(0, rowsV), at::indexing::Slice(0, colsV)}) << "\n";

            at::Tensor identH = sgtMatrixH.matmul(sgtMatrixH.t());
            at::Tensor trueIdentH = at::eye(sgtMatrixH.size(0), sgtMatrixH.options());
            double stabilityH = sgtMatrixH.size(0) > 0 ? at::abs(identH - trueIdentH).max().item<double>() : 0.0;

            at::Tensor identV = sgtMatrixV.matmul(sgtMatrixV.t());
            at::Tensor trueIdentV = at::eye(sgtMatrixV.size(0), sgtMatrixV.options());
            double stabilityV = sgtMatrixV.size(0) > 0 ? at::abs(identV - trueIdentV).max().item<double>() : 0.0;
            
            dec_out << "Stability/Orthogonality (Max Diff from Identity) H: " << stabilityH << "\n";
            dec_out << "Stability/Orthogonality (Max Diff from Identity) V: " << stabilityV << "\n";
            dec_out << "----------------------------------------\n";
            
            if (entropyDecoder.mSubbandLF.lightFieldPosition[2] == 1032 && entropyDecoder.mSubbandLF.lightFieldPosition[3] == 416) {
                torch::save(sgtMatrixH, g_inputFileName + "_1032_416_dec_H_" + channelStr + ".pt");
                torch::save(sgtMatrixV, g_inputFileName + "_1032_416_dec_V_" + channelStr + ".pt");
            }
        }

        entropyDecoder.mSubbandLF.isgtTransform(gain,ssi);
        //std::cout<<"ISG Transform DONE"<<std::endl;


        //std::cout<<"Block Decoded: "<<entropyDecoder.mSubbandLF.data.sizes()<<std::endl;
        //std::cout<<"DecompressedBlock:"<<std::endl<<entropyDecoder.mSubbandLF.data.index({0,0,at::indexing::Slice(0,4),at::indexing::Slice(0,4)})<<std::endl;

        
        //std::cout<<"Completed Transform"<< " position = (" << position[0] << " " << position[1] << " " << position[2] << " " << position[3] << ")" << std::endl;
        //std::cout<<"Partition Size: "<<mPartitionData.data.sizes()<<std::endl;
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
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded First Intra View Split Partition"<<std::endl;

        new_position[3] = position[3] + length[3]/2;
        new_length[3] = length[3] - length[3]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded Second Intra View Split Partition"<<std::endl;


        new_position[2] = position[2] + length[2]/2;
        new_length[2] = length[2] - length[2]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded Third Intra View Split Partition"<<std::endl;

        
        new_position[3] = position[3];
        new_length[3] = length[3]/2;
        
        DecodePartitionStep(new_position, new_length, entropyDecoder,disparityRange);
        std::cout<<"Decoded Last Intra View Split Partition"<<std::endl;
        return;
    }
    
    
}

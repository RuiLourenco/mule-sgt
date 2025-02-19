#include "Encoder/Hierarchical4DEncoder.h"
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <bitset>
/*******************************************************************************/
/*                        Hierachical4DEncoder class methods                   */
/*******************************************************************************/

Hierarchical4DEncoder :: Hierarchical4DEncoder(void) {
    mSuperiorBitPlane = 30;
    mInferiorBitPlane = 0;
    mPreSegmentation = 1;
    mSegmentationTreeCodeBuffer = NULL;
    mSegmentationTreeCodeBufferSize = 0;
    mSegmentationFlagProbabilityModelIndex = SEGMENTATION_PROB_MODEL_INDEX;
    mSymbolProbabilityModelIndex = SYMBOL_PROBABILITY_MODEL_INDEX;
    mPmodel = NULL;
    mOptimizationPmodel = NULL;
    
}
Hierarchical4DEncoder :: ~Hierarchical4DEncoder(void) {
    if(mSegmentationTreeCodeBuffer != NULL)
        delete [] mSegmentationTreeCodeBuffer;
    
    if(mPmodel != NULL)
        delete [] mPmodel;
    if(mOptimizationPmodel != NULL)
        delete [] mOptimizationPmodel;
}



void Hierarchical4DEncoder :: StartEncoder(FILE *outputFilePointer) {
    
    
    mEntropyCoder.InitEncoder(outputFilePointer);  //opens output file

    if(mPmodel != NULL)
        delete [] mPmodel;
    if(mOptimizationPmodel != NULL)
        delete [] mOptimizationPmodel;
    
    mPmodel = new ProbabilityModel[NUMBER_OF_MODELS];
    mOptimizationPmodel = new ProbabilityModel[NUMBER_OF_MODELS];
    for(int n = 0; n < NUMBER_OF_MODELS; n++) {
         mPmodel[n].ResetModel();
         mOptimizationPmodel[n].ResetModel();
    }
}

void Hierarchical4DEncoder :: RestartProbabilisticModel(void) {
    //  if(mPmodel != NULL && mOptimizationPmodel != NULL){
    //     std::cout<<" This shou;d be Fine"<<std::endl;
    //  }else{
    //      std::cout<<" This shou;d not be Fine"<<std::endl;
    //  }
     for(int n = 0; n < NUMBER_OF_MODELS; n++) {
         mPmodel[n].ResetModel();
         //std::cout<<mPmodel[n].mRate[0]<<" "<<mOptimizationPmodel[n].mRate[0]<<std::endl;
         mOptimizationPmodel[n].ResetModel();
    }
   
    
}
void Hierarchical4DEncoder :: EncodeSubblock_(double lambda) {
    
    int flagSearchIndex = 0;
    double Energy;
    double rate = 0;
    double distortion = 0;
    std::array<int64_t,4> size = {mSubbandLF_.data.size(0),mSubbandLF_.data.size(1),mSubbandLF_.data.size(2),mSubbandLF_.data.size(3)};
    this->ignored = at::zeros(size,at::kInt);

    strcpy(mSegmentationTreeCodeBuffer,"");
    this->currCost = RdOptimizeHexadecaTree_({0, 0, 0, 0}, size, lambda, mSuperiorBitPlane, &mSegmentationTreeCodeBuffer, Energy, rate,distortion);
    this->mRate = rate/(size[0]*size[1]*size[2]*size[3]);
    this->mDistortion = distortion/(size[0]*size[1]*size[2]*size[3]);

    //std::cout<<"Rate = "<<rate<< " WeightedRate = "<<lambda*rate<<"Distortion = "<<this->mDistortion<<" CurrentCost = "<<this->currCost<<" SumCheck = "<<lambda*rate + distortion<<std::endl;
    //std::cout<<"Calculated Rate = "<<(this->currCost - distortion)/lambda<<std::endl;
  
    flagSearchIndex = 0;
    RdEncodeHexadecatree_({0, 0, 0, 0}, size, mSuperiorBitPlane, flagSearchIndex);
}


double Hierarchical4DEncoder :: RdOptimizeHexadecaTree_(std::array<int64_t,4> position, std::array<int64_t,4> length, double lambda, int bitplane, char **codeString, double &signalEnergy, double &rate, double& distortion) {
   //std::cout<<"Length = "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl; 
    //std::cout<<"In"<<std::endl;
    double rate0 = 0;
    double rate1 = 0;
    double distortion0 = 0;
    double distortion1 = 0;
    // bool start = position[0] == 0 && position[1] == 0 && position[2] == 0 && position[3] == 0;
    // bool smallRelevant = position[0] <4  && position[1] < 4  && position[2] < 4 && position[3] < 4 && length[0] ==9 && length[1] == 9 && length[2] == 64 && length[3] == 64;
    // bool relevant = (start && smallRelevant)&& false;
    // if(relevant)std::cout<<"HELLO HELLO HELLO"<<std::endl;
    //lambda = 0;
    double J0, J1;
    double SignalEnergySum;
    double distortionSum = 0;
    ProbabilityModel currentProbabilityModel[NUMBER_OF_MODELS];
    //std::cout<<bitplane<<" "<<mInferiorBitPlane<<std::endl;
    //std::cout<<length[0]<<length[1]<<length[2]<<length[3]<<std::endl;
    int64_t length_t = length[0];
    int64_t length_s = length[1];
    int64_t length_v = length[2];
    int64_t length_u = length[3];

    int64_t position_t = position[0];
    int64_t position_s = position[1];
    int64_t position_v = position[2];
    int64_t position_u = position[3];
    //std::cout<<"bitplane = "<<bitplane<<std::endl;
    //std::cout<<length_t<<" "<<length_s<<" "<<length_v<<" "<<length_u<<std::endl;

    int* data = mSubbandLF_.data.data_ptr<int>();
    if(bitplane < mInferiorBitPlane) {
        signalEnergy = 0;
        for(int index_t = 0; index_t < length_t; index_t++) {
            for(int index_s = 0; index_s < length_s; index_s++) {
                for(int index_v = 0; index_v < length_v; index_v++) {
                    for(int index_u = 0; index_u < length_u; index_u++) {
                        int coefficient = data[mSubbandLF_.LinearPosition(position_t+index_t,position_s+index_s,position_v+index_v,position_u+index_u)];
                        //int coefficient = mSubbandLF_.data[position_t+index_t][position_s+index_s][position_v+index_v][position_u+index_u].item<int>();
                        //if(relevant && length_t < 4 && length_s < 4 && length_v < 4 && length_u < 4) std::cout<<"Same Check "<<"("<<position_t+index_t<<","<<position_s+index_s<<","<<position_v+index_v<<","<<position_u+index_u<<") "<<coefficient<<std::endl;
                        J0 = coefficient;
                        signalEnergy += J0*J0;
                    }
                }
            }
        }
        rate += 0;
        distortion += signalEnergy;
        //std::cout<<"skipping stuff"<<std::endl;
        return (signalEnergy);
    }
    if(length_t*length_s*length_v*length_u == 1) {
        //evaluate the cost to encode coefficient
        int magnitude = data[mSubbandLF_.LinearPosition(position_t,position_s,position_v,position_u)];
        //int magnitude = mSubbandLF_.data[position_t][position_s][position_v][position_u].item<int>();
        int signal = 0;
        if(magnitude < 0) {
            magnitude = -magnitude;
            signal = 1;
        }
        int allZeros = 1;
        
        signalEnergy = magnitude;
                
        signalEnergy *= signalEnergy;
        
        int onesMask = 0;
        onesMask = ~onesMask;

        double J = 0;
        double accumulatedRate = 0;
        for(int bit_position = bitplane; bit_position >= mInferiorBitPlane; bit_position--) {
            int bit = (magnitude >> bit_position)&01;
            accumulatedRate += mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].Rate(bit);
            if(bit_position > BITPLANE_BYPASS) 
                mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].UpdateModel(bit);
            
            if(bit == 1)
                allZeros = 0;
       }
        if(allZeros == 0)
            accumulatedRate += 1.0;

        int bitMask = onesMask << mInferiorBitPlane;
        int quantizedMagnitude = magnitude&bitMask;
        if(allZeros == 0) {
            quantizedMagnitude += (1 << mInferiorBitPlane)/2;
        }  
        J = magnitude - quantizedMagnitude;
        //std::cout<<"Cost: "<<J*J <<" "<<lambda*(accumulatedRate)<<std::endl;
        distortion = J*J;
        J = J*J + lambda*(accumulatedRate);
        rate = accumulatedRate;



        //std::cout<<"Magnitude = "<<magnitude<<" quantizedMagnitude = "<<quantizedMagnitude<<" Distortion = "<<distortion<<" rate = "<<accumulatedRate<<" weighed rate"<<lambda*accumulatedRate<<std::endl;
        //std::cout<<"Compression Energy = "<< J<< " Ignoring Energy  = "<<signalEnergy<<std::endl;
        
        *codeString[0] = 0;
        
        //std::cout<<"We have split everything down to a 1x1 block! Energy equals: "<<J<<" CodeString equals 0!"<<std::endl;
        //rate += accumulatedRate;
        //distortion+=J*J;
        //std::cout<<"compressing stuff;"<<std::endl;      

        return(J);
    }

    for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++)
        currentProbabilityModel[model_index].CopyModel(&mOptimizationPmodel[model_index]);


    char *codeString_0 = new char [2];
    strcpy(codeString_0, "");
    
    int Significance = 0;
    
    int Threshold = 1 << bitplane;

    //std::cout<<"Threshold = "<<Threshold<<std::endl;
    for(int index_t = position_t; index_t < position_t+length_t; index_t++) {
        
        for(int index_s = position_s; index_s < position_s+length_s; index_s++) {
            
            for(int index_v = position_v; index_v < position_v+length_v; index_v++) {
                
                for(int index_u = position_u; index_u < position_u+length_u; index_u++) {
                    
                    if((index_t < mSubbandLF_.data.size(0))&&(index_s < mSubbandLF_.data.size(1))&&(index_v < mSubbandLF_.data.size(2))&&(index_u < mSubbandLF_.data.size(3))) {
                        int magnitude = data[mSubbandLF_.LinearPosition(index_t,index_s,index_v,index_u)];

                        if(magnitude >= Threshold) Significance = 1;
                        if(magnitude <= -Threshold) Significance = 1;
                        if(Significance == 1) {
                            index_t = position_t+length_t;
                            index_s = position_s+length_s;
                            index_v = position_v+length_v;
                            index_u = position_u+length_u;
                        }
                    
                    }
                    
                }
                
            }
            
        }
        
    }
     //evaluate the cost of segmentation flags
    J0 = lambda*mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].Rate(0);
    J0 += lambda*mOptimizationPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].Rate(Significance);
    rate0+=J0/lambda;
   

    J1 = lambda*mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].Rate(1);
    rate1+=J1/lambda;

    if(bitplane > BITPLANE_BYPASS_FLAGS) {
        mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(0);
        mOptimizationPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].UpdateModel(Significance);
    }   
    if(Significance == 0) {
        //std::cout<<"No Significant Bits in this bitplane. Continuing with lower bitplane."<<std::endl;
        double rateTemp = 0;
        double distortionTemp = 0;
        J0 += RdOptimizeHexadecaTree_({position_t, position_s, position_v, position_u}, {length_t, length_s, length_v, length_u}, lambda, bitplane-1, &codeString_0, SignalEnergySum,rateTemp,distortionTemp);
        rate0+=rateTemp;    
        distortion0+=distortionTemp;

        

            //std::cout<<"Rate0 = "<<rate0*lambda<<" "<<"distortion0 = "<<distortion0<<" J0 = "<<J0<<" "<<rate0*lambda+distortion0<<std::endl;

    }
    else {

        SignalEnergySum = 0;
        double Energy;

        int half_length_t = (length_t > 1) ? length_t/2 : 1;
        int half_length_s = (length_s > 1) ? length_s/2 : 1;
        int half_length_v = (length_v > 1) ? length_v/2 : 1;
        int half_length_u = (length_u > 1) ? length_u/2 : 1;
        
        int number_of_subdivisions_t = (length_t > 1) ? 2 : 1;
        int number_of_subdivisions_s = (length_s > 1) ? 2 : 1;
        int number_of_subdivisions_v = (length_v > 1) ? 2 : 1;
        int number_of_subdivisions_u = (length_u > 1) ? 2 : 1;
        
        //std::cout<<"Significant Bits!!! Dividing block!"<<std::endl;

        
        for(int index_t = 0; index_t < number_of_subdivisions_t; index_t++) {
            
            for(int index_s = 0; index_s < number_of_subdivisions_s; index_s++) {
                
                for(int index_v = 0; index_v < number_of_subdivisions_v; index_v++) {
                    
                    for(int index_u = 0; index_u < number_of_subdivisions_u; index_u++) {
                        
                        int new_position_t = position_t+index_t*half_length_t;
                        int new_position_s = position_s+index_s*half_length_s;
                        int new_position_v = position_v+index_v*half_length_v;
                        int new_position_u = position_u+index_u*half_length_u;
                        
                        int new_length_t = (index_t == 0) ? half_length_t : (length_t-half_length_t);
                        int new_length_s = (index_s == 0) ? half_length_s : (length_s-half_length_s);
                        int new_length_v = (index_v == 0) ? half_length_v : (length_v-half_length_v);
                        int new_length_u = (index_u == 0) ? half_length_u : (length_u-half_length_u);
                        
                        char *codeString_1 = new char [2];
                            
                        strcpy(codeString_1, "");
                        //std::cout<<"Beginning: "<< rate0<<" "<<distortion0<<" "<<J0<<std::endl;
                        double rateTemp = 0;
                        double distortionTemp = 0;
                        J0 += RdOptimizeHexadecaTree_({new_position_t, new_position_s, new_position_v, new_position_u},{ new_length_t, new_length_s, new_length_v, new_length_u}, lambda, bitplane, &codeString_1, Energy,rateTemp,distortionTemp);
                        rate0+=rateTemp;
                        distortion0+=distortionTemp;
                        if(position_t == 0 && position_s == 0 && position_v == 0 && position_u == 0) {
                            if(length_v == 32 && length_u == 32) {
                                std::cout<<"("<<new_position_t<<" "<<new_position_s<<" "<<new_position_v<<" "<<new_position_u<<") "<<rateTemp/8<<" "<<distortionTemp<<" "<<J0<<std::endl;
                            }
                        }
                        
                        //std::cout<<"Beginning: "<< rate0<<" "<<distortion0<<" "<<J0<<" "<<rate0*lambda + distortion0<<std::endl;

                        //if(relevant) std::cout<<"counter = "<<counter<<" p:"<<new_position_t<<" "<<new_position_s<<" "<<new_position_v<<" "<<new_position_u<<" Cumm = "<<J0<<std::endl;
                        char *tempString = new char[strlen(codeString_0)+strlen(codeString_1)+2];
                        strcpy(tempString, codeString_0);
                        strcat(tempString, codeString_1);
                        delete [] codeString_0;
                        delete [] codeString_1;
                        codeString_0 = tempString;
                            
                        SignalEnergySum += Energy;

                    }
                    
                }
                
            }
            
        }         
    }    

    //evaluate the cost J1 to skip this subblock
    J1 += SignalEnergySum;
    distortion1 += SignalEnergySum;

    //Choose the lowest cost
    if((J0 < J1)||((bitplane == mInferiorBitPlane)&&(Significance == 0))) {
        //std::cout<<"j0 < j1"<<std::endl;
        //std::cout<<"J0 = "<<J0<<" Calc J0 = "<< lambda*rate0 + distortion0<<std::endl;

        rate = rate0;
        distortion = distortion0;

        char *tempString = new char[strlen(*codeString)+strlen(codeString_0)+3];
        strcpy(tempString, *codeString);
        delete [] *codeString;
        *codeString = tempString;
        
        if(Significance == 1) {
            strcat(*codeString, "1");         
        }
        else {
            strcat(*codeString, "0");
        }
        strcat(*codeString, codeString_0);
    }
    else {
        //std::cout<<"j0 > j1"<<std::endl;

        rate += rate1;   
        distortion = distortion1;     
        char *tempString = new char[strlen(*codeString)+3];
        strcpy(tempString, *codeString);
        delete [] *codeString;
        *codeString = tempString;
        strcat(*codeString, "2");
        J0 = J1;
        
        for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++)
            mOptimizationPmodel[model_index].CopyModel(&currentProbabilityModel[model_index]);
        
        if(bitplane > BITPLANE_BYPASS_FLAGS) 
            mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(1);
    }
    //std::cout<<"rate = "<<rate<<std::endl;
    //std::cout<<"J0 = "<<J0/lambda<<std::endl;
    delete [] codeString_0;
    
    signalEnergy = SignalEnergySum;
    //std::cout<<"J0 = "<<J0<<" Calc J0 = "<< lambda*rate + distortion<<" rate = "<<rate<<" distortion = "<<distortion<<" lambda = "<<lambda<<std::endl;

        
    return(J0);    
    




}

void Hierarchical4DEncoder :: RdEncodeHexadecatree_(std::array<int64_t,4> position, std::array<int64_t,4> length, int bitplane, int &flagIndex) {
    if(bitplane < mInferiorBitPlane) {
        return;
    }
    //If the block is a single bit long Encode the bit and return
    if(length[0]*length[1]*length[2]*length[3] == 1) {
        //rd encode coefficient     
        //std::cout<<"Encoded Coefficient"<<std::endl;   
        EncodeCoefficient(mSubbandLF_.data[position[0]][position[1]][position[2]][position[3]].item<int>(), bitplane);
        return;
    }
    //std::cout<<mSegmentationTreeCodeBuffer[flagIndex]<<std::endl;
    //First function call starts with flagIndex = 0, meaning we read the content of mSegmentationTreeCodeBuffer at 0
    //If mSegmentationTreeCodeBuffer is zero, we encode the segmentationflag and increment the index
    //we run the function recurseviley with bitplane -1;
    //std::cout<<mSegmentationTreeCodeBuffer[flagIndex]<<std::endl;
    if(mSegmentationTreeCodeBuffer[flagIndex] == '0') {
        
        EncodeSegmentationFlag(0, bitplane);
        flagZero++;
    
        flagIndex++;
        RdEncodeHexadecatree_(position, length, bitplane-1, flagIndex);
        
        return;
    }
    //If the Flag is 2 seems to just Encode the flag itself and get out This does not call the function recursively.
    
    if(mSegmentationTreeCodeBuffer[flagIndex] == '2') {
        flagTwo++;
        ignored.index({at::indexing::Slice(position[0],position[0]+length[0]),at::indexing::Slice(position[1],position[1]+length[1]),at::indexing::Slice(position[2],position[2]+length[2]),at::indexing::Slice(position[3],position[3]+length[3])}) = flagIndex*at::ones(length,at::kInt);
        EncodeSegmentationFlag(2, bitplane);
    
        flagIndex++;

        return;
    }
    //If Flag is 1 we divide the block in half in each of it's 4 subblocks and rerun the function.
    if(mSegmentationTreeCodeBuffer[flagIndex] == '1') {
        
        EncodeSegmentationFlag(1, bitplane);
        flagOne++;
        flagIndex++;
        
        std::array<int64_t,4> half_length;
        std::array<int64_t,4> number_of_subdivisions;
        for(int i = 0; i<4; i++){
            half_length[i] = (length[i] > 1) ? length[i]/2 : 1;
            number_of_subdivisions[i] = (length[i] > 1) ? 2 : 1;
        }
        
        
        
        for(int index_t = 0; index_t < number_of_subdivisions[0]; index_t++) {
            
            for(int index_s = 0; index_s < number_of_subdivisions[1]; index_s++) {
                
                for(int index_v = 0; index_v < number_of_subdivisions[2]; index_v++) {
                    
                    for(int index_u = 0; index_u < number_of_subdivisions[3]; index_u++) {
                        std::array<int64_t,4> new_position;
                        std::array<int64_t,4> new_length;
                        std::array<int64_t,4> index = {index_t,index_s,index_v,index_u};
                        for(int i = 0; i < 4; i++){
                            new_position[i] = position[i]+index[i]*half_length[i];
                            new_length[i] = (index[i] == 0) ? half_length[i] : (length[i] - half_length[i]);
                        }
                        RdEncodeHexadecatree_(new_position, new_length, bitplane, flagIndex);
                   }
                    
                }
                
            }
            
        }
        
        return;
    }
   
}

void Hierarchical4DEncoder :: EncodeCoefficient(int coefficient, int bitplane) {
    
    int signal = 0;
    int allZeros = 1;
    int magnitude = coefficient;
    if(magnitude < 0) {
        magnitude = -magnitude;
        signal = 1;
    }
    
    for(int bit_position = bitplane; bit_position >= mInferiorBitPlane; bit_position--) {

        int bit = (magnitude >> (bit_position))&01;
        mEntropyCoder.EncodeBit(bit, mPmodel[bit_position+mSymbolProbabilityModelIndex]);
        if(bit_position > BITPLANE_BYPASS) 
            mPmodel[bit_position+mSymbolProbabilityModelIndex].UpdateModel(bit);        
         if(bit == 1) {
            allZeros = 0;
        }
    }
    if(allZeros == 0) {
        mEntropyCoder.EncodeBit(signal, mPmodel[0]);
    }
    
    int quantizedMagnitude = magnitude >> mInferiorBitPlane;
    quantizedMagnitude = quantizedMagnitude << mInferiorBitPlane;
    if(allZeros == 0) {
        quantizedMagnitude += (1 << mInferiorBitPlane)/2;
    }
    double D = magnitude-quantizedMagnitude;
    //std::cout<<"magnitude = "<<magnitude<<" quantizedMagnitude = "<<quantizedMagnitude<<" D = "<<D<<std::endl; 
}

void Hierarchical4DEncoder :: EncodeSegmentationFlag(int flag, int bitplane) {
        
    if(flag == 0) {
        mEntropyCoder.EncodeBit(0, mPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex]);
        mEntropyCoder.EncodeBit(0, mPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex]);
        if(bitplane > BITPLANE_BYPASS_FLAGS) {
            mPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(0);
            mPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].UpdateModel(0);
        }
    }
    if(flag == 1) {
        mEntropyCoder.EncodeBit(0, mPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex]);
        mEntropyCoder.EncodeBit(1, mPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex]);
        if(bitplane > BITPLANE_BYPASS_FLAGS) {
            mPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(0);
            mPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].UpdateModel(1);
        }
    }
    if(flag == 2) {

        mEntropyCoder.EncodeBit(1, mPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex]);
        if(bitplane > BITPLANE_BYPASS_FLAGS) 
            mPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(1);
    }
    
}

void Hierarchical4DEncoder :: EncodePartitionFlag(int symbol) {
     
    if(symbol == 2) {
        mEntropyCoder.EncodeBit(1, mPmodel[0]);
        mEntropyCoder.EncodeBit(1, mPmodel[0]);
    }
    if(symbol == 1) {
        mEntropyCoder.EncodeBit(1, mPmodel[0]);
        mEntropyCoder.EncodeBit(0, mPmodel[0]);
    }
    if(symbol == 0) {
        mEntropyCoder.EncodeBit(0, mPmodel[0]);
    }
}
void Hierarchical4DEncoder :: EncodeSSI_(SgtSideInfo ssi){
    //std::cout<<"Encode SSI: ";
    //ssi.print();
    int precisionRho = ssi.getRhoPrecision();
    int precisionD = ssi.getAnglePrecision();
    //std::cout<<"precision: "<<precisionD<<" "<<precisionRho<<std::endl;
    //EncodeInteger(5,1);
    EncodeInteger(ssi.getAngleVCode(),precisionD);
    EncodeInteger(ssi.getAngleHCode(),precisionD);

}


void Hierarchical4DEncoder :: EncodeInteger(int integerValue, int precision)  {

    for(int n = precision-1; n >= 0; n--) { 
        int bit = (integerValue >> n)&01;
        mEntropyCoder.EncodeBit(bit, mPmodel[0]);
    }
        
}

void Hierarchical4DEncoder :: DoneEncoding(void) {
    std::cout<<"0: "<<flagZero<<" 1: "<<flagOne<<" 2: "<<flagTwo<<std::endl;
    mEntropyCoder.Flush();      //flushes entropy encoder
    
}





int Hierarchical4DEncoder :: OptimumBitplaneFaster_(double lambda) {
    //std::cout<<"hello"<<std::endl;
    std::cout<<"Optimum Calc Lambda: "<<lambda<<std::endl;
    long int subbandSize = mSubbandLF_.data.numel(); 
   // std::cout<<"subbandSize = "<<subbandSize<<std::endl;
    double Jmin=0;            //Irrelevant initial value
    int optimumBitplane=0;    //Irrelevant initial value
    
    double accumulatedRate = 0;
    int* flattened_data = mSubbandLF_.data.data_ptr<int>();
    for(int bit_position = mSuperiorBitPlane; bit_position >= 0; bit_position--) {
        
        double distortion = 0.0;
        double coefficientsDistortion = 0.0;
        double signalRate = 0.0;
        double J;
        int numberOfCoefficients = 0;
        
        int onesMask = 0;
        onesMask = ~onesMask;
        int bitMask = onesMask << bit_position;
        int Threshold = (1 << bit_position);
        
        //For Each SGT Coefficient
        for(long int coefficient_index=0; coefficient_index < subbandSize; coefficient_index++) {
            //Calc the magnitude
            int magnitude = flattened_data[coefficient_index];
            if(magnitude < 0) {
                magnitude = -magnitude;
            }
            //If the coefficient is represented in the bitplane
            if(magnitude >= Threshold) {
                //Get the actual bit to be represented
                int bit = (magnitude >> bit_position)&01;
                //Calculate the rate used to compress this bit and update model
                accumulatedRate += mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].Rate(bit);
                mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].UpdateModel(bit);
                //calculate the quantized magnitude, if the magnitude is over 0, the signalRate is incremented.
                int quantizedMagnitude = magnitude&bitMask;
                if(quantizedMagnitude > 0) {
                    signalRate += 1.0;
                }
                numberOfCoefficients++;
            }
            //Calculate the quantized magnitude again! Why isn't this just calculated once?
            int quantizedMagnitude = magnitude&bitMask;
            //If it's over 0, we add 2^(bit_position-1) Not sure why.
            if(quantizedMagnitude > 0) {
                quantizedMagnitude += (1 << bit_position)/2;
            }  
            //Calculate Distortion
            double magnitude_error = magnitude - quantizedMagnitude;
            distortion += magnitude_error*magnitude_error;
            if(magnitude >= (1 << bit_position)) {
                coefficientsDistortion += magnitude_error*magnitude_error;
            }  
        }
           
        J = distortion + lambda*(accumulatedRate + signalRate);
        
        if((J <= Jmin)||(bit_position == mSuperiorBitPlane)) {
            Jmin = J;
            optimumBitplane = bit_position;
        }
       
    }

    
    return(optimumBitplane);
}

int Hierarchical4DEncoder :: OptimumBitplane_(double lambda) {
    std::chrono::time_point<std::chrono::steady_clock> starter;
    std::chrono::time_point<std::chrono::steady_clock> ender;
    starter = std::chrono::steady_clock::now();
    
    long int subbandSize = mSubbandLF_.data.numel(); 
    double Jmin=0;            //Irrelevant initial value
    int optimumBitplane=0;    //Irrelevant initial value
    
    double accumulatedRate = 0;
    at::Tensor flattened_data = mSubbandLF_.data.flatten();
    for(int bit_position = mSuperiorBitPlane; bit_position >= 0; bit_position--) {
        
        double distortion = 0.0;
        double coefficientsDistortion = 0.0;
        double signalRate = 0.0;
        double J;
        int numberOfCoefficients = 0;
        
        int onesMask = 0;
        onesMask = ~onesMask;
        int bitMask = onesMask << bit_position;
        int Threshold = (1 << bit_position);
        

       //only flatten once (maybe faster?)
        for(long int coefficient_index=0; coefficient_index < subbandSize; coefficient_index++) {
        
            int magnitude = flattened_data[coefficient_index].item<int>();
            if(magnitude < 0) {
                magnitude = -magnitude;
            }
            if(magnitude >= Threshold) {
                int bit = (magnitude >> bit_position)&01;
                accumulatedRate += mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].Rate(bit);
                mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].UpdateModel(bit);
                int quantizedMagnitude = magnitude&bitMask;
                if(quantizedMagnitude > 0) {
                    signalRate += 1.0;
                }
                numberOfCoefficients++;
            }
            int quantizedMagnitude = magnitude&bitMask;
            if(quantizedMagnitude > 0) {
                quantizedMagnitude += (1 << bit_position)/2;
            }  
            double magnitude_error = magnitude - quantizedMagnitude;
            
            distortion += magnitude_error*magnitude_error;
            if(magnitude >= (1 << bit_position)) {
                coefficientsDistortion += magnitude_error*magnitude_error;
            }  
        }
           
        J = distortion + lambda*(accumulatedRate + signalRate);

        //std::cout<<bit_position<<": "<<J<<" = "<<distortion<<" + "<<lambda*(accumulatedRate + signalRate)<<std::endl;
        
        if((J <= Jmin)||(bit_position == mSuperiorBitPlane)) {
            Jmin = J;
            optimumBitplane = bit_position;
        }
       
    }
    ender = std::chrono::steady_clock::now();
    //std::cout<<"TIMER = "<<std::chrono::duration_cast<std::chrono::nanoseconds>(ender - starter).count()/1e6<<"ms"<<std::endl;
    
    return(optimumBitplane);
}

void Hierarchical4DEncoder :: GetOptimizerProbabilisticModelState(ProbabilityModel **state) {
       
    ProbabilityModel *pmodelArray = new ProbabilityModel [NUMBER_OF_MODELS];
    for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++) {
        pmodelArray[model_index].CopyModel(&mOptimizationPmodel[model_index]);    
    }
    *state = pmodelArray;
}

void Hierarchical4DEncoder :: SetOptimizerProbabilisticModelState(ProbabilityModel *state) {

     for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++) {
        mOptimizationPmodel[model_index].CopyModel(&state[model_index]);    
    }
  
}

void Hierarchical4DEncoder :: DeleteProbabilisticModelState(ProbabilityModel *state) {
    
    if(state != NULL) {
        delete [] state;
        state = NULL;
    }
     
}

void Hierarchical4DEncoder :: LoadOptimizerState(void) {

     for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++) {
        mOptimizationPmodel[model_index].CopyModel(&mPmodel[model_index]);    
    }
  
}

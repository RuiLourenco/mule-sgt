#include "Encoder/Hierarchical4DEncoder.h"
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <bitset>
/*******************************************************************************/
/*                        Hierachical4DEncoder class methods                   */
/*******************************************************************************/

Hierarchical4DEncoder :: Hierarchical4DEncoder(int height, int width)
    : mProcessingContext(height, width, 30)
    {

    mSuperiorBitPlane = 30;
    mInferiorBitPlane = 0;
    mPreSegmentation = 1;
    mSegmentationTreeCodeBuffer = "";
    mSegmentationTreeCodeBufferSize = 0;
    mSegmentationFlagProbabilityModelIndex = SEGMENTATION_PROB_MODEL_INDEX;
    mSymbolProbabilityModelIndex = SYMBOL_PROBABILITY_MODEL_INDEX;
    
}
Hierarchical4DEncoder :: ~Hierarchical4DEncoder(void) {

}



void Hierarchical4DEncoder :: StartEncoder(FILE *outputFilePointer) {
    
    
    mEntropyCoder.InitEncoder(outputFilePointer);  //opens output file

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


bool isElement(std::array<int64_t,4> length){
    int count = 1;
    for(int i = 0; i < 4; i++) {
        count*= length[i];
    }
    if(count == 1) {
        return true;    
    }
    return false;
}

CostResults Hierarchical4DEncoder::splitInFour(uint32_t current_node_idx,std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane){
    std::array<int64_t,4> half_length;
    std::array<int64_t,4> number_of_subdivisions;
    std::array<int64_t,4> index;
    double J0 = 0.0;
    double J1 = 0.0;
    mProcessingContext.nodePool[current_node_idx].decision = '1'; // 'L' for lower bitplane

    for (int i = 0; i < 4; i++){
        half_length[i] = (length[i] > 1) ? length[i]/2 : 1;
        number_of_subdivisions[i] = (length[i] > 1) ? 2 : 1;
    }
    
    int j = 0;
    for(index[0] = 0; index[0] < number_of_subdivisions[0]; index[0]++)        
    for(index[1] = 0; index[1] < number_of_subdivisions[1]; index[1]++)        
    for(index[2] = 0; index[2] < number_of_subdivisions[2]; index[2]++)     
    for(index[3] = 0; index[3] < number_of_subdivisions[3]; index[3]++) { 
        
        std::array<int64_t,4> new_position; 
        std::array<int64_t,4> new_length;                      
        for(int i = 0; i < 4; i++) {
            new_position[i]  = position[i] + index[i] * half_length[i];
            new_length[i] = (index[i] == 0) ? half_length[i] : (length[i] - half_length[i]);
        }

        uint32_t childNodeIdx =  mProcessingContext.add_default_node();
        mProcessingContext.nodePool[current_node_idx].children_idx[j++] = childNodeIdx;
        
        build_from_node(childNodeIdx, new_length, new_position, bitplane);
        J0 += mProcessingContext.nodePool[childNodeIdx].costResults.cost;
        J1 += mProcessingContext.nodePool[childNodeIdx].costResults.signalEnergy;
    }
    return CostResults{J0, J1};                  
}

CostResults Hierarchical4DEncoder::calculateTotalEnergy(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane) {
    CostResults costResult;
    int* data = mSubbandLF_.data.data_ptr<int>();
    std::array<int64_t,4> index;
    double signalEnergy = 0;
    for(index[0] = 0; index[0] < length[0]; index[0]++)
    for(index[1] = 0; index[1] < length[1]; index[1]++)
    for(index[2] = 0; index[2] < length[2]; index[2]++) 
    for(index[3] = 0; index[3] < length[3]; index[3]++) {
        int coefficient = data[mSubbandLF_.LinearPosition(position[0]+index[0],position[1]+index[1],position[2]+index[2],position[3]+index[3])];
        double J0 = coefficient;
        signalEnergy += J0*J0;
    }
    costResult.cost = signalEnergy;
    costResult.signalEnergy = signalEnergy;
    return costResult;
}



// Note: Tried Torch and OpenMP parallel versions. The simple serial loop is faster on average due to frequent early-exits and small block sizes.
bool Hierarchical4DEncoder::checkSignificance(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane){
    int* data = mSubbandLF_.data.data_ptr<int>();
    int threshold = 1 << bitplane;
    
    // Pre-compute bounds
    int64_t end0 = std::min(position[0] + length[0], mSubbandLF_.data.size(0));
    int64_t end1 = std::min(position[1] + length[1], mSubbandLF_.data.size(1));
    int64_t end2 = std::min(position[2] + length[2], mSubbandLF_.data.size(2));
    int64_t end3 = std::min(position[3] + length[3], mSubbandLF_.data.size(3));
    
    for(int64_t i0 = position[0]; i0 < end0; i0++)
    for(int64_t i1 = position[1]; i1 < end1; i1++)
    for(int64_t i2 = position[2]; i2 < end2; i2++)
    for(int64_t i3 = position[3]; i3 < end3; i3++) {
        int magnitude = abs(data[mSubbandLF_.LinearPosition(i0,i1,i2,i3)]);
        if(magnitude >= threshold) return true;
    }
    return false;
}



CostResults Hierarchical4DEncoder::calculateElementCost(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane) {
    CostResults costResult;
    int* data = mSubbandLF_.data.data_ptr<int>();
    int magnitude = data[mSubbandLF_.LinearPosition(position[0],position[1],position[2],position[3])]; 
    int signal = 0;
    if(magnitude < 0) {
        magnitude = -magnitude;
        signal = 1;
    }
    int allZeros = 1;          
    int onesMask = 0;
    onesMask = ~onesMask;
    double J = 0;
    double accumulatedRate = 0;
    for(int bit_position = bitplane; bit_position >= mInferiorBitPlane; bit_position--) {
        int bit = (magnitude >> bit_position)&01;
        accumulatedRate += mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].Rate(bit);
        if (position[2] <= 8 && position[3] <= 8){
            if(bit_position == bitplane){
                //std::cout<<"Bitplane: "<<bitplane<<" bit: "<<bit<<" rate: "<<mOptimizationPmodel[bit_position+mSymbolProbabilityModelIndex].Rate(bit)<<" accumulatedRate: "<<accumulatedRate<<std::endl;
            }
        }
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
    double distortion = J*J;
    costResult.cost = J*J + mLambda*(accumulatedRate);
    costResult.signalEnergy = (double) magnitude * (double) magnitude;

    return costResult;
}

double Hierarchical4DEncoder::build_optimal_tree_from_pool(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane, double lambda){
    mLambda = lambda;
    mProcessingContext.resetCounter();
    int firstNodeIdx = mProcessingContext.add_default_node();
    this->mSubbandLF_.data = this->mSubbandLF_.data.contiguous();
    //std::cout<<"Building optimal tree from pool with length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<" and position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<" and bitplane: "<<bitplane<<std::endl;
    build_from_node(firstNodeIdx, length, position, bitplane);
    double J0 = mProcessingContext.nodePool[firstNodeIdx].costResults.cost;
    return J0;
}

void Hierarchical4DEncoder::build_from_node(uint32_t current_node_idx,std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane) {    
    ProbabilityModelCollection currentProbabilityModel;
    double J0 = 0.0, J1 = 0.0; 


    // --- 1. Base Case: Reached a single pixel ---
    if (bitplane < mInferiorBitPlane) {



        mProcessingContext.nodePool[current_node_idx].decision = 'L'; // 'L' for Low Energy
        mProcessingContext.nodePool[current_node_idx].costResults = calculateTotalEnergy(length, position, bitplane);
        return;
    }
    
    if (isElement(length)) {
        

 
        
        mProcessingContext.nodePool[current_node_idx].decision = 'T'; // 'T' for Terminal
        mProcessingContext.nodePool[current_node_idx].costResults = calculateElementCost(length, position, bitplane);

        
        return;
    }
    
    currentProbabilityModel = mOptimizationPmodel;

    
    int significance = checkSignificance(length,position, bitplane);
    
    
    J0 = mLambda*mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].Rate(0);
    J0 += mLambda*mOptimizationPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].Rate(significance);
    J1 = mLambda*mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].Rate(1);
    double signalEnergy = 0;

    if(bitplane > BITPLANE_BYPASS_FLAGS) {
        mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(0);
        mOptimizationPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].UpdateModel(significance);
    }   
    if(significance == 0) {
        
        double rateTemp = 0;
        double distortionTemp = 0;
        mProcessingContext.nodePool[current_node_idx].decision = '0'; // 'L' for lower bitplane
        uint32_t childNodeIdx = mProcessingContext.add_default_node();
        mProcessingContext.nodePool[current_node_idx].children_idx[0] = childNodeIdx;
        build_from_node(childNodeIdx, length, position, bitplane - 1);

        J0 += mProcessingContext.nodePool[childNodeIdx].costResults.cost;
        signalEnergy += mProcessingContext.nodePool[childNodeIdx].costResults.signalEnergy;

    }
    else {

        CostResults costResult = splitInFour(current_node_idx, length, position, bitplane);
        J0 += costResult.cost;
        signalEnergy += costResult.signalEnergy;
        if(position[0] == 0 && position[1] == 0 && position[2] <= 5 && position[3] <= 5) {
        }
    }   
    J1 += signalEnergy;
    
    if((J0 < J1)||((bitplane == mInferiorBitPlane)&&(significance == 0))) {
             mProcessingContext.nodePool[current_node_idx].costResults.cost = J0; 
        }
        else {
            
            mProcessingContext.nodePool[current_node_idx].costResults.cost = J1; 
            mProcessingContext.nodePool[current_node_idx].decision = '2';       
            mOptimizationPmodel = currentProbabilityModel;
    
                
            if(bitplane > BITPLANE_BYPASS_FLAGS) 
                mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(1);
        }      
        mProcessingContext.nodePool[current_node_idx].costResults.signalEnergy = signalEnergy;
        return;  
}

void Hierarchical4DEncoder::encodeSubblockFromPool(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane, double lambda){
    if(length[0] * length[1] * length [2] * length[3] == 0){
        //std::cout<<"Correctly skipping empty subblock encoding"<<std::endl;
        return; // Nothing to encode
    }
    int flagSearchIndex = 0;
    //std::cout<<"Encoding Subblock from Pool with length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<" and position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<" and bitplane: "<<bitplane<<std::endl;
    this->mSubbandLF_.data = this->mSubbandLF_.data.contiguous();
    this->currCost = build_optimal_tree_from_pool(length,position, bitplane, lambda);
    iterateEncoding(0, length, position, bitplane);
    //std::cout<<"Next Available Index: "<<mProcessingContext.next_available_idx<<std::endl;
    //std::cout<<std::endl;

    return;
}

void Hierarchical4DEncoder::iterateEncoding(uint32_t current_node_idx, std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane) {
    int* data = mSubbandLF_.data.data_ptr<int>();
    if (current_node_idx >= mProcessingContext.next_available_idx) {
        std::cout<<"Error: Current Node Index is out of bounds!"<<std::endl;
        abort();
    }
    
    if(mProcessingContext.nodePool[current_node_idx].decision == 'L'){
        return;
    }
    if(mProcessingContext.nodePool[current_node_idx].decision == 'T'){
        if(length[0]*length[1]*length[2]*length[3] != 1) {
            std::cout<<"Error: Trying to encode a non-element node as an element node!"<<std::endl;
            abort();
        }
        EncodeCoefficient(data[mSubbandLF_.LinearPosition(position[0],position[1],position[2],position[3])], bitplane);
        return;
    }
    //std::cout<<mProcessingContext.nodePool[current_node_idx].decision;

    if(mProcessingContext.nodePool[current_node_idx].decision == '0') {
        //std::cout<<"Lower Bitplane"<<std::endl;
        EncodeSegmentationFlag(0, bitplane);
        int childNodeIdx = mProcessingContext.nodePool[current_node_idx].children_idx[0];
        iterateEncoding(childNodeIdx, length, position, bitplane - 1);
        return;
    }
    if(mProcessingContext.nodePool[current_node_idx].decision == '1') {
        //std::cout<<"Splitting in Four"<<std::endl;

        EncodeSegmentationFlag(1, bitplane);
    
        
        std::array<int64_t,4> half_length;
        std::array<int64_t,4> number_of_subdivisions;
        for(int i = 0; i<4; i++){
            half_length[i] = (length[i] > 1) ? length[i]/2 : 1;
            number_of_subdivisions[i] = (length[i] > 1) ? 2 : 1;
        }
        int j = 0;
        std::array<int64_t,4> indexes;
        for(indexes[0] = 0; indexes[0] < number_of_subdivisions[0]; indexes[0]++)        
        for(indexes[1] = 0; indexes[1] < number_of_subdivisions[1]; indexes[1]++)        
        for(indexes[2] = 0; indexes[2] < number_of_subdivisions[2]; indexes[2]++)     
        for(indexes[3] = 0; indexes[3] < number_of_subdivisions[3]; indexes[3]++) { 
            std::array<int64_t,4> new_position; 
            std::array<int64_t,4> new_length;                      
            for(int i = 0; i < 4; i++) {
                new_position[i]  = position[i] + indexes[i] * half_length[i];
                new_length[i] = (indexes[i] == 0) ? half_length[i] : (length[i] - half_length[i]);
            }
        
            uint32_t childNodeIdx = mProcessingContext.nodePool[current_node_idx].children_idx[j++];
            iterateEncoding(childNodeIdx, new_length, new_position, bitplane);
        }
       
        return;
    }
    if(mProcessingContext.nodePool[current_node_idx].decision == '2') {
        EncodeSegmentationFlag(2, bitplane);
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
    //
    int precisionRho = ssi.getRhoPrecision();
    int precisionD = ssi.getAnglePrecision();
    //std::cout<<"precision: "<<precisionD<<" "<<precisionRho<<std::endl;
    //ssi.print();
    //EncodeInteger(5,1);
    
    EncodeInteger(ssi.getAngleVCode(),precisionD);
    EncodeInteger(ssi.getAngleHCode(),precisionD);
    //std::cout<<ADAPTIVE_RHO_CALC<<std::endl;
    #if ADAPTIVE_RHO_CALC == 1
    EncodeInteger(ssi.getRhoSCode(),precisionRho);
    EncodeInteger(ssi.getRhoTCode(),precisionRho);
    EncodeInteger(ssi.getRhoUCode(),precisionRho);
    EncodeInteger(ssi.getRhoVCode(),precisionRho);
    #endif
}


void Hierarchical4DEncoder :: EncodeInteger(int integerValue, int precision)  {

    for(int n = precision-1; n >= 0; n--) { 
        int bit = (integerValue >> n)&01;
        mEntropyCoder.EncodeBit(bit, mPmodel[0]);
    }
        
}

void Hierarchical4DEncoder :: DoneEncoding(void) {
    //std::cout<<"0: "<<flagZero<<" 1: "<<flagOne<<" 2: "<<flagTwo<<" Ignore Efficiency = "<<mIgnoreEfficiency<<" "<<mIgnored<<std::endl;
    mEntropyCoder.Flush();      //flushes entropy encoder
    
}





int Hierarchical4DEncoder :: OptimumBitplaneFaster_(double lambda) {

    long int subbandSize = mSubbandLF_.data.numel(); 
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



void Hierarchical4DEncoder::LoadOptimizerState(void) {
    mOptimizationPmodel = mPmodel; // Simple struct copy
}

void Hierarchical4DEncoder::RestoreOptimizerState(const ProbabilityModelCollection& collection) {
    mPmodel = collection; // Simple struct copy
}

ProbabilityModelCollection Hierarchical4DEncoder::GetOptimizerSnapshot() {
    return mPmodel; // This automatically creates a copy to return!
}

#include "Encoder/Hierarchical4DEncoder.h"
#include <string.h>
#include <stdlib.h>
#include <chrono>
#include <bitset>
/*******************************************************************************/
/*                        Hierachical4DEncoder class methods                   */
/*******************************************************************************/

Hierarchical4DEncoder :: Hierarchical4DEncoder(int height, int width): mProcessingContext(height, width, 30) {
    mSuperiorBitPlane = 30;
    mInferiorBitPlane = 0;
    mPreSegmentation = 1;
    mSegmentationTreeCodeBuffer = "";
    mSegmentationTreeCodeBufferSize = 0;
    mSegmentationFlagProbabilityModelIndex = SEGMENTATION_PROB_MODEL_INDEX;
    mSymbolProbabilityModelIndex = SYMBOL_PROBABILITY_MODEL_INDEX;
    mPmodel = NULL;
    mOptimizationPmodel = NULL;
    
}
Hierarchical4DEncoder :: ~Hierarchical4DEncoder(void) {
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
    this->mSubbandLF_.data = this->mSubbandLF_.data.contiguous();
    HexResult res = RdOptimizeHexadecaTree_({0, 0, 0, 0}, size, lambda, mSuperiorBitPlane, Energy, rate,distortion);
    mSegmentationTreeCodeBuffer = res.codeStream;
    this->currCost = res.cost;
    // std::cout<<"First Coeffs = "<<mSubbandLF_.data[0][0][0][0].item<int>()<<" "<<mSubbandLF_.data[0][0][31][31].item<int>()<<" "<<mSubbandLF_.data[0][0][31][0].item<int>()<<std::endl;
    // std::cout<<"absolute rate = "<<rate<<" "<<"absolute distortion = "<<distortion<<" Ratio = "<<distortion/(lambda * rate)<<std::endl;
    // std::cout<<"size = "<<size[0]<<" "<<size[1]<<" "<<size[2]<<" "<<size[3]<<std::endl;
    // std::cout<<"mSubbandLF size: "<<mSubbandLF_.size[0]<<" "<<mSubbandLF_.size[1]<<" "<<mSubbandLF_.size[2]<<" "<<mSubbandLF_.size[3]<<std::endl;
    // std::cout<<"mSubbandLF transform size: "<<mSubbandLF_.transformSize[0]<<" "<<mSubbandLF_.transformSize[1]<<" "<<mSubbandLF_.transformSize[2]<<" "<<mSubbandLF_.transformSize[3]<<std::endl;
    // std::cout<<"mSubbandLF lfPosition: "<<mSubbandLF_.lightFieldPosition[0]<<" "<<mSubbandLF_.lightFieldPosition[1]<<" "<<mSubbandLF_.lightFieldPosition[2]<<" "<<mSubbandLF_.lightFieldPosition[3]<<std::endl;
    // std::cout<<"lambda: "<<lambda<<" mSuperiorBitplane: "<<mSuperiorBitPlane<<" inferiorBitPlane = "<<mInferiorBitPlane<<std::endl;
    // std::cout<<"currCost = "<<currCost<<std::endl;
    this->mRate = rate/(size[0]*size[1]*size[2]*size[3]);
    this->mDistortion = distortion;

    //std::cout<<"Rate = "<<rate<< " WeightedRate = "<<lambda*rate<<"Distortion = "<<this->mDistortion<<" CurrentCost = "<<this->currCost<<" SumCheck = "<<lambda*rate + distortion<<std::endl;
    //std::cout<<"Calculated Rate = "<<(this->currCost - distortion)/lambda<<std::endl;
  
    flagSearchIndex = 0;
    RdEncodeHexadecatree_({0, 0, 0, 0}, size, mSuperiorBitPlane, flagSearchIndex);
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
    //std::cout<<"length = "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //std::cout<<number_of_subdivisions[0]<<" "<<number_of_subdivisions[1]<<" "<<number_of_subdivisions[2]<<" "<<number_of_subdivisions[3]<<std::endl;
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
        if(position[0] <=9 && position[1] <=9 && position[2] <= 9 && position[3] <= 9) {
            //std::cout<<"BEFORE RECURSION: "<<j<<": "<<new_length[2]<<" "<<new_length[3]<<" | "<<new_position[2]<<" "<<new_position[3]<<std::endl;
        }
        uint32_t childNodeIdx =  mProcessingContext.add_default_node();
        mProcessingContext.nodePool[current_node_idx].children_idx[j++] = childNodeIdx;
        
        build_from_node(childNodeIdx, new_length, new_position, bitplane);
        //std::cout<<"Child Node "<< childNodeIdx<<std::endl;
        J0 += mProcessingContext.nodePool[childNodeIdx].costResults.cost;
        J1 += mProcessingContext.nodePool[childNodeIdx].costResults.signalEnergy;
        if (position[2] <= 8 && position[3] <= 8 && length[2] <= 9 && length[3] <= 9) {
            //std::cout<<j-1<<":"<<new_length[2]<<" "<<new_length[3]<<" "<<mProcessingContext.nodePool[childNodeIdx].costResults.cost<<std::endl;
        }
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
        //int coefficient = mSubbandLF_.data[position_t+index_t][position_s+index_s][position_v+index_v][position_u+index_u].item<int>();
        //if(relevant && length_t < 4 && length_s < 4 && length_v < 4 && length_u < 4) std::cout<<"Same Check "<<"("<<position_t+index_t<<","<<position_s+index_s<<","<<position_v+index_v<<","<<position_u+index_u<<") "<<coefficient<<std::endl;
        double J0 = coefficient;
        signalEnergy += J0*J0;
    }
    costResult.cost = signalEnergy;
    costResult.signalEnergy = signalEnergy;
    return costResult;
}


bool Hierarchical4DEncoder::checkSignificance(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane){
    bool significance = false;
    int* data = mSubbandLF_.data.data_ptr<int>();

    
    int threshold = 1 << bitplane;
    // if(position[0] == 0 && position[1] == 0 && position[2] <= 2 && position[3] <= 2) {
    //     std::cout<<"Checking significance at bitplane: "<<bitplane<<" threshold: "<<threshold<<std::endl;
    // }
    int magnitudeSuperior = 0;
    //std::cout<<"Threshold = "<<Threshold<<std::endl;
    std::array<int64_t,4> index;
    for(index[0] = position[0]; index[0] < position[0]+length[0]; index[0]++)
    for(index[1] = position[1]; index[1] < position[1]+length[1]; index[1]++)
    for(index[2] = position[2]; index[2] < position[2]+length[2]; index[2]++) 
    for(index[3] = position[3]; index[3] < position[3]+length[3]; index[3]++) {            
        if((index[0] < mSubbandLF_.data.size(0))&&(index[1] < mSubbandLF_.data.size(1))&&(index[2] < mSubbandLF_.data.size(2))&&(index[3] < mSubbandLF_.data.size(3))) {
            int magnitude = data[mSubbandLF_.LinearPosition(index[0],index[1],index[2],index[3])];
            if (magnitude < 0) {
                magnitude = -magnitude;
            }
            if (magnitude > magnitudeSuperior){
                magnitudeSuperior = magnitude;
            }
            if(magnitude >= threshold) significance = true;
            // if(magnitude <= -threshold) significance = true;
            if(significance) {

                return significance;
            }
        }
    }
    // if(position[0] == 0 && position[1] == 0 && position[2] <= 2 && position[3] <= 2) {
    //     std::cout<<"Significance: "<<significance<< " "<<magnitudeSuperior<< " < "<<threshold<<std::endl;
    // }

    return significance;
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

    if (position[2] <= 8 && position[3] <= 8){
        // std::cout<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
        // std::cout<<"Magnitude:"<<magnitude<<" Quantized Magnitude: "<<quantizedMagnitude<<" Error: "<<magnitude - quantizedMagnitude<<std::endl;
        // std::cout<<"Element Cost: "<<costResult.cost<<" Signal Energy: "<<costResult.signalEnergy<<" Rate: "<<accumulatedRate<<" Distortion: "<<distortion<<" mLambda = "<<mLambda<<std::endl;
    }
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
    if(position[2] <= 5 && position[3] <= 5 && length[2] <= 5 && length[3] <= 5){ 
        //std::cout<<"Beginning: "<<length[2]<<" "<<length[3]<<std::endl;
    }
    
    ProbabilityModel currentProbabilityModel[NUMBER_OF_MODELS];
    double J0 = 0.0, J1 = 0.0;
    // --- 1. Base Case: Reached a single pixel ---
    if (bitplane < mInferiorBitPlane) {
        if(position[2] <= 5 && position[3] <= 5 && length[2] <= 5 && length[3] <= 5){
            //std::cout<<"Minimum Bit Plane! "<<std::endl;
        }


        mProcessingContext.nodePool[current_node_idx].decision = 'L'; // 'L' for Low Energy
        mProcessingContext.nodePool[current_node_idx].costResults = calculateTotalEnergy(length, position, bitplane);
        // node_pool[current_node_idx].cost = calculate_pixel_energy(); // Your logic here
        return;
    }
    if (isElement(length)) {
        if(position[2] <= 5 && position[3] <= 5 && length[2] <= 5 && length[3] <= 5){
            //std::cout<<"Elementary??? "<<std::endl;
        }

        // if (position[2] == 0 && position[3] == 0) {
        //     std::cout<<"Am I really a number? My length is: "<<length[2]<<" "<<length[3]<<std::endl;
        // }  
        
        mProcessingContext.nodePool[current_node_idx].decision = 'T'; // 'T' for Terminal
        mProcessingContext.nodePool[current_node_idx].costResults = calculateElementCost(length, position, bitplane);

        
        // node_pool[current_node_idx].cost = calculate_pixel_energy(); // Your logic here
        return;
    }
    

    //update probability model
    for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++){
        currentProbabilityModel[model_index].CopyModel(&mOptimizationPmodel[model_index]);
    }
    
    int significance = checkSignificance(length,position, bitplane);
    
   

     //evaluate the cost of segmentation flags and update model;
    J0 = mLambda*mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].Rate(0);
    J0 += mLambda*mOptimizationPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].Rate(significance);
    J1 = mLambda*mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].Rate(1);
    double signalEnergy = 0;

    if(bitplane > BITPLANE_BYPASS_FLAGS) {
        mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(0);
        mOptimizationPmodel[2*bitplane+1+mSegmentationFlagProbabilityModelIndex].UpdateModel(significance);
    }   
    if(significance == 0) {
        //std::cout<<"No Significant Bits in this bitplane. Continuing with lower bitplane."<<std::endl;
        
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
            //std::cout<<"Exited Splitting and ended all recursion!"<<std::endl;
        }
    }   
    J1 += signalEnergy;
    
    if((J0 < J1)||((bitplane == mInferiorBitPlane)&&(significance == 0))) {
            //std::cout<<"j0 < j1"<<std::endl;
            //std::cout<<"J0 = "<<J0<<" Calc J0 = "<< lambda*rate0 + distortion0<<std::endl;
            mProcessingContext.nodePool[current_node_idx].costResults.cost = J0; 
        }
        else {
            // if (position[2] <= 8 && position[3] <= 8 && length[2] <= 9 && length[3] <= 9) {
            //     std::cout<<"Comivos todos"<<std::endl;
            //     std::cout<<"Position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
            //     std::cout<<"Length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<< " Bitplane: "<<bitplane<<std::endl;
            //     std::cout<<J0<<" "<<J1<<" "<<mProcessingContext.nodePool[current_node_idx].decision<<std::endl<<std::endl;
            // }
            mProcessingContext.nodePool[current_node_idx].costResults.cost = J1; 
            mProcessingContext.nodePool[current_node_idx].decision = '2';       
            for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++)
                mOptimizationPmodel[model_index].CopyModel(&currentProbabilityModel[model_index]);
            
            if(bitplane > BITPLANE_BYPASS_FLAGS) 
                mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(1);
        }      
        mProcessingContext.nodePool[current_node_idx].costResults.signalEnergy = signalEnergy;
        // if (position[2] <= 8 && position[3] <= 8 && length[2] <= 9 && length[3] <= 9) {
        //     std::cout<<"Position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
        //     std::cout<<"Length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<< " Bitplane: "<<bitplane<<std::endl;
        //     std::cout<<J0<<" "<<J1<<" "<<mProcessingContext.nodePool[current_node_idx].decision<<std::endl;
        //     std::cout<<mOptimizationPmodel<<std::endl;
        // } 
        return;  
}

HexResult Hierarchical4DEncoder :: RdOptimizeHexadecaTree_(std::array<int64_t,4> position, std::array<int64_t,4> length, double lambda, int bitplane, double &signalEnergy, double &rate, double& distortion) {

    signalEnergy = 0;
    rate = 0;
    distortion = 0;

    double rate0 = 0;
    double rate1 = 0;
    double distortion0 = 0;
    double distortion1 = 0;

    double J0, J1;
    double SignalEnergySum;
    double distortionSum = 0;
    ProbabilityModel currentProbabilityModel[NUMBER_OF_MODELS];

    int64_t length_t = length[0];
    int64_t length_s = length[1];
    int64_t length_v = length[2];
    int64_t length_u = length[3];

    int64_t position_t = position[0];
    int64_t position_s = position[1];
    int64_t position_v = position[2];
    int64_t position_u = position[3];

    int* data = mSubbandLF_.data.data_ptr<int>();
    if(bitplane < mInferiorBitPlane) {
        
        if(length[2] < 9 && length[3] < 9 && position[2] < 9 && position[3] < 9){
            std::cout<<"Somehow I am blank"<<std::endl;
            std::cout<<"Position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
            std::cout<<"Length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<< " Bitplane: "<<bitplane<<std::endl;
        }
        if (position[2] < 2 && position[3] < 2) {
            std::cout<<"Somehow I am blank"<<std::endl;
        }  
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
        //std::cout<<"skipping stuff sized: "<<length[0]<<" "<<length[1] << " " <<length[2]<<" "<<length[3]<<std::endl;
        return HexResult{signalEnergy, ""};
    }
    if(length_t*length_s*length_v*length_u == 1) {
        //evaluate the cost to encode coefficient
        int magnitude = data[mSubbandLF_.LinearPosition(position_t,position_s,position_v,position_u)];
        // if(this->mSubbandLF_.lightFieldPosition[2] == 32 && this->mSubbandLF_.lightFieldPosition[3] == 32){
        //     if(position[2] < 8 && position[3] < 8){
        //         std::cout<<"Magnitude = "<<magnitude<<" "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
        //         std::cout<<"Tensor Direct = "<<this->mSubbandLF_.data[position[0]][position[1]][position[2]][position[3]].item<int>()<<" "<< mSubbandLF_.LinearPosition(position_t,position_s,position_v,position_u)<<std::endl;
            
        //     }
        // }
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
        distortion = J*J;
        J = J*J + lambda*(accumulatedRate);
        rate = accumulatedRate;

        if (position[2] <= 8 && position[3] <= 8){
            // std::cout<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
            // std::cout<<"Magnitude:"<<magnitude<<" Quantized Magnitude: "<<quantizedMagnitude<<" Error: "<<magnitude - quantizedMagnitude<<std::endl;
            // std::cout<<"Element Cost: "<<J<<" Signal Energy: "<<signalEnergy<<" Rate: "<<accumulatedRate<<" Distortion: "<<distortion<<" mLambda = "<<mLambda<<std::endl;
        }



        //std::cout<<"Magnitude = "<<magnitude<<" quantizedMagnitude = "<<quantizedMagnitude<<" Distortion = "<<distortion<<" rate = "<<accumulatedRate<<" weighed rate"<<lambda*accumulatedRate<<std::endl;
        //std::cout<<"Compression Energy = "<< J<< " Ignoring Energy  = "<<signalEnergy<<std::endl;
        
        //codeString = "";
        
        //std::cout<<"We have split everything down to a 1x1 block! Energy equals: "<<J<<" CodeString equals 0!"<<std::endl;
        //rate += accumulatedRate;
        //distortion+=J*J;
        //std::cout<<"compressing stuff;"<<std::endl;      

        return HexResult{J, ""};
    }

    for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++){
        currentProbabilityModel[model_index].CopyModel(&mOptimizationPmodel[model_index]);
    }


    std::string codeString_0 ="";    
    std::stringstream ss_for_J0;
    
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
        HexResult res = RdOptimizeHexadecaTree_({position_t, position_s, position_v, position_u}, {length_t, length_s, length_v, length_u}, lambda, bitplane-1, SignalEnergySum,rateTemp,distortionTemp);
        J0 += res.cost;
        ss_for_J0 << res.codeStream;
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
        //std::string codeString_1 =  "";
        //std::stringstream ss_for_J0;
        //ss_for_J0 << codeString_0;
        int j = 0;
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
                        
                        
                        //std::cout<<"Beginning: "<< rate0<<" "<<distortion0<<" "<<J0<<std::endl;
                        double rateTemp = 0;
                        double distortionTemp = 0;
                        HexResult res = RdOptimizeHexadecaTree_({new_position_t, new_position_s, new_position_v, new_position_u},{ new_length_t, new_length_s, new_length_v, new_length_u}, lambda, bitplane, Energy,rateTemp,distortionTemp);
                        J0 += res.cost;
                        ss_for_J0 << res.codeStream;
                        rate0+=rateTemp;
                        distortion0+=distortionTemp;
                        if (position[2] <= 8 && position[3] <= 8 && length[2] <= 9 && length[3] <= 9) {
                            //std::cout<<j<<":"<<new_length_v<<" "<<new_length_u<<" "<<res.cost<<std::endl;
                        }
                        j++;
                        //std::cout<<"Beginning: "<< rate0<<" "<<distortion0<<" "<<J0<<" "<<rate0*lambda + distortion0<<std::endl;

                        //if(relevant) std::cout<<"counter = "<<counter<<" p:"<<new_position_t<<" "<<new_position_s<<" "<<new_position_v<<" "<<new_position_u<<" Cumm = "<<J0<<std::endl;
                        //codeString_0 += codeString_1;
                            
                        SignalEnergySum += Energy;
                    }
                    
                }
                
            }
            
        }
        
        //std::cout<<"CodeString_0 = "<<codeString_0<<std::endl;
    }    
    codeString_0 = ss_for_J0.str();

    //evaluate the cost J1 to skip this subblock
    J1 += SignalEnergySum;
    distortion1 += SignalEnergySum;
    //Choose the lowest cost

    HexResult finalResult;
    std::stringstream ss_final;
    double safeJ0 = J0;

    if((J0 < J1)||((bitplane == mInferiorBitPlane)&&(Significance == 0))) {
        //std::cout<<"j0 < j1"<<std::endl;
        //std::cout<<"J0 = "<<J0<<" Calc J0 = "<< lambda*rate0 + distortion0<<std::endl;

        rate = rate0;
        distortion = distortion0;


        if(Significance == 1) {
            ss_final<<'1';
        }
        else {
            ss_final<<'0';
        }
        ss_final<<codeString_0;
    }
    else {
        //std::cout<<"j0 > j1"<<std::endl;
        // if (position[2] <= 8 && position[3] <= 8 && length[2] <= 9 && length[3] <= 9) {
        //     std::cout<<"Comivos todos"<<std::endl;
        //     std::cout<<"Position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
        //     std::cout<<"Length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<< " Bitplane: "<<bitplane<<std::endl;
        //     std::cout<<safeJ0<<" "<<J1<<" "<<finalResult.codeStream[0]<<std::endl;
        // }

        rate += rate1;   
        distortion = distortion1;   
        ss_final<< "2";  
 
        J0 = J1;
        
        for(int model_index = 0; model_index < NUMBER_OF_MODELS; model_index++)
            mOptimizationPmodel[model_index].CopyModel(&currentProbabilityModel[model_index]);
        
        if(bitplane > BITPLANE_BYPASS_FLAGS) 
            mOptimizationPmodel[2*bitplane+mSegmentationFlagProbabilityModelIndex].UpdateModel(1);
    }
    finalResult.cost = J0;
    finalResult.codeStream = ss_final.str();
    // if(position[2] == 0 && position[3] == 0){
    //     std::cout<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<std::endl;
    //     std::cout<<"CodeString_0 = "<<codeString_0<<std::endl;
    //     std::cout<<"finalResult.codeStream = "<<finalResult.codeStream<<std::endl<<std::endl;
    // }
    // if (position[2] <= 8 && position[3] <= 8 && length[2] <= 9 && length[3] <= 9){
    //     std::cout<<"Position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<std::endl;
    //     std::cout<<"Length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<< " Bitplane: "<<bitplane<<std::endl;
    //     std::cout<<safeJ0<<" "<<J1<<" "<<finalResult.codeStream[0]<<std::endl;
    //     std::cout<<mOptimizationPmodel<<std::endl;
    // }
    
    //std::cout<<"rate = "<<rate<<std::endl;
    //std::cout<<"J0 = "<<J0/lambda<<std::endl;
    
    signalEnergy = SignalEnergySum;
    //std::cout<<"J0 = "<<J0<<" Calc J0 = "<< lambda*rate + distortion<<" rate = "<<rate<<" distortion = "<<distortion<<" lambda = "<<lambda<<std::endl;

        
    return(finalResult);    
}

void Hierarchical4DEncoder::encodeSubblockFromPool(std::array<int64_t,4> length, std::array<int64_t,4> position, int bitplane, double lambda){
    int flagSearchIndex = 0;
    //std::cout<<"Encoding Subblock from Pool with length: "<<length[0]<<" "<<length[1]<<" "<<length[2]<<" "<<length[3]<<" and position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" "<<position[3]<<" and bitplane: "<<bitplane<<std::endl;
    this->mSubbandLF_.data = this->mSubbandLF_.data.contiguous();
    this->currCost = build_optimal_tree_from_pool(length,position, bitplane, lambda);
    iterateEncoding(0, length, position, bitplane);
    std::cout<<"Next Available Index: "<<mProcessingContext.next_available_idx<<std::endl;
    std::cout<<std::endl;

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
void Hierarchical4DEncoder :: RdEncodeHexadecatree_(std::array<int64_t,4> position, std::array<int64_t,4> length, int bitplane, int &flagIndex) {
    if(bitplane < mInferiorBitPlane) {
        return;
    }
    //If the block is a single bit long Encode the bit and return
    int* data = mSubbandLF_.data.data_ptr<int>();
    if(length[0]*length[1]*length[2]*length[3] == 1) {
        //rd encode coefficient     
        //std::cout<<"Encoded Coefficient"<<std::endl;   
        EncodeCoefficient(data[mSubbandLF_.LinearPosition(position[0],position[1],position[2],position[3])], bitplane);
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
        mIgnored += length[0] * length[1] * length[2] * length[3];
        mIgnoreEfficiency = (double)mIgnored/(double)flagTwo;
        //ignored.index({at::indexing::Slice(position[0],position[0]+length[0]),at::indexing::Slice(position[1],position[1]+length[1]),at::indexing::Slice(position[2],position[2]+length[2]),at::indexing::Slice(position[3],position[3]+length[3])}) = flagIndex*at::ones(length,at::kInt);
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
    std::cout<<"0: "<<flagZero<<" 1: "<<flagOne<<" 2: "<<flagTwo<<" Ignore Efficiency = "<<mIgnoreEfficiency<<" "<<mIgnored<<std::endl;
    mEntropyCoder.Flush();      //flushes entropy encoder
    
}





int Hierarchical4DEncoder :: OptimumBitplaneFaster_(double lambda) {
    //std::cout<<"hello"<<std::endl;
    //std::cout<<"Optimum Calc Lambda: "<<lambda<<std::endl;
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

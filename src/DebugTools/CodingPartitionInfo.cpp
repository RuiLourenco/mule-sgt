#include "DebugTools/CodingPartitionInfo.h"
#include "DebugTools/CodingUnitInfo.h"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;


// Constructor
CodingPartitionInfo::CodingPartitionInfo(std::array<int64_t, 4> lightFieldPosition, std::array<int64_t, 4> size, const std::vector<CodingUnitInfo>& codingUnitInfos)
    : lightFieldPosition(lightFieldPosition), size(size), codingUnitInfos(codingUnitInfos) {}

// Constructor with empty codingUnitInfos
CodingPartitionInfo::CodingPartitionInfo(std::array<int64_t, 4> lightFieldPosition, std::array<int64_t, 4> size)
    : lightFieldPosition(lightFieldPosition), size(size), codingUnitInfos() {}

// Method to append a CodingUnitInfo to the vector
void CodingPartitionInfo::appendCodingUnitInfo(const CodingUnitInfo& codingUnitInfo) {
    codingUnitInfos.push_back(codingUnitInfo);
}

// Getter methods
std::array<int64_t, 4> CodingPartitionInfo::getLightFieldPosition() const {
    return lightFieldPosition;
}

std::array<int64_t, 4> CodingPartitionInfo::getSize() const {
    return size;
}

const std::vector<CodingUnitInfo>& CodingPartitionInfo::getCodingUnitInfos() const {
    return codingUnitInfos;
}

json CodingPartitionInfo::toJson() const {
    json j = json::object_t();
    j["lightFieldPosition"] = {lightFieldPosition[0], lightFieldPosition[1], lightFieldPosition[2], lightFieldPosition[3]};
    j["size"] = {size[0], size[1], size[2], size[3]};
    j["codingUnitInfos"] = json::array();
    for (const auto& info : codingUnitInfos) {
        j["codingUnitInfos"].push_back(info.toJson());
    }
    return j;
}

void CodingPartitionInfo::printToJsonFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios_base::app);
    if (file.is_open()) {
        file << toJson().dump(); // Compact representation
        file.close();
        std::cout << "File written to: " << filename << std::endl;
    } else {
        throw std::runtime_error("Unable to open file");
    }
}

at::Tensor CodingPartitionInfo::getTensorFromInfo(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize, std::function<double(const CodingUnitInfo&)> getData){
    at::Tensor tensor = torch::zeros({totalSize[0],totalSize[1]},torch::kDouble);
    std::vector<std::array<int64_t,2>> visitedPositions;
    for (auto & partitionInfo : partitionInfos){
        bool skip = false;
        for(auto& visitedPosition : visitedPositions){
            if(partitionInfo.lightFieldPosition[2] == visitedPosition[0] && partitionInfo.lightFieldPosition[3] == visitedPosition[1]){
                skip = true;
                break;
            }
        }
        if (skip)
        {
            continue;
        }
        for(auto & cuInfo : partitionInfo.getCodingUnitInfos()){
            std::array<int64_t,4> position = cuInfo.getLightFieldPosition();
            std::array<int64_t,4> size = cuInfo.getSize();
            double data = getData(cuInfo);
            for (int i = 0; i < size[2]; i++){
                
                for (int j = 0; j < size[3]; j++){
                    tensor.index_put_({position[2]+i,position[3]+j},data);
                }
            }
        }
        visitedPositions.push_back({partitionInfo.lightFieldPosition[2],partitionInfo.lightFieldPosition[3]});
    }
    return tensor;
}

at::Tensor CodingPartitionInfo::getStructureTensorHorizontal(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getStructureTensorHorizontalAngle);
}
at::Tensor CodingPartitionInfo::getStructureTensorVertical(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getStructureTensorVerticalAngle);
}
at::Tensor CodingPartitionInfo::getStructureTensorAverage(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getStructureTensorAverageAngle);
}
at::Tensor CodingPartitionInfo::getLogdetHorizontal(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getLogdetHorizontalAngle);
}
at::Tensor CodingPartitionInfo::getLogdetVertical(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getLogdetVerticalAngle);
}
at::Tensor CodingPartitionInfo::getLogdetAverage(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getLogdetAverageAngle);
}
at::Tensor CodingPartitionInfo::getRate(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getRate);
}
at::Tensor CodingPartitionInfo::getPSNR(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getPSNR);
}
at::Tensor CodingPartitionInfo::getAngleHeuristicUsed(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getAngleHeuristicUsed);
}
at::Tensor CodingPartitionInfo::getBestStructureTensorAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestStructureTensorAngle);
}
at::Tensor CodingPartitionInfo::getBestLogdetAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestLogdetAngle);
}
at::Tensor CodingPartitionInfo::getBestGridSearchAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestGridSearchAngle);
}
at::Tensor CodingPartitionInfo::getBestGridSearchCost(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestGridSearchCost);
}
at::Tensor CodingPartitionInfo::getBestStructureTensorCost(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestStructureTensorCost);
}
at::Tensor CodingPartitionInfo::getBestLogdetCost(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestLogdetCost);
}
at::Tensor CodingPartitionInfo::getChosenAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getChosenAngle);
}

at::Tensor CodingPartitionInfo::getStructureTensorError(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    at::Tensor error = (getBestStructureTensorAngle(partitionInfos,totalSize) - getChosenAngle(partitionInfos,totalSize)).abs();
    return error;
}
at::Tensor CodingPartitionInfo::getLogdetError(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    at::Tensor error = (getBestLogdetAngle(partitionInfos,totalSize) - getChosenAngle(partitionInfos,totalSize)).abs();
    return error;
}

at::Tensor CodingPartitionInfo::getStructureTensorConfidence(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    at::Tensor confidence = (getStructureTensorHorizontal(partitionInfos,totalSize) - getStructureTensorVertical(partitionInfos,totalSize)).abs();
    return confidence;
}

at::Tensor CodingPartitionInfo::getStructureTensorCostDiff(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    at::Tensor error = (getBestStructureTensorCost(partitionInfos,totalSize) - getBestGridSearchCost(partitionInfos,totalSize))/getBestGridSearchCost(partitionInfos,totalSize);
    return error;
}
at::Tensor CodingPartitionInfo::getLogdetCostDiff(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    at::Tensor error = (getBestLogdetCost(partitionInfos,totalSize) - getBestGridSearchCost(partitionInfos,totalSize))/getBestGridSearchCost(partitionInfos,totalSize);
    return error;
}

void CodingPartitionInfo::printVectorToJsonFile(const std::vector<CodingPartitionInfo>& partitionInfos, const std::string& filename) {
    json j = json::array();
    std::cout<<"Writing json File to: "<<filename<<std::endl;
    for (const auto& partitionInfo : partitionInfos) {
        j.push_back(partitionInfo.toJson());
    }
    std::ofstream file(filename,std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << j.dump(); // Compact representation
        file.close();
        std::cout << "File written to: " << filename << std::endl;
    } else {
        throw std::runtime_error("Unable to open file");
    }
}

std::vector<CodingPartitionInfo> CodingPartitionInfo::fromJsonFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file");
    }

    json j;
    file >> j;
    file.close();
    //std::cout<<"File Read"<<std::endl;
    std::vector<CodingPartitionInfo> partitionInfos;
    int i = 0;
    for (const auto& item : j) {
        //std::cout<<"Reading Position"<<std::endl;
        std::array<int64_t, 4> lightFieldPosition = {
            item["lightFieldPosition"][0].get<int64_t>(),
            item["lightFieldPosition"][1].get<int64_t>(),
            item["lightFieldPosition"][2].get<int64_t>(),
            item["lightFieldPosition"][3].get<int64_t>()
        };
        //std::cout<<"position read"<<std::endl;
        std::array<int64_t, 4> size = {
            item["size"][0].get<int64_t>(),
            item["size"][1].get<int64_t>(),
            item["size"][2].get<int64_t>(),
            item["size"][3].get<int64_t>()
        };
        //std::cout<<"size read"<<std::endl;
        std::vector<CodingUnitInfo> codingUnitInfos;
        for (const auto& cuItem : item["codingUnitInfos"]) {
            codingUnitInfos.push_back(CodingUnitInfo::fromJson(cuItem));
        }
        //std::cout<<"Coding Unit Read "<<i++<<std::endl;

        partitionInfos.emplace_back(lightFieldPosition, size, codingUnitInfos);
    }


    return partitionInfos;
}

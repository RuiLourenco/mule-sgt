#ifndef CODING_PARTITION_INFO_H
#define CODING_PARTITION_INFO_H

#include <vector>
#include <array>
#include <string>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <functional>


class CodingUnitInfo; // Forward declaration

class CodingPartitionInfo {
public:
    CodingPartitionInfo() = default;
    // Constructor
    CodingPartitionInfo(std::array<int64_t,4> lightFieldPosition, std::array<int64_t,4> size, const std::vector<CodingUnitInfo>& codingUnitInfos);
    // Constructor with empty codingUnitInfos
    CodingPartitionInfo(std::array<int64_t,4> lightFieldPosition, std::array<int64_t,4> size);
    void incrementTotalSize(double rate);
    void incrementTotalDistortion(double psnr);
    double getTotalSize() const;
    double getTotalDistortion() const;
    static int countUnitsWithin10Degrees(int delta, const std::vector<CodingPartitionInfo>& partitionInfos, const std::string& outputDirectory);


    // Method to convert the object to JSON
    nlohmann::json toJson() const;


    // Method to print the object to a JSON file
    void printToJsonFile(const std::string& filename) const;
    // New static method to print a vector of CodingPartitionInfo to a JSON file
    static void printVectorToJsonFile(const std::vector<CodingPartitionInfo>& partitionInfos, const std::string& filename);

    // New static method to parse a JSON file and return a vector of CodingPartitionInfo
    static std::vector<CodingPartitionInfo> fromJsonFile(const std::string& filename);

    // New static method to get an image of angles chosen.
    static at::Tensor getStructureTensorHorizontal(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getStructureTensorVertical(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getStructureTensorAverage(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getCovarianceHorizontal(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getCovarianceVertical(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getCovarianceAverage(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getLogdetHorizontal(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getLogdetVertical(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getLogdetAverage(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getRate(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getPSNR(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getMSE(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    double getTotalRate() const;

    static at::Tensor getAngleHeuristicUsed(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getChosenAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getChosenRhoS(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getChosenRhoU(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestStructureTensorAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestStructureTensorCost(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestCovarianceAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestCovarianceCost(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestGridSearchAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestLogdetAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestLogdetCost(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getBestGridSearchCost (const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getStructureTensorError(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getStructureTensorCostDiff(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getLogdetError(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getLogdetCostDiff(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    static at::Tensor getTensorFromInfo(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize, std::function<double(const CodingUnitInfo&)> getData);
    static at::Tensor getStructureTensorConfidence(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize);
    // Method to append a CodingUnitInfo to the vector
    void appendCodingUnitInfo(const CodingUnitInfo& codingUnitInfo);
    // Getter methods
    std::array<int64_t,4> getLightFieldPosition() const;
    std::array<int64_t,4> getSize() const;
    const std::vector<CodingUnitInfo>& getCodingUnitInfos() const;
    static void findLargestAngleDifferenceAndGeneratePlot(
        const std::vector<CodingPartitionInfo>& partitionInfos,
        const std::string& outputDirectory);
    static void generatePlotsForAngleDifferencesBelowThreshold(
        const std::vector<CodingPartitionInfo>& partitionInfos,
        const std::string& outputDirectory,
        double threshold = 10.0);
    
    static CodingPartitionInfo findPartitionInfoByPosition(
        const std::vector<CodingPartitionInfo>& partitionInfos,
        const std::array<int64_t, 4>& lightFieldPosition);

    void generatePythonScriptsForPartition(const std::string& outputDirectory) const;

private:
    std::array<int64_t,4> lightFieldPosition;
    std::array<int64_t,4> size;
    double totalDistortion = 0.0; // Total error for the partition
    double totalBitsize = 0.0; // Total size for the partition in bits
    std::vector<CodingUnitInfo> codingUnitInfos;
};

#endif // CODING_PARTITION_INFO_H
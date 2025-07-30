#include "DebugTools/CodingPartitionInfo.h"
#include "DebugTools/CodingUnitInfo.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept> // For std::runtime_error
#include <cstdlib>


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

int CodingPartitionInfo::countUnitsWithin10Degrees(int delta, const std::vector<CodingPartitionInfo>& partitionInfos, const std::string& outputDirectory) {
    int totalChecked = 0;
    int within10Count = 0;
    std::vector<double> relDiffsAll;
    std::vector<double> relDiffsFailing;

    // Map for counting total and failing units per size
    std::map<int, int> totalUnitsPerSize;
    std::map<int, int> failingUnitsPerSize;
    for (int sz : {4, 8, 16, 32, 64, 128}) {
        totalUnitsPerSize[sz] = 0;
        failingUnitsPerSize[sz] = 0;
    }

    // Ensure the output directory exists
    std::filesystem::create_directories(outputDirectory);

    for (size_t partitionIdx = 0; partitionIdx < partitionInfos.size(); ++partitionIdx) {
        const auto& partitionInfo = partitionInfos[partitionIdx];
        const auto& cuInfos = partitionInfo.getCodingUnitInfos();
        for (size_t cuIdx = 0; cuIdx < cuInfos.size(); ++cuIdx) {
            const auto& cuInfo = cuInfos[cuIdx];
            int size2 = cuInfo.getSize()[2];
            if (totalUnitsPerSize.find(size2) != totalUnitsPerSize.end()) {
                totalUnitsPerSize[size2]++;
            }
            try {
                auto [angle10, minAngle, isWithin10, minCostWithin10, relCostDiff] = cuInfo.analyzeGridSearchAngle(delta);
                totalChecked++;
                relDiffsAll.push_back(relCostDiff);
                if (isWithin10) {
                    within10Count++;
                } else {
                    relDiffsFailing.push_back(relCostDiff);
                    if (failingUnitsPerSize.find(size2) != failingUnitsPerSize.end()) {
                        failingUnitsPerSize[size2]++;
                    }
                    // Only generate a plot file if size is 128
                    if (size2 == 128) {
                        std::string filename = outputDirectory + "/partition_" + std::to_string(partitionIdx) +
                                              "_cu_" + std::to_string(cuIdx) + "_grid_search.py";
                        cuInfo.generatePythonScriptForGridSearchAngle(filename);
                    }
                }
            } catch (const std::exception& e) {
                continue;
            }
        }
    }
    std::cout << "Checked " << totalChecked << " CodingUnitInfos, " << within10Count << " were within 10 degrees." << std::endl;
    std::cout << (totalChecked - within10Count) << " Python scripts generated for failing units in: " << outputDirectory << std::endl;

    auto printStats = [](const std::vector<double>& diffs, const std::string& label) {
        if (diffs.empty()) {
            std::cout << "No " << label << " relative cost differences found." << std::endl;
            return;
        }
        std::vector<double> sortedDiffs = diffs;
        std::sort(sortedDiffs.begin(), sortedDiffs.end());
        double avg = std::accumulate(sortedDiffs.begin(), sortedDiffs.end(), 0.0) / sortedDiffs.size();
        double min = sortedDiffs.front();
        double max = sortedDiffs.back();
        double median = sortedDiffs.size() % 2 == 0 ?
            (sortedDiffs[sortedDiffs.size()/2 - 1] + sortedDiffs[sortedDiffs.size()/2]) / 2.0 :
            sortedDiffs[sortedDiffs.size()/2];
        std::cout << label << " (" << sortedDiffs.size() << "):" << std::endl;
        std::cout << "  Average relative cost difference: " << avg << std::endl;
        std::cout << "  Min relative cost difference: " << min << std::endl;
        std::cout << "  Max relative cost difference: " << max << std::endl;
        std::cout << "  Median relative cost difference: " << median << std::endl;
    };

    printStats(relDiffsAll, "All cases");
    printStats(relDiffsFailing, "Failing cases (not within 10 degrees)");

    // Print percentage of failing units per size
    std::cout << "Percentage of failing CodingUnits per size (not within 10 degrees):" << std::endl;
    for (int sz : {4, 8, 16, 32, 64, 128}) {
        int total = totalUnitsPerSize[sz];
        int fail = failingUnitsPerSize[sz];
        double percent = (total > 0) ? (100.0 * fail / total) : 0.0;
        std::cout << "  Size " << sz << ": " << percent << "% (" << fail << "/" << total << ")" << std::endl;
    }

    return within10Count;
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

void CodingPartitionInfo::incrementTotalSize(double bitSize) {
    totalBitsize += bitSize;
}
void CodingPartitionInfo::incrementTotalDistortion(double distortion) {
    totalDistortion += distortion;
}
double CodingPartitionInfo::getTotalSize() const {
    return totalBitsize;
}
double CodingPartitionInfo::getTotalDistortion() const {
    return totalDistortion;
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
    std::vector<double> visitedRates;
    std::cout<<"Total number of partitions: "<<partitionInfos.size()<<std::endl;
    for (auto & partitionInfo : partitionInfos){
        bool skip = false;
        for(int i = 0; i < visitedPositions.size(); i++){
            std::array<int64_t,2> visitedPosition = visitedPositions[i];
            double visitedRate = visitedRates[i];
            // Check if the current partition position matches any visited position
            if(partitionInfo.lightFieldPosition[2] == 0 && partitionInfo.lightFieldPosition[3] == 0)
                std::cout<<"This is a partition at (0,0)"<<std::endl;
            if(partitionInfo.lightFieldPosition[2] == visitedPosition[0] && partitionInfo.lightFieldPosition[3] == visitedPosition[1]){
                //Check if the rate is larger than the visited rate
                
                //if(partitionInfo.getTotalRate() < visitedRate){
                    // If the rate is smaller, skip this partition
                    if(partitionInfo.lightFieldPosition[2] == 0 && partitionInfo.lightFieldPosition[3] == 0){
                        std::cout<<"Skipping partition at position: "<<partitionInfo.lightFieldPosition[2]<<","<<partitionInfo.lightFieldPosition[3]<<" with rate: "<<partitionInfo.getTotalRate()<<std::endl;
                        std::cout<<visitedRate<<" > "<<partitionInfo.getTotalRate()<<std::endl;
                    }
                    skip = true;
                //}
        
                break;
            }
        }
        if (skip)
        {
            continue;
        }
        for(auto & cuInfo : partitionInfo.getCodingUnitInfos()){
            
            if(partitionInfo.lightFieldPosition[2] == 0 && partitionInfo.lightFieldPosition[3] == 0)
                std::cout<<"Chosen Angle:"<<cuInfo.getBestGridSearchAngle()<<" "<<skip<<std::endl;

        
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
        visitedRates.push_back(partitionInfo.getTotalRate());
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
at::Tensor CodingPartitionInfo::getCovarianceHorizontal(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getCovarianceHorizontalAngle);
}
at::Tensor CodingPartitionInfo::getCovarianceVertical(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getCovarianceVerticalAngle);
}
at::Tensor CodingPartitionInfo::getCovarianceAverage(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getCovarianceAverageAngle);
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
double CodingPartitionInfo::getTotalRate() const {
    double totalRate = 0.0;
    for (const auto& cuInfo : codingUnitInfos) {
        totalRate += cuInfo.getRate() * (cuInfo.getSize()[2] * cuInfo.getSize()[3])/(size[2] * size[3]);
    }
    totalRate = totalRate / codingUnitInfos.size();
    return totalRate;
}
at::Tensor CodingPartitionInfo::getRate(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getRate);
}
at::Tensor CodingPartitionInfo::getPSNR(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getPSNR);
}
at::Tensor CodingPartitionInfo::getMSE(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getMSE);
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
at::Tensor CodingPartitionInfo::getBestCovarianceAngle(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestCovarianceAngle);
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
at::Tensor CodingPartitionInfo::getBestCovarianceCost(const std::vector<CodingPartitionInfo>& partitionInfos,std::array<int64_t,2> totalSize){
    return getTensorFromInfo(partitionInfos,totalSize,&CodingUnitInfo::getBestCovarianceCost);
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

void CodingPartitionInfo::generatePythonScriptsForPartition(const std::string& outputDirectory) const{
    // Ensure the output directory ends with a slash
    std::string directory = outputDirectory;
    if (!directory.empty() && directory.back() != '/') {
        directory += '/';
    }

    // Iterate through each CodingUnitInfo in the partition
    const auto& codingUnitInfos = this->getCodingUnitInfos();
    for (size_t i = 0; i < codingUnitInfos.size(); ++i) {
        // Generate a unique filename for each CodingUnitInfo
        std::string filename = directory + "coding_unit_" + std::to_string(i) + "_grid_search.py";

        // Call the generatePythonScriptForGridSearchAngle method
        codingUnitInfos[i].generatePythonScriptForGridSearchAngle(filename);
    }
}

CodingPartitionInfo CodingPartitionInfo::findPartitionInfoByPosition(
    const std::vector<CodingPartitionInfo>& partitionInfos,
    const std::array<int64_t, 4>& lightFieldPosition) 
{
    for (const auto& partitionInfo : partitionInfos) {
        if (partitionInfo.getLightFieldPosition() == lightFieldPosition) {
            return partitionInfo;
        }
    }

    // If no match is found, throw an exception
    throw std::runtime_error("No CodingPartitionInfo found for the given lightFieldPosition.");
}

void CodingPartitionInfo::findLargestAngleDifferenceAndGeneratePlot(
    const std::vector<CodingPartitionInfo>& partitionInfos,
    const std::string& outputDirectory) 
{
    int maxPartitionIndex = -1;
    int maxUnitIndex = -1;
    double maxDifference = -1.0;
    double worstStructureTensorAngle = 0.0;
    double  worstGridSearchAngle = 0.0;

    // Find the CodingUnitInfo with the largest angle difference
    for (size_t partitionIndex = 0; partitionIndex < partitionInfos.size(); ++partitionIndex) {
        const auto& partitionInfo = partitionInfos[partitionIndex];
        const auto& codingUnitInfos = partitionInfo.getCodingUnitInfos();

        for (size_t unitIndex = 0; unitIndex < codingUnitInfos.size(); ++unitIndex) {
            const auto& cuInfo = codingUnitInfos[unitIndex];
            double chosenAngle = cuInfo.getChosenAngle();
            double bestStructureTensorAngle = cuInfo.getBestStructureTensorAngle();
            std::array<int64_t,4> size = cuInfo.getSize();
            if(size[2] == 4 || size[3] == 4){
                continue;
            }
            if(bestStructureTensorAngle <= -80 || bestStructureTensorAngle >= 80){
                continue;
            }

            double angleDifference = std::abs(chosenAngle - bestStructureTensorAngle);

            if (angleDifference > maxDifference) {
                worstGridSearchAngle = chosenAngle;
                worstStructureTensorAngle = bestStructureTensorAngle;
                maxDifference = angleDifference;
                maxPartitionIndex = partitionIndex;
                maxUnitIndex = unitIndex;
            }
        }
    }
    std::cout<<"Max Difference: "<<maxDifference<<" "<<worstStructureTensorAngle<<" "<<worstGridSearchAngle<<std::endl;
    double checkSTAngle = partitionInfos[maxPartitionIndex].getCodingUnitInfos()[maxUnitIndex].getBestStructureTensorAngle();
    double checkGSAngle = partitionInfos[maxPartitionIndex].getCodingUnitInfos()[maxUnitIndex].getBestGridSearchAngle();
    std::cout<<"Check Difference: "<<std::abs(checkSTAngle - checkGSAngle)<<" "<<checkSTAngle<<" "<<checkGSAngle<<std::endl;
    if (maxPartitionIndex == -1 || maxUnitIndex == -1) {
        throw std::runtime_error("No CodingUnitInfo found with a valid angle difference.");
    }

    // Generate Python script for the CodingUnitInfo with the largest angle difference
    const auto& targetPartition = partitionInfos[maxPartitionIndex];
    const auto& targetUnit = targetPartition.getCodingUnitInfos()[maxUnitIndex];

    std::string scriptFilename = outputDirectory + "/largest_angle_difference_plot.py";
    std::string figureFilename = outputDirectory + "/largest_angle_difference_plot.png";

    std::ofstream file(scriptFilename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + scriptFilename);
    }

    file << "import matplotlib.pyplot as plt\n";
    file << "angles = [";
    for (const auto& [angle, cost] : targetUnit.getGridSearchAngle()) {
        file << angle << ", ";
    }
    file << "]\n";

    file << "costs = [";
    for (const auto& [angle, cost] : targetUnit.getGridSearchAngle()) {
        file << cost << ", ";
    }
    file << "]\n";

    file << "sorted_data = sorted(zip(angles, costs))\n";
    file << "sorted_angles, sorted_costs = zip(*sorted_data)\n";

    file << "plt.figure(figsize=(10, 6))\n";
    file << "plt.plot(sorted_angles, sorted_costs, marker='o', label='Grid Search Costs')\n";
    file << "plt.axvline(x=" << targetUnit.getBestStructureTensorAngle()
         << ", color='r', linestyle='--', label='Best Structure Tensor Angle')\n";
         std::cout<<"Best Structure Tensor Angle: "<<targetUnit.getBestStructureTensorAngle()<<std::endl;
    file << "plt.axvline(x=" << targetUnit.getBestLogdetAngle()
         << ", color='g', linestyle='--', label='Best Logdet Angle')\n";
    file << "plt.title('Grid Search Angle vs Cost (Partition: " << maxPartitionIndex
         << ", Unit: " << maxUnitIndex << ")')\n";
    file << "plt.xlabel('Angle')\n";
    file << "plt.ylabel('Cost')\n";
    file << "plt.legend()\n";
    file << "plt.grid(True)\n";
    file << "plt.savefig('" << figureFilename << "')\n";
    file << "plt.show()\n";

    file.close();

    // Execute the Python script
    std::string command = "python3 " + scriptFilename;
    int result = system(command.c_str());
    if (result != 0) {
        throw std::runtime_error("Failed to execute Python script.");
    }

    // Delete the Python script
    if (std::remove(scriptFilename.c_str()) != 0) {
        throw std::runtime_error("Failed to delete Python script: " + scriptFilename);
    }

    std::cout << "Python script executed and deleted successfully." << std::endl;
    std::cout << "Figure saved to: " << figureFilename << std::endl;
    std::cout << "Partition Index: " << maxPartitionIndex << ", Unit Index: " << maxUnitIndex
              << ", Largest Angle Difference: " << maxDifference << " degrees." << std::endl;
}

void CodingPartitionInfo::generatePlotsForAngleDifferencesBelowThreshold(
    const std::vector<CodingPartitionInfo>& partitionInfos,
    const std::string& outputDirectory,
    double threshold) 
{
    size_t plotCount = 0;
    std::map<int, int> codingUnitsPerSize; // Map to count coding units per size[2]
    std::map<int, int> totalCodingUnitsPerSize; // Map to count coding units per size[2]

    // Initialize counts for sizes 4, 8, 16, 32, and 64
    for (int size : {4, 8, 16, 32, 64}) {
        codingUnitsPerSize[size] = 0;
        totalCodingUnitsPerSize[size] = 0;
    }

    // Iterate through all CodingPartitionInfo and CodingUnitInfo
    for (size_t partitionIndex = 0; partitionIndex < partitionInfos.size(); ++partitionIndex) {
        const auto& partitionInfo = partitionInfos[partitionIndex];
        const auto& codingUnitInfos = partitionInfo.getCodingUnitInfos();

        for (size_t unitIndex = 0; unitIndex < codingUnitInfos.size(); ++unitIndex) {
            const auto& cuInfo = codingUnitInfos[unitIndex];
            double chosenAngle = cuInfo.getChosenAngle();
            double bestStructureTensorAngle = cuInfo.getBestStructureTensorAngle();
            double angleDifference = std::abs(chosenAngle - bestStructureTensorAngle);
            int size2 = cuInfo.getSize()[2];
            if (totalCodingUnitsPerSize.find(size2) != totalCodingUnitsPerSize.end()) {
                totalCodingUnitsPerSize[size2]++;
            }

            // Check if the angle difference is below the threshold
            if (angleDifference < threshold) {
                continue;
            }

            // Collect size[2] statistics
            if (codingUnitsPerSize.find(size2) != codingUnitsPerSize.end()) {
                codingUnitsPerSize[size2]++;
            }

            // Calculate relative cost difference
            double stCost = cuInfo.getBestStructureTensorCost();
            double gsCost = cuInfo.getBestGridSearchCost();
            double relativeCostDifference = (stCost - gsCost) / gsCost;

            // Generate Python script for the current CodingUnitInfo
            std::string scriptFilename = outputDirectory + "/plot_partition_" + std::to_string(partitionIndex) +
                                         "_unit_" + std::to_string(unitIndex) + ".py";
            std::string figureFilename = outputDirectory + "/plot_partition_" + std::to_string(partitionIndex) +
                                         "_unit_" + std::to_string(unitIndex) + ".png";

            std::ofstream file(scriptFilename);
            if (!file.is_open()) {
                throw std::runtime_error("Unable to open file: " + scriptFilename);
            }

            file << "import matplotlib.pyplot as plt\n";
            file << "angles = [";
            for (const auto& [angle, cost] : cuInfo.getGridSearchAngle()) {
                file << angle << ", ";
            }
            file << "]\n";

            file << "costs = [";
            for (const auto& [angle, cost] : cuInfo.getGridSearchAngle()) {
                file << cost << ", ";
            }
            file << "]\n";

            file << "sorted_data = sorted(zip(angles, costs))\n";
            file << "sorted_angles, sorted_costs = zip(*sorted_data)\n";

            file << "plt.figure(figsize=(10, 6))\n";
            file << "plt.plot(sorted_angles, sorted_costs, marker='o', label='Grid Search Costs')\n";
            file << "plt.axvline(x=" << cuInfo.getBestStructureTensorAngle()
                 << ", color='r', linestyle='--', label='Best Structure Tensor Angle')\n";
            file << "plt.axvline(x=" << cuInfo.getBestLogdetAngle()
                 << ", color='g', linestyle='--', label='Best Logdet Angle')\n";
            file << "plt.title('Grid Search Angle vs Cost (Partition: " << partitionIndex
                 << ", Unit: " << unitIndex << ")\\nRelative Cost Difference: "
                 << relativeCostDifference * 100 << "%')\n";
            file << "plt.xlabel('Angle')\n";
            file << "plt.ylabel('Cost')\n";
            file << "plt.legend()\n";
            file << "plt.grid(True)\n";
            file << "plt.savefig('" << figureFilename << "')\n";
            file << "plt.show()\n";

            file.close();

            // Execute the Python script
            std::string command = "python3 " + scriptFilename;
            int result = system(command.c_str());
            if (result != 0) {
                throw std::runtime_error("Failed to execute Python script: " + scriptFilename);
            }

            // Delete the Python script
            if (std::remove(scriptFilename.c_str()) != 0) {
                throw std::runtime_error("Failed to delete Python script: " + scriptFilename);
            }

            std::cout << "Python script executed and deleted successfully for Partition: " << partitionIndex
                      << ", Unit: " << unitIndex << ", Angle Difference: " << angleDifference << " degrees."
                      << std::endl;

            ++plotCount;
        }
    }

    // Print statistics
    std::cout << "Number of Coding Units per size[2]:" << std::endl;
    for (const auto& [size, count] : codingUnitsPerSize) {
        std::cout << "  Size " << size << ": " << (double) count/(double)totalCodingUnitsPerSize[size] <<" "<<count<<" "<< totalCodingUnitsPerSize[size]<<std::endl;
    }

    if (plotCount == 0) {
        std::cout << "No CodingUnitInfo found with an angle difference below the threshold of " << threshold
                  << " degrees." << std::endl;
    } else {
        std::cout << "Generated and executed " << plotCount << " plots for angle differences below the threshold."
                  << std::endl;
    }
}



#include "DebugTools/CodingUnitInfo.h"
#include <nlohmann/json.hpp>

CodingUnitInfo::CodingUnitInfo(const std::array<int64_t, 4>& size, const std::array<int64_t, 4>& lightFieldPosition)
    : size(size), lightFieldPosition(lightFieldPosition) {}

void CodingUnitInfo::setGridSearchAngle(const std::map<double, double>& gridSearchAngle) {
    this->gridSearchAngle = gridSearchAngle;
}

void CodingUnitInfo::setStructureTensorHorizontal(const std::array<double,2>& structureTensorHorizontal) {
    this->structureTensorHorizontal = structureTensorHorizontal;
}
void CodingUnitInfo::setStructureTensorVertical(const std::array<double,2>& structureTensorVertical) {
    this->structureTensorVertical = structureTensorVertical;
}
void CodingUnitInfo::setStructureTensorAverage(const std::array<double,2>& structureTensorAverage) {
    this->structureTensorAverage = structureTensorAverage;
}

void CodingUnitInfo::setCovarianceHorizontal(const std::array<double,2>& covarianceHorizontal) {
    this->covarianceHorizontal = covarianceHorizontal;
}
void CodingUnitInfo::setCovarianceVertical(const std::array<double,2>& covarianceVertical) {
    this->covarianceVertical = covarianceVertical;
}
void CodingUnitInfo::setCovarianceAverage(const std::array<double,2>& covarianceAverage) {
    this->covarianceAverage = covarianceAverage;
}
void CodingUnitInfo::setLogdetHorizontal(const std::array<double,2>& logdetHorizontal) {
    this->logdetHorizontal = logdetHorizontal;
}
void CodingUnitInfo::setLogdetVertical(const std::array<double,2>& logdetVertical) {
    this->logdetVertical = logdetVertical;
}
void CodingUnitInfo::setLogdetAverage(const std::array<double,2>& logdetAverage) {
    this->logdetAverage = logdetAverage;
}
void CodingUnitInfo::setRate(double rate) {
    this->rate = rate;
}
void CodingUnitInfo::setPSNR(double PSNR) {
    this->PSNR = PSNR;
}
void CodingUnitInfo::setAngleHeuristicUsed(int angleHeuristicUsed) {
    this->angleHeuristicUsed = angleHeuristicUsed;
}

void CodingUnitInfo::setSgtSideInfo(const SgtSideInfo& sgtSideInfo) {
    this->sgtSideInfo = sgtSideInfo;
}

void CodingUnitInfo::generatePythonScriptForGridSearchAngle(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    file << "import matplotlib.pyplot as plt\n";
    file << "angles = [";
    for (const auto& [angle, cost] : gridSearchAngle) {
        file << angle << ", ";
    }
    file << "]\n";

    file << "costs = [";
    for (const auto& [angle, cost] : gridSearchAngle) {
        file << cost << ", ";
    }
    file << "]\n";

    file << "sorted_data = sorted(zip(angles, costs))\n";
    file << "sorted_angles, sorted_costs = zip(*sorted_data)\n";

    file << "plt.figure(figsize=(10, 6))\n";
    file << "plt.plot(sorted_angles, sorted_costs, marker='o')\n";
    file << "plt.axvline(x=" << getBestStructureTensorAngle() << ", color='r', linestyle='--', label='Best Structure Tensor Angle')\n";
    file << "plt.title('Grid Search Angle vs Cost (Light Field Position: [" 
         << lightFieldPosition[2] << ", " 
         << lightFieldPosition[3] << "], Size: [" 
         << size[2] << "x" 
         << size[3] << "])')\n";
    file << "plt.xlabel('Angle')\n";
    file << "plt.ylabel('Cost')\n";
    file << "plt.legend()\n";
    file << "plt.grid(True)\n";
    file << "plt.savefig('grid_search_angle_plot_" 
         << lightFieldPosition[2] << "x" 
         << lightFieldPosition[3] << ".png')\n";
    file << "plt.show()\n";

    file.close();
}


void CodingUnitInfo::addGridSearchAngle(double angle, double cost) {
    this->gridSearchAngle[angle] = cost;
}
std::array<int64_t, 4> CodingUnitInfo::getLightFieldPosition() const {
    return lightFieldPosition;
}
std::array<int64_t, 4> CodingUnitInfo::getSize() const {
    return size;
}
std::array<double,2> CodingUnitInfo::getStructureTensorHorizontal() const {
    return structureTensorHorizontal;
}
std::array<double,2> CodingUnitInfo::getStructureTensorVertical() const {
    return structureTensorVertical;
}
std::array<double,2> CodingUnitInfo::getStructureTensorAverage() const {
    return structureTensorAverage;
}
std::array<double,2> CodingUnitInfo::getCovarianceHorizontal() const {
    return covarianceHorizontal;
}
std::array<double,2> CodingUnitInfo::getCovarianceVertical() const {
    return covarianceVertical;
}
std::array<double,2> CodingUnitInfo::getCovarianceAverage() const {
    return covarianceAverage;
}
double CodingUnitInfo::getStructureTensorHorizontalAngle() const{
    return structureTensorHorizontal[0];
}
double CodingUnitInfo::getStructureTensorVerticalAngle() const{
    return structureTensorVertical[0];
}
double CodingUnitInfo::getStructureTensorAverageAngle() const{
    return structureTensorAverage[0];
}
double CodingUnitInfo::getLogdetHorizontalAngle() const{
    return logdetHorizontal[0];
}
double CodingUnitInfo::getLogdetVerticalAngle() const{
    return logdetVertical[0];
}
double CodingUnitInfo::getLogdetAverageAngle() const{
    return logdetAverage[0];
}
double CodingUnitInfo::getCovarianceHorizontalAngle() const{
    return covarianceHorizontal[0];
}
double CodingUnitInfo::getCovarianceVerticalAngle() const{
    return covarianceVertical[0];
}
double CodingUnitInfo::getCovarianceAverageAngle() const{
    return covarianceAverage[0];
}

double CodingUnitInfo::getBestStructureTensorAngle() const{
    double bestAngle = 0;
    double bestCost = 0;
    if(structureTensorHorizontal[1] < structureTensorVertical[1]){
        bestAngle = structureTensorHorizontal[0];
        bestCost = structureTensorHorizontal[1];

    }else{
        bestAngle = structureTensorVertical[0];
        bestCost = structureTensorVertical[1];
    }
    if(structureTensorAverage[1] < bestCost){
        bestAngle = structureTensorAverage[0];
        bestCost = structureTensorAverage[1];
    }
    return bestAngle;
}
double CodingUnitInfo::getBestStructureTensorCost() const {
    double bestCost = std::numeric_limits<double>::max();
    bestCost = std::min(bestCost, structureTensorHorizontal[1]);
    bestCost = std::min(bestCost, structureTensorVertical[1]);
    bestCost = std::min(bestCost, structureTensorAverage[1]);
    return bestCost/(size[0]*size[1]*size[2]*size[3]);
}
double CodingUnitInfo::getBestLogdetAngle() const{
    double bestAngle = 0;
    double bestCost = 0;
    if(logdetHorizontal[1] < logdetVertical[1]){
        bestAngle = logdetHorizontal[0];
        bestCost = logdetHorizontal[1];

    }else{
        bestAngle = logdetVertical[0];
        bestCost = logdetVertical[1];
    }
    if(logdetAverage[1] < bestCost){
        bestAngle = logdetAverage[0];
        bestCost = logdetAverage[1];
    }
    return bestAngle;
}
double CodingUnitInfo::getBestCovarianceAngle() const{
    double bestAngle = 0;
    double bestCost = 0;
    if(covarianceHorizontal[1] < covarianceVertical[1]){
        bestAngle = covarianceHorizontal[0];
        bestCost = covarianceHorizontal[1];

    }else{
        bestAngle = covarianceVertical[0];
        bestCost = covarianceVertical[1];
    }
    if(covarianceAverage[1] < bestCost){
        bestAngle = covarianceAverage[0];
        bestCost = covarianceAverage[1];
    }
    return bestAngle;
}
double CodingUnitInfo::getBestLogdetCost() const {
    double bestCost = std::numeric_limits<double>::max();
    bestCost = std::min(bestCost, logdetHorizontal[1]);
    bestCost = std::min(bestCost, logdetVertical[1]);
    bestCost = std::min(bestCost, logdetAverage[1]);
    return bestCost/(size[0]*size[1]*size[2]*size[3]);
}
double CodingUnitInfo::getBestCovarianceCost() const {
    double bestCost = std::numeric_limits<double>::max();
    bestCost = std::min(bestCost, covarianceHorizontal[1]);
    bestCost = std::min(bestCost, covarianceVertical[1]);
    bestCost = std::min(bestCost, covarianceAverage[1]);
    return bestCost/(size[0]*size[1]*size[2]*size[3]);
}

double CodingUnitInfo::getBestGridSearchAngle() const{
    double bestAngle = 0;
    double bestCost = std::numeric_limits<double>::max();
    for (const auto& [angle, cost] : gridSearchAngle) {
        if (cost < bestCost) {
            bestCost = cost;
            bestAngle = angle;
        }
    }
    return bestAngle;
}

double CodingUnitInfo::getBestGridSearchCost() const{
    double bestCost = std::numeric_limits<double>::max();
    for (const auto& [angle, cost] : gridSearchAngle) {
        if (cost < bestCost) {
            bestCost = cost;
        }
    }
    return bestCost/(size[0]*size[1]*size[2]*size[3]);
}

double CodingUnitInfo::getChosenAngle() const{
    return this->sgtSideInfo.getAngleH();
}

std::array<double,2> CodingUnitInfo::getLogdetHorizontal() const {
    return logdetHorizontal;
}
std::array<double,2> CodingUnitInfo::getLogdetVertical() const {
    return logdetVertical;
}
std::array<double,2> CodingUnitInfo::getLogdetAverage() const {
    return logdetAverage;
}
double CodingUnitInfo::getRate() const {
    return rate;
}
double CodingUnitInfo::getPSNR() const {
    return PSNR;
}
double CodingUnitInfo::getMSE() const {
    return std::pow(1024.0, 2) / std::pow(10.0, PSNR / 10.0);
}
int CodingUnitInfo::getAngleHeuristicUsed() const {
    int heuristic;
    if (angleHeuristicUsed == 0)
        return 0;
    if (angleHeuristicUsed > 0 && angleHeuristicUsed < 4)
        return 1;
    if (angleHeuristicUsed > 3 && angleHeuristicUsed < 7)
        return 2;

    return 3;
}
nlohmann::json CodingUnitInfo::toJson() const {
    nlohmann::json j = nlohmann::json::object_t();
    j["lightFieldPosition"] = {lightFieldPosition[0], lightFieldPosition[1], lightFieldPosition[2], lightFieldPosition[3]};
    j["size"] = {size[0], size[1], size[2], size[3]};
    j["gridSearchAngle"] = gridSearchAngle;
    j["structureTensorHorizontal"] = {structureTensorHorizontal[0],structureTensorHorizontal[1]};
    j["structureTensorVertical"] = {structureTensorVertical[0],structureTensorVertical[1]};
    j["structureTensorAverage"] = {structureTensorAverage[0],structureTensorAverage[1]};
    j["logdetHorizontal"] = {logdetHorizontal[0],logdetHorizontal[1]};
    j["logdetVertical"] = {logdetVertical[0],logdetVertical[1]};
    j["logdetAverage"] = {logdetAverage[0],logdetAverage[1]};
    j["covarianceHorizontal"] = {covarianceHorizontal[0],covarianceHorizontal[1]};
    j["covarianceVertical"] = {covarianceVertical[0],covarianceVertical[1]};
    j["covarianceAverage"] = {covarianceAverage[0],covarianceAverage[1]};
    j["rate"] = rate;
    j["PSNR"] = PSNR;
    j["angleHeuristicUsed"] = angleHeuristicUsed;
    j["sgtSideInfo"] = sgtSideInfo.toJson(); // Assuming SgtSideInfo has a toJson method
    return j;
}
SgtSideInfo CodingUnitInfo::getSgtSideInfo() const {
    return sgtSideInfo;
}
std::map<double, double> CodingUnitInfo::getGridSearchAngle() const {
    return gridSearchAngle;
}

CodingUnitInfo CodingUnitInfo::fromJson(const nlohmann::json& j) {
    std::array<int64_t, 4> size = {
        j["size"][0].get<int64_t>(),
        j["size"][1].get<int64_t>(),
        j["size"][2].get<int64_t>(),
        j["size"][3].get<int64_t>()
    };
    //std::cout<<size[2]<<" "<<size[3]<<std::endl;

    //std::cout<<"    1"<<std::endl;
    std::array<int64_t, 4> lightFieldPosition = {
        j["lightFieldPosition"][0].get<int64_t>(),
        j["lightFieldPosition"][1].get<int64_t>(),
        j["lightFieldPosition"][2].get<int64_t>(),
        j["lightFieldPosition"][3].get<int64_t>()
    };
    //std::cout<<lightFieldPosition[2]<<" "<<lightFieldPosition[3]<<std::endl;
    //std::cout<<"    2"<<std::endl;

    std::map<double, double> gridSearchAngle = j["gridSearchAngle"].get<std::map<double, double>>();
    //std::cout<<"    3"<<std::endl;
    //std::cout<<"Reading Structure Tensor "<<std::endl;
    std::array<double,2> structureTensorHorizontal = {
        j["structureTensorHorizontal"][0].is_null() ? -90.0 : j["structureTensorHorizontal"][0].get<double>(),
        j["structureTensorHorizontal"][1].is_null() ? -90.0 : j["structureTensorHorizontal"][1].get<double>(),
    };
    //std::cout<<"    4"<<std::endl;

    std::array<double,2> structureTensorVertical = {
        j["structureTensorVertical"][0].is_null() ? -90.0 : j["structureTensorVertical"][0].get<double>(),
        j["structureTensorVertical"][1].is_null() ? -90.0 : j["structureTensorVertical"][1].get<double>(),
    };
    //std::cout<<"    5"<<std::endl;

    std::array<double,2> structureTensorAverage = {
        j["structureTensorAverage"][0].is_null() ? -90.0 : j["structureTensorAverage"][0].get<double>(),
        j["structureTensorAverage"][1].is_null() ? -90.0 : j["structureTensorAverage"][1].get<double>(),
    };
    //std::cout<<"Read Structure Tensor "<<std::endl;

    //std::cout<<"    6"<<std::endl;

    std::array<double,2> logdetHorizontal = {
        j["logdetHorizontal"][0].get<double>(),
        j["logdetHorizontal"][1].get<double>(),
    };
    //std::cout<<"    7"<<std::endl;

    std::array<double,2> logdetVertical = {
        j["logdetVertical"][0].get<double>(),
        j["logdetVertical"][1].get<double>(),
    };
    //std::cout<<"    8"<<std::endl;

    std::array<double,2> logdetAverage = {
        j["logdetAverage"][0].get<double>(),
        j["logdetAverage"][1].get<double>(),
    };
    //std::cout<<"Reading Cov "<<std::endl;

    std::array<double,2> covarianceHorizontal = {
        j["covarianceHorizontal"][0].get<double>(),
        j["covarianceHorizontal"][1].get<double>(),
    };
    //std::cout<<"    7"<<std::endl;
    //std::cout<<"Read Horizontal "<<covarianceHorizontal[0]<<" "<<covarianceHorizontal[1]<<std::endl;

    std::array<double,2> covarianceVertical = {
        j["covarianceVertical"][0].get<double>(),
        j["covarianceVertical"][1].get<double>(),
    };
    //std::cout<<"    8"<<std::endl;
    //std::cout<<"Read Vertical "<<std::endl;

    std::array<double,2> covarianceAverage = {
        j["covarianceAverage"][0].get<double>(),
        j["covarianceAverage"][1].get<double>(),
    };
    //std::cout<<"Read Average "<<std::endl;

    //std::cout<<"    9"<<std::endl;
    //std::cout<<"Read Cov "<<std::endl;

    double rate = j["rate"].get<double>();
    //std::cout<<"    10"<<std::endl;

    double PSNR = j["PSNR"].get<double>();
    //std::cout<<"    11"<<std::endl;

    int angleHeuristicUsed = j["angleHeuristicUsed"].get<int>();
    //std::cout<<"    12"<<std::endl;

    SgtSideInfo sgtSideInfo = SgtSideInfo::fromJson(j["sgtSideInfo"]); // Assuming SgtSideInfo has a fromJson method
    //std::cout<<"    13"<<std::endl;

    CodingUnitInfo info(size, lightFieldPosition);
    info.setGridSearchAngle(gridSearchAngle);
    info.setStructureTensorHorizontal(structureTensorHorizontal);
    info.setStructureTensorVertical(structureTensorVertical);
    info.setStructureTensorAverage(structureTensorAverage);
    info.setCovarianceHorizontal(covarianceHorizontal);
    info.setCovarianceVertical(covarianceVertical);
    info.setCovarianceAverage(covarianceAverage);
    info.setLogdetHorizontal(logdetHorizontal);
    info.setLogdetVertical(logdetVertical);
    info.setLogdetAverage(logdetAverage);
    info.setRate(rate);
    info.setPSNR(PSNR);
    info.setAngleHeuristicUsed(angleHeuristicUsed);
    info.setSgtSideInfo(sgtSideInfo);

    return info;
}
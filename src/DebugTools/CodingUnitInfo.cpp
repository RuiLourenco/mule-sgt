#include "DebugTools/CodingUnitInfo.h"
#include <nlohmann/json.hpp>


// Returns a tuple: (angle_multiple_of_ten_with_min_cost, overall_min_angle, is_within_10_degrees, min_cost_within_10, rel_cost_diff)
std::tuple<double, double, bool, double, double> CodingUnitInfo::analyzeGridSearchAngle(int delta) const {
    if (gridSearchAngle.empty()) {
        throw std::runtime_error("gridSearchAngle map is empty.");
    }

    // Find the angle (multiple of 10) with minimum cost
    double min_cost_multiple_of_ten = std::numeric_limits<double>::max();
    double angle_multiple_of_ten = 0;
    for (const auto& [angle, cost] : gridSearchAngle) {
        if (static_cast<int>(angle) % delta == 0) {
            if (cost < min_cost_multiple_of_ten) {
                min_cost_multiple_of_ten = cost;
                angle_multiple_of_ten = angle;
            }
        }
    }

    // Find the overall minimum angle
    double min_cost_overall = std::numeric_limits<double>::max();
    double min_angle_overall = 0;
    for (const auto& [angle, cost] : gridSearchAngle) {
        if (cost < min_cost_overall) {
            min_cost_overall = cost;
            min_angle_overall = angle;
        }
    }

    // Find the minimum cost within 10 degrees of the multiple-of-ten minimum
    double min_cost_within_10 = std::numeric_limits<double>::max();
    for (const auto& [angle, cost] : gridSearchAngle) {
        if (std::abs(angle - angle_multiple_of_ten) <=  delta) {
            if (cost < min_cost_within_10) {
                min_cost_within_10 = cost;
            }
        }
    }

    // Compute relative cost difference (zero if they match)
    double rel_cost_diff = 0.0;
    if (min_cost_within_10 > 0.0) {
        rel_cost_diff = (min_cost_within_10 - min_cost_overall) / min_cost_within_10;
    }

    // Check if the overall minimum angle is within 10 degrees of the multiple-of-ten minimum
    bool is_within_10_degrees = std::abs(min_angle_overall - angle_multiple_of_ten) <= delta;

    return std::make_tuple(angle_multiple_of_ten, min_angle_overall, is_within_10_degrees, min_cost_within_10, rel_cost_diff);
}
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
    file << "plt.axvline(x=" << getBestLogdetAngle() << ", color='g', linestyle='--', label='Best Logdet Angle')\n";
    file << "plt.axvline(x=" << getBestCovarianceAngle() << ", color='b', linestyle='--', label='Best Covariance Angle')\n";
    file << "plt.axvline(x=" << getBestGridSearchAngle() << ", color='m', linestyle='--', label='Best GridSearch Angle')\n";
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
double CodingUnitInfo::getChosenRhoS() const{
    return this->sgtSideInfo.getRhoS();
}
double CodingUnitInfo::getChosenRhoU() const{
    return this->sgtSideInfo.getRhoU();
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
    auto get_int64 = [](const nlohmann::json& arr, size_t idx) {
        if (!arr.is_array() || arr.size() <= idx || arr[idx].is_null()) return int64_t(0);
        try {
            return arr[idx].get<int64_t>();
        } catch (...) {
            return int64_t(0);
        }
    };
    auto get_double = [](const nlohmann::json& arr, size_t idx) {
        if (!arr.is_array() || arr.size() <= idx || arr[idx].is_null()) return 0.0;
        try {
            return arr[idx].get<double>();
        } catch (...) {
            return 0.0;
        }
    };

    std::array<int64_t, 4> size = {
        get_int64(j.value("size", nlohmann::json::array()), 0),
        get_int64(j.value("size", nlohmann::json::array()), 1),
        get_int64(j.value("size", nlohmann::json::array()), 2),
        get_int64(j.value("size", nlohmann::json::array()), 3)
    };

    std::array<int64_t, 4> lightFieldPosition = {
        get_int64(j.value("lightFieldPosition", nlohmann::json::array()), 0),
        get_int64(j.value("lightFieldPosition", nlohmann::json::array()), 1),
        get_int64(j.value("lightFieldPosition", nlohmann::json::array()), 2),
        get_int64(j.value("lightFieldPosition", nlohmann::json::array()), 3)
    };

    std::map<double, double> gridSearchAngle;
    if (j.contains("gridSearchAngle") && !j["gridSearchAngle"].is_null()) {
        try {
            gridSearchAngle = j["gridSearchAngle"].get<std::map<double, double>>();
        } catch (...) {
            gridSearchAngle.clear();
        }
    }

    std::array<double,2> structureTensorHorizontal = {
        get_double(j.value("structureTensorHorizontal", nlohmann::json::array()), 0),
        get_double(j.value("structureTensorHorizontal", nlohmann::json::array()), 1)
    };

    std::array<double,2> structureTensorVertical = {
        get_double(j.value("structureTensorVertical", nlohmann::json::array()), 0),
        get_double(j.value("structureTensorVertical", nlohmann::json::array()), 1)
    };

    std::array<double,2> structureTensorAverage = {
        get_double(j.value("structureTensorAverage", nlohmann::json::array()), 0),
        get_double(j.value("structureTensorAverage", nlohmann::json::array()), 1)
    };

    std::array<double,2> logdetHorizontal = {
        get_double(j.value("logdetHorizontal", nlohmann::json::array()), 0),
        get_double(j.value("logdetHorizontal", nlohmann::json::array()), 1)
    };

    std::array<double,2> logdetVertical = {
        get_double(j.value("logdetVertical", nlohmann::json::array()), 0),
        get_double(j.value("logdetVertical", nlohmann::json::array()), 1)
    };

    std::array<double,2> logdetAverage = {
        get_double(j.value("logdetAverage", nlohmann::json::array()), 0),
        get_double(j.value("logdetAverage", nlohmann::json::array()), 1)
    };

    std::array<double,2> covarianceHorizontal = {
        get_double(j.value("covarianceHorizontal", nlohmann::json::array()), 0),
        get_double(j.value("covarianceHorizontal", nlohmann::json::array()), 1)
    };

    std::array<double,2> covarianceVertical = {
        get_double(j.value("covarianceVertical", nlohmann::json::array()), 0),
        get_double(j.value("covarianceVertical", nlohmann::json::array()), 1)
    };

    std::array<double,2> covarianceAverage = {
        get_double(j.value("covarianceAverage", nlohmann::json::array()), 0),
        get_double(j.value("covarianceAverage", nlohmann::json::array()), 1)
    };

    double rate = (j.contains("rate") && !j["rate"].is_null()) ? j["rate"].get<double>() : 0.0;
    double PSNR = (j.contains("PSNR") && !j["PSNR"].is_null()) ? j["PSNR"].get<double>() : 0.0;
    int angleHeuristicUsed = (j.contains("angleHeuristicUsed") && !j["angleHeuristicUsed"].is_null()) ? j["angleHeuristicUsed"].get<int>() : 0;

    SgtSideInfo sgtSideInfo;
    if (j.contains("sgtSideInfo") && !j["sgtSideInfo"].is_null()) {
        try {
            sgtSideInfo = SgtSideInfo::fromJson(j["sgtSideInfo"]);
        } catch (...) {
            // leave as default
        }
    }

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

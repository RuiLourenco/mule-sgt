#ifndef CODING_UNIT_INFO_H
#define CODING_UNIT_INFO_H

#include <array>
#include <tuple>
#include <LightField/Block4D_.h> // Include the header for SgtSideInfo
#include <map> // Include map header
#include <nlohmann/json.hpp> // Include the nlohmann/json library


class SgtSideInfo; // Forward declaration of SgtSideInfo

enum AngleHeuristic {
    GRID_SEARCH = 0,
    STRUCTURE_TENSOR_HORIZONTAL = 1,
    STRUCTURE_TENSOR_VERTICAL = 2,
    STRUCTURE_TENSOR_AVERAGE = 3,
    LOGDET_HORIZONTAL = 4,
    LOGDET_VERTICAL = 5,
    LOGDET_AVERAGE = 6
};

class CodingUnitInfo {
public:
    CodingUnitInfo(const std::array<int64_t, 4>& size, const std::array<int64_t, 4>& lightFieldPosition);
    std::array<int64_t, 4> getLightFieldPosition() const;
    std::array<int64_t,4> getSize() const;
    std::array<double, 2> getStructureTensorHorizontal() const;
    std::array<double, 2> getStructureTensorVertical() const;
    std::array<double, 2> getStructureTensorAverage() const;
    double getStructureTensorHorizontalAngle() const;
    double getStructureTensorVerticalAngle() const;
    double getStructureTensorAverageAngle() const;
    std::array<double, 2> getLogdetHorizontal() const;
    std::array<double, 2> getLogdetVertical() const;
    std::array<double, 2> getLogdetAverage() const;
    double getLogdetHorizontalAngle() const;
    double getLogdetVerticalAngle() const;
    double getLogdetAverageAngle() const;
    double getBestGridSearchAngle() const;
    double getBestGridSearchCost() const;
    double getChosenAngle() const;
    double getBestStructureTensorAngle() const;
    double getBestStructureTensorCost() const;  
    double getBestLogdetCost() const ;
    double getBestLogdetAngle() const;


    double getRate() const;
    double getPSNR() const;
    int getAngleHeuristicUsed() const;
    std::map<double, double> getGridSearchAngle() const; // Changed to std::map
    SgtSideInfo getSgtSideInfo() const;
    void setGridSearchAngle(const std::map<double, double>& gridSearchAngle); // Changed to std::map
    void setStructureTensorHorizontal(const std::array<double, 2>& structureTensorHorizontal);
    void setStructureTensorVertical(const std::array<double, 2>& structureTensorVertical);
    void setStructureTensorAverage(const std::array<double, 2>& structureTensorAverage);
    void setLogdetHorizontal(const std::array<double, 2>& logdetHorizontal);
    void setLogdetVertical(const std::array<double, 2>& logdetVertical);
    void setLogdetAverage(const std::array<double, 2>& logdetAverage);
    void setRate(double rate);
    void setPSNR(double PSNR);
    void setAngleHeuristicUsed(int angleHeuristicUsed);
    void setSgtSideInfo(const SgtSideInfo& sgtSideInfo);
    void addGridSearchAngle(double angle, double cost);
    nlohmann::json toJson() const;
    static CodingUnitInfo fromJson(const nlohmann::json& j);

private:
    std::array<int64_t, 4> size;
    std::array<int64_t, 4> lightFieldPosition;
    std::map<double, double> gridSearchAngle; // Changed to std::map
    std::array<double, 2> structureTensorHorizontal;
    std::array<double, 2> structureTensorVertical;
    std::array<double, 2> structureTensorAverage;
    std::array<double, 2> logdetHorizontal;
    std::array<double, 2> logdetVertical;
    std::array<double, 2> logdetAverage;
    double rate; // Rate in BPP
    double PSNR; // PSNR in dB
    int angleHeuristicUsed;
    SgtSideInfo sgtSideInfo;
};
#endif // CODING_UNIT_INFO_H
#ifndef CODING_UNIT_INFO_H
#define CODING_UNIT_INFO_H

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <LightField/Block4D_.h> // SgtSideInfo

// Which angle/rho estimator produced a search candidate.
enum class CandidateMethod : uint8_t {
    Zero,
    StructureTensorH,
    StructureTensorV,
    StructureTensorAvg,
    StructureTensorPooled,
    StructureTensorPerDirection,
    StructureTensorEigen4D,
    StructureTensorEpiH,
    StructureTensorEpiV,
    CovarianceH,
    CovarianceV,
    CovarianceAvg,
    LogdetH,
    LogdetV,
    LogdetAvg,
    GridSearch,
    RhoSearch,
    LeastSquaresRho,
    Unknown
};

const char* toString(CandidateMethod method);
CandidateMethod candidateMethodFromString(const std::string& name);

// One configuration the RD search evaluated for a coding unit.
// `cost` is the Lagrangian J = D + lambda*R returned by the search, before the
// partition-flag and side-information bits are added.
struct SearchCandidate {
    CandidateMethod method = CandidateMethod::Unknown;
    SgtSideInfo ssi;
    double cost = 0.0;
};

// A leaf of the partition tree: one block that was transformed and entropy coded.
//
// Encoder records hold the search trace, the chosen side information and the
// measured bits/distortion. Decoder records hold geometry, side information and
// bits only (the decoder has no original to measure distortion against).
class CodingUnitInfo {
public:
    CodingUnitInfo() = default;
    CodingUnitInfo(const std::array<int64_t, 4>& lightFieldPosition, const std::array<int64_t, 4>& size);

    // --- Geometry -----------------------------------------------------------
    const std::array<int64_t, 4>& getLightFieldPosition() const { return lightFieldPosition; }
    const std::array<int64_t, 4>& getSize() const { return size; }
    int64_t numSamples() const;

    // --- Search trace (encoder only) ----------------------------------------
    void addCandidate(CandidateMethod method, const SgtSideInfo& ssi, double cost);
    const std::vector<SearchCandidate>& getCandidates() const { return candidates; }
    // Lowest-cost candidate among those whose method passes `filter`.
    std::optional<SearchCandidate> bestCandidate(const std::function<bool(CandidateMethod)>& filter) const;
    std::optional<SearchCandidate> bestCandidate(CandidateMethod method) const;

    // --- Diagnostics (encoder, --probe-st-estimators) ---------------------------
    // Configurations evaluated only for analysis. They never influence the coding
    // decision and are not candidates for setChosen.
    void addProbe(CandidateMethod method, const SgtSideInfo& ssi, double cost);
    const std::vector<SearchCandidate>& getProbes() const { return probes; }
    // 4D structure tensor of the block (axes t, s, v, u), row-major.
    void setStructureTensor(const std::array<double, 16>& tensor) { structureTensor = tensor; }
    const std::optional<std::array<double, 16>>& getStructureTensor() const { return structureTensor; }

    // --- Outcome --------------------------------------------------------------
    // Stores the side information that was actually coded and links it to the
    // candidate that produced it (matched on the quantized side-information codes).
    void setChosen(const SgtSideInfo& ssi);
    const std::optional<SgtSideInfo>& getSgtSideInfo() const { return chosenSsi; }
    // The candidate that produced the coded side information, if it was recorded.
    std::optional<SearchCandidate> chosenCandidate() const;
    // Drops every candidate except the chosen one ("winner" recording level).
    void keepOnlyChosenCandidate();

    // --- Measurements ---------------------------------------------------------
    // Bits spent on this unit: partition flag + side information + coefficients.
    void setBits(double bits) { this->bits = bits; }
    double getBits() const { return bits; }
    double getBitsPerSample() const;

    // Sum of squared errors in the pixel domain (transform-domain SSE / gain^2).
    void setSse(double sse) { this->sse = sse; }
    std::optional<double> getSse() const { return sse; }
    std::optional<double> getMse() const;
    std::optional<double> getPsnr(double peak = 1024.0) const;

    nlohmann::json toJson() const;
    static CodingUnitInfo fromJson(const nlohmann::json& j);

    // Column order of each serialized candidate (stored as a compact array).
    static const std::vector<std::string>& candidateFields();

private:
    std::array<int64_t, 4> lightFieldPosition{};
    std::array<int64_t, 4> size{};
    std::vector<SearchCandidate> candidates;
    std::vector<SearchCandidate> probes;
    std::optional<std::array<double, 16>> structureTensor;
    std::optional<SgtSideInfo> chosenSsi;
    int chosenIndex = -1; // index into `candidates`, -1 if not recorded
    double bits = 0.0;
    std::optional<double> sse;
};

#endif // CODING_UNIT_INFO_H

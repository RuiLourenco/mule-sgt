#include "DebugTools/CodingUnitInfo.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

struct MethodName {
    CandidateMethod method;
    const char* name;
};

constexpr MethodName METHOD_NAMES[] = {
    {CandidateMethod::Zero, "zero"},
    {CandidateMethod::StructureTensorH, "structure_tensor_h"},
    {CandidateMethod::StructureTensorV, "structure_tensor_v"},
    {CandidateMethod::StructureTensorAvg, "structure_tensor_avg"},
    {CandidateMethod::StructureTensorPooled, "structure_tensor_pooled"},
    {CandidateMethod::StructureTensorPerDirection, "structure_tensor_per_direction"},
    {CandidateMethod::StructureTensorEigen4D, "structure_tensor_eigen4d"},
    {CandidateMethod::StructureTensorEpiH, "structure_tensor_epi_h"},
    {CandidateMethod::StructureTensorEpiV, "structure_tensor_epi_v"},
    {CandidateMethod::CovarianceH, "covariance_h"},
    {CandidateMethod::CovarianceV, "covariance_v"},
    {CandidateMethod::CovarianceAvg, "covariance_avg"},
    {CandidateMethod::LogdetH, "logdet_h"},
    {CandidateMethod::LogdetV, "logdet_v"},
    {CandidateMethod::LogdetAvg, "logdet_avg"},
    {CandidateMethod::GridSearch, "grid_search"},
    {CandidateMethod::RhoSearch, "rho_search"},
    {CandidateMethod::LeastSquaresRho, "least_squares_rho"},
    {CandidateMethod::Unknown, "unknown"},
};

// Side information is coded as integers, so two configurations are the same
// coded configuration exactly when all their integer codes agree.
bool sameCodes(const SgtSideInfo& a, const SgtSideInfo& b) {
    return a.getAngleVCode() == b.getAngleVCode() && a.getAngleHCode() == b.getAngleHCode() &&
           a.getRhoSCode() == b.getRhoSCode() && a.getRhoTCode() == b.getRhoTCode() &&
           a.getRhoUCode() == b.getRhoUCode() && a.getRhoVCode() == b.getRhoVCode();
}

// Wide enough that any angle the codec produces survives the round trip.
constexpr std::array<double, 2> JSON_DISPARITY_RANGE = {-100, 100};

SgtSideInfo ssiFromValues(double angleV, double angleH, double rhoS, double rhoT, double rhoU, double rhoV) {
    SgtSideInfo ssi(JSON_DISPARITY_RANGE);
    ssi.setAngleV(angleV);
    ssi.setAngleH(angleH);
    ssi.setRhoS(rhoS);
    ssi.setRhoT(rhoT);
    ssi.setRhoU(rhoU);
    ssi.setRhoV(rhoV);
    return ssi;
}

nlohmann::json candidatesToJson(const std::vector<SearchCandidate>& list) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& c : list) {
        out.push_back({toString(c.method), c.ssi.getAngleV(), c.ssi.getAngleH(), c.ssi.getRhoS(),
                       c.ssi.getRhoT(), c.ssi.getRhoU(), c.ssi.getRhoV(), c.cost});
    }
    return out;
}

std::vector<SearchCandidate> candidatesFromJson(const nlohmann::json& list) {
    std::vector<SearchCandidate> out;
    for (const auto& c : list) {
        out.push_back({candidateMethodFromString(c.at(0).get<std::string>()),
                       ssiFromValues(c.at(1), c.at(2), c.at(3), c.at(4), c.at(5), c.at(6)),
                       c.at(7).is_null() ? std::numeric_limits<double>::quiet_NaN() : c.at(7).get<double>()});
    }
    return out;
}

template <typename T>
std::array<T, 4> array4(const nlohmann::json& j) {
    return {j.at(0).get<T>(), j.at(1).get<T>(), j.at(2).get<T>(), j.at(3).get<T>()};
}

} // namespace

const char* toString(CandidateMethod method) {
    for (const auto& entry : METHOD_NAMES) {
        if (entry.method == method) return entry.name;
    }
    return "unknown";
}

CandidateMethod candidateMethodFromString(const std::string& name) {
    for (const auto& entry : METHOD_NAMES) {
        if (name == entry.name) return entry.method;
    }
    return CandidateMethod::Unknown;
}

CodingUnitInfo::CodingUnitInfo(const std::array<int64_t, 4>& lightFieldPosition, const std::array<int64_t, 4>& size)
    : lightFieldPosition(lightFieldPosition), size(size) {}

int64_t CodingUnitInfo::numSamples() const {
    return size[0] * size[1] * size[2] * size[3];
}

void CodingUnitInfo::addCandidate(CandidateMethod method, const SgtSideInfo& ssi, double cost) {
    candidates.push_back({method, ssi, cost});
}

void CodingUnitInfo::addProbe(CandidateMethod method, const SgtSideInfo& ssi, double cost) {
    probes.push_back({method, ssi, cost});
}

std::optional<SearchCandidate> CodingUnitInfo::bestCandidate(const std::function<bool(CandidateMethod)>& filter) const {
    const SearchCandidate* best = nullptr;
    for (const auto& candidate : candidates) {
        if (!filter(candidate.method)) continue;
        if (best == nullptr || candidate.cost < best->cost) best = &candidate;
    }
    if (best == nullptr) return std::nullopt;
    return *best;
}

std::optional<SearchCandidate> CodingUnitInfo::bestCandidate(CandidateMethod method) const {
    return bestCandidate([method](CandidateMethod m) { return m == method; });
}

void CodingUnitInfo::setChosen(const SgtSideInfo& ssi) {
    chosenSsi = ssi;
    chosenIndex = -1;
    // Search backwards: later stages refine earlier ones, so the last match is the
    // stage that actually produced the coded configuration.
    for (int i = static_cast<int>(candidates.size()) - 1; i >= 0; --i) {
        if (sameCodes(candidates[i].ssi, ssi)) {
            chosenIndex = i;
            break;
        }
    }
}

std::optional<SearchCandidate> CodingUnitInfo::chosenCandidate() const {
    if (chosenIndex < 0) return std::nullopt;
    return candidates[chosenIndex];
}

void CodingUnitInfo::keepOnlyChosenCandidate() {
    if (chosenIndex < 0) {
        candidates.clear();
        return;
    }
    SearchCandidate chosen = candidates[chosenIndex];
    candidates.assign(1, chosen);
    chosenIndex = 0;
}

double CodingUnitInfo::getBitsPerSample() const {
    int64_t n = numSamples();
    return n > 0 ? bits / static_cast<double>(n) : 0.0;
}

std::optional<double> CodingUnitInfo::getMse() const {
    if (!sse || numSamples() == 0) return std::nullopt;
    return *sse / static_cast<double>(numSamples());
}

std::optional<double> CodingUnitInfo::getPsnr(double peak) const {
    auto mse = getMse();
    if (!mse) return std::nullopt;
    if (*mse <= 0.0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(peak * peak / *mse);
}

const std::vector<std::string>& CodingUnitInfo::candidateFields() {
    static const std::vector<std::string> fields = {
        "method", "angleV", "angleH", "rhoS", "rhoT", "rhoU", "rhoV", "cost"};
    return fields;
}

nlohmann::json CodingUnitInfo::toJson() const {
    nlohmann::json j;
    j["lightFieldPosition"] = lightFieldPosition;
    j["size"] = size;
    j["bits"] = bits;
    j["sse"] = sse ? nlohmann::json(*sse) : nlohmann::json(nullptr);
    j["ssi"] = chosenSsi ? chosenSsi->toJson() : nlohmann::json(nullptr);

    auto chosen = chosenCandidate();
    j["method"] = chosen ? nlohmann::json(toString(chosen->method)) : nlohmann::json(nullptr);
    j["searchCost"] = chosen ? nlohmann::json(chosen->cost) : nlohmann::json(nullptr);
    j["chosenCandidate"] = chosenIndex >= 0 ? nlohmann::json(chosenIndex) : nlohmann::json(nullptr);

    j["candidates"] = candidatesToJson(candidates);
    // Diagnostic fields are only written when recorded, to keep ordinary files small.
    if (!probes.empty()) j["probes"] = candidatesToJson(probes);
    if (structureTensor) j["structureTensor"] = *structureTensor;
    return j;
}

CodingUnitInfo CodingUnitInfo::fromJson(const nlohmann::json& j) {
    CodingUnitInfo unit(array4<int64_t>(j.at("lightFieldPosition")), array4<int64_t>(j.at("size")));
    unit.bits = j.value("bits", 0.0);
    if (j.contains("sse") && !j["sse"].is_null()) unit.sse = j["sse"].get<double>();
    if (j.contains("ssi") && !j["ssi"].is_null()) {
        const auto& s = j["ssi"];
        unit.chosenSsi = ssiFromValues(s.at("angleV"), s.at("angleH"), s.at("rhoS"), s.at("rhoT"), s.at("rhoU"), s.at("rhoV"));
    }
    if (j.contains("candidates")) unit.candidates = candidatesFromJson(j["candidates"]);
    if (j.contains("probes")) unit.probes = candidatesFromJson(j["probes"]);
    if (j.contains("structureTensor") && !j["structureTensor"].is_null()) {
        unit.structureTensor = j["structureTensor"].get<std::array<double, 16>>();
    }
    if (j.contains("chosenCandidate") && !j["chosenCandidate"].is_null()) {
        int index = j["chosenCandidate"].get<int>();
        if (index < 0 || index >= static_cast<int>(unit.candidates.size())) {
            throw std::runtime_error("CodingUnitInfo::fromJson: chosenCandidate out of range");
        }
        unit.chosenIndex = index;
    }
    return unit;
}

#include "DebugTools/CodingPartitionInfo.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

constexpr const char* FORMAT_NAME = "mule-sgt-partition-info";

template <typename T>
std::array<T, 4> array4(const nlohmann::json& j) {
    return {j.at(0).get<T>(), j.at(1).get<T>(), j.at(2).get<T>(), j.at(3).get<T>()};
}

} // namespace

const char* toString(PartitionInfoLevel level) {
    switch (level) {
        case PartitionInfoLevel::Off: return "off";
        case PartitionInfoLevel::Winner: return "winner";
        case PartitionInfoLevel::Full: return "full";
    }
    return "off";
}

PartitionInfoLevel partitionInfoLevelFromString(const std::string& name) {
    if (name == "off") return PartitionInfoLevel::Off;
    if (name == "winner") return PartitionInfoLevel::Winner;
    if (name == "full") return PartitionInfoLevel::Full;
    throw std::invalid_argument("Unknown partition info level '" + name + "' (expected off, winner or full)");
}

// --- CodingPartitionInfo ------------------------------------------------------

CodingPartitionInfo::CodingPartitionInfo(const std::array<int64_t, 4>& lightFieldPosition,
                                         const std::array<int64_t, 4>& size,
                                         int channel)
    : lightFieldPosition(lightFieldPosition), size(size), channel(channel) {}

std::optional<double> CodingPartitionInfo::getTotalSse() const {
    double total = 0.0;
    for (const auto& unit : codingUnitInfos) {
        auto sse = unit.getSse();
        if (!sse) return std::nullopt;
        total += *sse;
    }
    return total;
}

nlohmann::json CodingPartitionInfo::toJson() const {
    nlohmann::json j;
    j["channel"] = channel;
    j["lightFieldPosition"] = lightFieldPosition;
    j["size"] = size;
    j["splitCode"] = splitCode;
    j["bits"] = totalBits;
    auto sse = getTotalSse();
    j["sse"] = sse ? nlohmann::json(*sse) : nlohmann::json(nullptr);
    nlohmann::json units = nlohmann::json::array();
    for (const auto& unit : codingUnitInfos) units.push_back(unit.toJson());
    j["codingUnits"] = std::move(units);
    return j;
}

CodingPartitionInfo CodingPartitionInfo::fromJson(const nlohmann::json& j) {
    CodingPartitionInfo partition(array4<int64_t>(j.at("lightFieldPosition")),
                                  array4<int64_t>(j.at("size")),
                                  j.value("channel", 0));
    partition.splitCode = j.value("splitCode", std::string());
    partition.totalBits = j.value("bits", 0.0);
    for (const auto& unit : j.at("codingUnits")) {
        partition.codingUnitInfos.push_back(CodingUnitInfo::fromJson(unit));
    }
    return partition;
}

// --- PartitionInfoFile --------------------------------------------------------

nlohmann::json PartitionInfoFile::toJson() const {
    nlohmann::json j;
    j["format"] = FORMAT_NAME;
    j["version"] = VERSION;
    j["producer"] = producer;
    j["level"] = toString(level);
    j["metadata"] = metadata;
    j["candidateFields"] = CodingUnitInfo::candidateFields();
    nlohmann::json list = nlohmann::json::array();
    for (const auto& partition : partitions) list.push_back(partition.toJson());
    j["partitions"] = std::move(list);
    return j;
}

PartitionInfoFile PartitionInfoFile::fromJson(const nlohmann::json& j) {
    if (!j.is_object() || j.value("format", std::string()) != FORMAT_NAME) {
        throw std::runtime_error("Not a partition info file (old files from before version 2 are not supported)");
    }
    if (j.value("version", 0) != VERSION) {
        throw std::runtime_error("Unsupported partition info version " + std::to_string(j.value("version", 0)));
    }
    PartitionInfoFile file;
    file.producer = j.value("producer", std::string());
    file.level = partitionInfoLevelFromString(j.value("level", std::string("full")));
    file.metadata = j.value("metadata", nlohmann::json::object());
    for (const auto& partition : j.at("partitions")) {
        file.partitions.push_back(CodingPartitionInfo::fromJson(partition));
    }
    return file;
}

void PartitionInfoFile::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("Unable to open " + filename + " for writing");
    out << toJson().dump();
    if (!out) throw std::runtime_error("Failed writing " + filename);
}

PartitionInfoFile PartitionInfoFile::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Unable to open " + filename);
    nlohmann::json j;
    in >> j;
    return fromJson(j);
}

// --- rasterize ----------------------------------------------------------------

at::Tensor rasterize(const std::vector<CodingPartitionInfo>& partitions,
                     int channel,
                     std::array<int64_t, 2> view,
                     const std::function<std::optional<double>(const CodingUnitInfo&)>& field) {
    int64_t minRow = std::numeric_limits<int64_t>::max();
    int64_t minCol = std::numeric_limits<int64_t>::max();
    int64_t maxRow = std::numeric_limits<int64_t>::min();
    int64_t maxCol = std::numeric_limits<int64_t>::min();
    for (const auto& partition : partitions) {
        if (partition.getChannel() != channel) continue;
        const auto& pos = partition.getLightFieldPosition();
        const auto& size = partition.getSize();
        minRow = std::min(minRow, pos[2]);
        minCol = std::min(minCol, pos[3]);
        maxRow = std::max(maxRow, pos[2] + size[2]);
        maxCol = std::max(maxCol, pos[3] + size[3]);
    }
    if (minRow > maxRow) return at::empty({0, 0}, at::kDouble);

    at::Tensor image = at::full({maxRow - minRow, maxCol - minCol},
                                std::numeric_limits<double>::quiet_NaN(), at::kDouble);
    auto accessor = image.accessor<double, 2>();
    for (const auto& partition : partitions) {
        if (partition.getChannel() != channel) continue;
        for (const auto& unit : partition.getCodingUnitInfos()) {
            const auto& pos = unit.getLightFieldPosition();
            const auto& size = unit.getSize();
            bool coversView = view[0] >= pos[0] && view[0] < pos[0] + size[0] &&
                              view[1] >= pos[1] && view[1] < pos[1] + size[1];
            if (!coversView) continue;
            auto value = field(unit);
            if (!value) continue;
            int64_t rowEnd = std::min(pos[2] + size[2], maxRow);
            int64_t colEnd = std::min(pos[3] + size[3], maxCol);
            for (int64_t r = pos[2]; r < rowEnd; ++r)
                for (int64_t c = pos[3]; c < colEnd; ++c)
                    accessor[r - minRow][c - minCol] = *value;
        }
    }
    return image;
}

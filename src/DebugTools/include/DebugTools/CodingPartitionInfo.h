#ifndef CODING_PARTITION_INFO_H
#define CODING_PARTITION_INFO_H

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include "DebugTools/CodingUnitInfo.h"

// How much the codec records about each partition.
//   Off    - nothing is recorded and no info file is written.
//   Winner - one record per coded unit, holding only the chosen candidate.
//   Full   - one record per coded unit, holding every candidate the search tried.
enum class PartitionInfoLevel { Off, Winner, Full };

const char* toString(PartitionInfoLevel level);
PartitionInfoLevel partitionInfoLevelFromString(const std::string& name);

// One maximum-size block of one colour channel, i.e. one call to the partition
// optimizer. Its coding units are the leaves of the partition tree, in coding order.
class CodingPartitionInfo {
public:
    CodingPartitionInfo() = default;
    CodingPartitionInfo(const std::array<int64_t, 4>& lightFieldPosition,
                        const std::array<int64_t, 4>& size,
                        int channel);

    const std::array<int64_t, 4>& getLightFieldPosition() const { return lightFieldPosition; }
    const std::array<int64_t, 4>& getSize() const { return size; }
    int getChannel() const { return channel; }

    // Partition tree flags in coding order: 'T' leaf, 'S' spatial split, 'V' view split.
    void setSplitCode(const std::string& code) { splitCode = code; }
    void appendSplitFlag(char flag) { splitCode += flag; }
    const std::string& getSplitCode() const { return splitCode; }

    // Every bit spent on this partition, including the minimum-bitplane header
    // and the split flags that are not attributed to any single unit.
    void setTotalBits(double bits) { totalBits = bits; }
    double getTotalBits() const { return totalBits; }
    // Sum of unit SSEs, or nothing if any unit has no distortion measurement.
    std::optional<double> getTotalSse() const;

    void appendCodingUnitInfo(CodingUnitInfo unit) { codingUnitInfos.push_back(std::move(unit)); }
    const std::vector<CodingUnitInfo>& getCodingUnitInfos() const { return codingUnitInfos; }

    nlohmann::json toJson() const;
    static CodingPartitionInfo fromJson(const nlohmann::json& j);

private:
    std::array<int64_t, 4> lightFieldPosition{};
    std::array<int64_t, 4> size{};
    int channel = 0;
    std::string splitCode;
    double totalBits = 0.0;
    std::vector<CodingUnitInfo> codingUnitInfos;
};

// Contents of an info JSON file.
struct PartitionInfoFile {
    static constexpr int VERSION = 2;

    std::string producer;                 // "encoder" or "decoder"
    PartitionInfoLevel level = PartitionInfoLevel::Full;
    nlohmann::json metadata = nlohmann::json::object(); // free-form run parameters
    std::vector<CodingPartitionInfo> partitions;

    nlohmann::json toJson() const;
    static PartitionInfoFile fromJson(const nlohmann::json& j);
    void save(const std::string& filename) const;
    static PartitionInfoFile load(const std::string& filename);
};

// Paints a per-unit value onto one view of one channel.
//
// The image covers the bounding box of all partitions of that channel, so cropped
// light fields work without knowing the crop offset. Pixels not covered by a unit
// that contains view (t, s) are NaN. Units whose `field` returns nothing stay NaN.
at::Tensor rasterize(const std::vector<CodingPartitionInfo>& partitions,
                     int channel,
                     std::array<int64_t, 2> view,
                     const std::function<std::optional<double>(const CodingUnitInfo&)>& field);

#endif // CODING_PARTITION_INFO_H

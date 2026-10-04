#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>

#include <DebugTools/CodingPartitionInfo.h>
#include <DebugTools/CodingUnitInfo.h>

namespace {

const std::array<double, 2> DISPARITY_RANGE = {-3.5, 3.5};

SgtSideInfo ssiWithAngle(double angle) {
    return SgtSideInfo(angle, angle, DISPARITY_RANGE);
}

// A 9x9x16x16 unit with a small search trace whose winner is the grid candidate at 2 degrees.
CodingUnitInfo makeUnit(std::array<int64_t, 4> position, std::array<int64_t, 4> size) {
    CodingUnitInfo unit(position, size);
    unit.addCandidate(CandidateMethod::Zero, ssiWithAngle(0), 500.0);
    unit.addCandidate(CandidateMethod::StructureTensorH, ssiWithAngle(5), 300.0);
    unit.addCandidate(CandidateMethod::GridSearch, ssiWithAngle(1), 250.0);
    unit.addCandidate(CandidateMethod::GridSearch, ssiWithAngle(2), 200.0);
    unit.setChosen(ssiWithAngle(2));
    unit.setBits(1234.5);
    unit.setSse(4096.0);
    return unit;
}

} // namespace

TEST(CodingUnitInfoTests, ChosenCandidateIsMatchedOnSideInformation) {
    CodingUnitInfo unit = makeUnit({0, 0, 0, 0}, {9, 9, 16, 16});
    auto chosen = unit.chosenCandidate();
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(chosen->method, CandidateMethod::GridSearch);
    EXPECT_DOUBLE_EQ(chosen->cost, 200.0);

    auto bestTensor = unit.bestCandidate(CandidateMethod::StructureTensorH);
    ASSERT_TRUE(bestTensor.has_value());
    EXPECT_NEAR(bestTensor->ssi.getAngleH(), 5.0, SgtSideInfo::PRECISION_ANGLE);
}

TEST(CodingUnitInfoTests, UnrecordedChoiceHasNoCandidate) {
    CodingUnitInfo unit({0, 0, 0, 0}, {9, 9, 16, 16});
    unit.addCandidate(CandidateMethod::Zero, ssiWithAngle(0), 10.0);
    unit.setChosen(ssiWithAngle(3));
    EXPECT_FALSE(unit.chosenCandidate().has_value());
    unit.keepOnlyChosenCandidate();
    EXPECT_TRUE(unit.getCandidates().empty());
}

TEST(CodingUnitInfoTests, WinnerLevelKeepsOnlyTheChosenCandidate) {
    CodingUnitInfo unit = makeUnit({0, 0, 0, 0}, {9, 9, 16, 16});
    unit.keepOnlyChosenCandidate();
    ASSERT_EQ(unit.getCandidates().size(), 1u);
    ASSERT_TRUE(unit.chosenCandidate().has_value());
    EXPECT_DOUBLE_EQ(unit.chosenCandidate()->cost, 200.0);
}

TEST(CodingUnitInfoTests, DerivedMeasurements) {
    CodingUnitInfo unit = makeUnit({0, 0, 0, 0}, {9, 9, 16, 16});
    const double samples = 9.0 * 9 * 16 * 16;
    EXPECT_DOUBLE_EQ(unit.getBitsPerSample(), 1234.5 / samples);
    EXPECT_DOUBLE_EQ(*unit.getMse(), 4096.0 / samples);
    EXPECT_DOUBLE_EQ(*unit.getPsnr(1024.0), 10.0 * std::log10(1024.0 * 1024.0 / (4096.0 / samples)));

    CodingUnitInfo decoded({0, 0, 0, 0}, {9, 9, 16, 16});
    EXPECT_FALSE(decoded.getPsnr().has_value());
}

TEST(PartitionInfoFileTests, JsonRoundTripPreservesEverything) {
    PartitionInfoFile file;
    file.producer = "encoder";
    file.level = PartitionInfoLevel::Full;
    file.metadata = {{"lambda", 0.5}};
    CodingPartitionInfo partition({0, 0, 32, 64}, {9, 9, 32, 32}, 1);
    partition.setSplitCode("STTTT");
    partition.setTotalBits(5000.0);
    partition.appendCodingUnitInfo(makeUnit({0, 0, 32, 64}, {9, 9, 16, 16}));
    partition.appendCodingUnitInfo(makeUnit({0, 0, 32, 80}, {9, 9, 16, 16}));
    file.partitions.push_back(partition);

    auto path = std::filesystem::temp_directory_path() / "debugtools_roundtrip_info.json";
    file.save(path.string());
    PartitionInfoFile loaded = PartitionInfoFile::load(path.string());
    std::filesystem::remove(path);

    EXPECT_EQ(loaded.producer, "encoder");
    EXPECT_EQ(loaded.level, PartitionInfoLevel::Full);
    EXPECT_DOUBLE_EQ(loaded.metadata["lambda"].get<double>(), 0.5);
    ASSERT_EQ(loaded.partitions.size(), 1u);
    const auto& p = loaded.partitions[0];
    EXPECT_EQ(p.getChannel(), 1);
    EXPECT_EQ(p.getSplitCode(), "STTTT");
    EXPECT_DOUBLE_EQ(p.getTotalBits(), 5000.0);
    EXPECT_DOUBLE_EQ(*p.getTotalSse(), 2 * 4096.0);
    ASSERT_EQ(p.getCodingUnitInfos().size(), 2u);

    const auto& unit = p.getCodingUnitInfos()[1];
    EXPECT_EQ(unit.getLightFieldPosition(), (std::array<int64_t, 4>{0, 0, 32, 80}));
    EXPECT_EQ(unit.getCandidates().size(), 4u);
    ASSERT_TRUE(unit.chosenCandidate().has_value());
    EXPECT_EQ(unit.chosenCandidate()->method, CandidateMethod::GridSearch);
    EXPECT_NEAR(unit.getSgtSideInfo()->getAngleH(), 2.0, SgtSideInfo::PRECISION_ANGLE);
    EXPECT_DOUBLE_EQ(unit.getBits(), 1234.5);
}

TEST(PartitionInfoFileTests, RejectsOldFormat) {
    nlohmann::json old = nlohmann::json::array({{{"codingUnitInfos", nlohmann::json::array()}}});
    EXPECT_THROW(PartitionInfoFile::fromJson(old), std::runtime_error);
}

TEST(PartitionInfoLevelTests, ParsesNames) {
    EXPECT_EQ(partitionInfoLevelFromString("off"), PartitionInfoLevel::Off);
    EXPECT_EQ(partitionInfoLevelFromString("winner"), PartitionInfoLevel::Winner);
    EXPECT_EQ(partitionInfoLevelFromString("full"), PartitionInfoLevel::Full);
    EXPECT_THROW(partitionInfoLevelFromString("everything"), std::invalid_argument);
}

TEST(RasterizeTests, PaintsUnitsRelativeToTheChannelBoundingBox) {
    // Two partitions of a crop that starts at row 100, column 200.
    std::vector<CodingPartitionInfo> partitions;
    CodingPartitionInfo left({0, 0, 100, 200}, {9, 9, 4, 4}, 0);
    CodingUnitInfo whole({0, 0, 100, 200}, {9, 9, 4, 4});
    whole.setBits(1.0);
    left.appendCodingUnitInfo(whole);
    partitions.push_back(left);

    CodingPartitionInfo right({0, 0, 100, 204}, {9, 9, 4, 4}, 0);
    CodingUnitInfo topViews({0, 0, 100, 204}, {4, 9, 4, 4}); // view split: views t=0..3 only
    topViews.setBits(2.0);
    right.appendCodingUnitInfo(topViews);
    partitions.push_back(right);

    // A Cb partition that must be ignored for channel 0.
    CodingPartitionInfo chroma({0, 0, 100, 200}, {9, 9, 4, 4}, 1);
    chroma.appendCodingUnitInfo(whole);
    partitions.push_back(chroma);

    auto bits = [](const CodingUnitInfo& u) { return std::optional<double>(u.getBits()); };

    at::Tensor top = rasterize(partitions, 0, {0, 0}, bits);
    ASSERT_EQ(top.size(0), 4);
    ASSERT_EQ(top.size(1), 8);
    EXPECT_DOUBLE_EQ(top[0][0].item<double>(), 1.0);
    EXPECT_DOUBLE_EQ(top[3][7].item<double>(), 2.0);

    at::Tensor bottom = rasterize(partitions, 0, {8, 0}, bits);
    EXPECT_DOUBLE_EQ(bottom[0][0].item<double>(), 1.0);
    EXPECT_TRUE(std::isnan(bottom[0][4].item<double>())); // view 8 is not covered on the right
}

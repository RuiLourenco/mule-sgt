#include <gtest/gtest.h>

#include <cmath>
#include <LightField/Block4D_.h>

// Structure tensors of a Lambertian light field L(t,s,v,u) = f(u - d*s, v - d*t).
// Its gradient (axes t, s, v, u) is (-d*b, -d*a, b, a) with a = df/du and b = df/dv, so
// T = E[g g^T] is fixed by the texture moments E[a^2], E[b^2] and E[ab].

namespace {

using Estimator = Block4D_::StructureTensorEstimator;
const std::array<double, 2> WIDE_RANGE = {-100, 100};

at::Tensor lambertianTensor(double d, double aa, double bb, double ab) {
    // g = a * x + b * y with x = (0, -d, 0, 1), y = (-d, 0, 1, 0)
    at::Tensor x = torch::tensor({0.0, -d, 0.0, 1.0}, at::kDouble);
    at::Tensor y = torch::tensor({-d, 0.0, 1.0, 0.0}, at::kDouble);
    return aa * at::outer(x, x) + bb * at::outer(y, y) + ab * (at::outer(x, y) + at::outer(y, x));
}

double trueAngle(double d) { return std::atan(d) * 180.0 / PI; }

} // namespace

TEST(StructureTensorEstimatorTests, AllEstimatorsRecoverIsotropicTexture) {
    for (double d : {-3.0, -1.0, 0.0, 0.5, 2.2}) {
        at::Tensor T = lambertianTensor(d, 1.0, 0.8, 0.1);
        for (auto estimator : {Estimator::Pooled, Estimator::PerDirection, Estimator::Eigen4D}) {
            EXPECT_NEAR(Block4D_::angleFromStructureTensor(T, estimator, WIDE_RANGE), trueAngle(d), 1e-6)
                << "d = " << d << ", estimator " << static_cast<int>(estimator);
        }
    }
}

TEST(StructureTensorEstimatorTests, SingleDirectionEstimatorsReadTheirOwnEpi) {
    // Different disparities per direction isolate which EPI each estimator reads.
    at::Tensor horizontal = lambertianTensor(1.5, 1.0, 0.0, 0.0);
    at::Tensor vertical = lambertianTensor(-0.5, 0.0, 1.0, 0.0);
    at::Tensor T = horizontal + vertical;
    EXPECT_NEAR(Block4D_::angleFromStructureTensor(T, Estimator::EpiHorizontal, WIDE_RANGE), trueAngle(1.5), 1e-6);
    EXPECT_NEAR(Block4D_::angleFromStructureTensor(T, Estimator::EpiVertical, WIDE_RANGE), trueAngle(-0.5), 1e-6);
}

TEST(StructureTensorEstimatorTests, PooledIgnoresAnEmptyVerticalDirection) {
    // One-directional texture (varies along u only) plus a small isotropic noise floor:
    // the vertical EPIs carry only noise.
    for (double d : {-2.0, 0.7, 1.5}) {
        at::Tensor T = lambertianTensor(d, 1.0, 0.0, 0.0) + 1e-3 * at::eye(4, at::kDouble);
        double pooled = Block4D_::angleFromStructureTensor(T, Estimator::Pooled, WIDE_RANGE);
        double perDirection = Block4D_::angleFromStructureTensor(T, Estimator::PerDirection, WIDE_RANGE);
        EXPECT_NEAR(pooled, trueAngle(d), 0.5);
        // The noise-only vertical estimate is 0 here, which drags the mean halfway.
        EXPECT_NEAR(perDirection, trueAngle(d) / 2, 0.5);
    }
}

TEST(StructureTensorEstimatorTests, ClampsToTheAngleRangeAndHandlesZero) {
    std::array<double, 2> range = SgtSideInfo::angleRangeFromDispRange({-3.1, 3.5});
    at::Tensor steep = lambertianTensor(10.0, 1.0, 1.0, 0.0);
    EXPECT_DOUBLE_EQ(Block4D_::angleFromStructureTensor(steep, Estimator::Pooled, {-3.1, 3.5}), range[1]);
    at::Tensor zero = at::zeros({4, 4}, at::kDouble);
    for (auto estimator : {Estimator::Pooled, Estimator::PerDirection, Estimator::Eigen4D}) {
        EXPECT_DOUBLE_EQ(Block4D_::angleFromStructureTensor(zero, estimator, {-3.1, 3.5}), 0.0);
    }
}

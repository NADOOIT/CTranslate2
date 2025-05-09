#include "test_utils.h"
#include "ctranslate2/primitives.h"
#include "dispatch.h"

class PrimitiveTest : public ::testing::TestWithParam<ctranslate2::Device> {
};

TEST_P(PrimitiveTest, StridedFill) {
  const ctranslate2::Device device = GetParam();
  ctranslate2::StorageView x({3, 2}, float(0), device);
  ctranslate2::StorageView expected({3, 2}, std::vector<float>{1, 0, 1, 0, 1, 0}, device);
  DEVICE_DISPATCH(device, ctranslate2::primitives<D>::strided_fill(x.data<float>(), 1.f, 2, 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, IndexedFill) {
  const ctranslate2::Device device = GetParam();
  ctranslate2::StorageView x({6}, float(0), device);
  ctranslate2::StorageView ids({3}, std::vector<int32_t>{0, 2, 5}, device);
  ctranslate2::StorageView expected({6}, std::vector<float>{1, 0, 1, 0, 0, 1}, device);
  DEVICE_DISPATCH(device, ctranslate2::primitives<D>::indexed_fill(x.data<float>(), 1.f, ids.data<int32_t>(), 3));
  expect_storage_eq(x, expected);
}

TEST_P(PrimitiveTest, LogSumExp) {
  const ctranslate2::Device device = GetParam();
  ctranslate2::StorageView x({8}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2}, device);
  float result = 0;
  DEVICE_DISPATCH(device, result = ctranslate2::primitives<D>::logsumexp(x.data<float>(), x.size()));
  EXPECT_NEAR(result, 2.1908040046691895, 1e-6);
}

TEST_P(PrimitiveTest, PenalizePreviousTokens) {
  const ctranslate2::Device device = GetParam();
  const float penalty = 1.2f;
  ctranslate2::StorageView scores({2, 4}, std::vector<float>{0.6, 0.2, -1.2, 0.1, 0.3, 0.5, -1.3, 0.2});
  ctranslate2::StorageView previous_ids({2, 2}, std::vector<int32_t>{2, 2, 1, 2}, device);
  ctranslate2::StorageView previous_scores({2, 2}, std::vector<float>{-1.2, -1.2, 0.5, -1.3}, device);
  ctranslate2::StorageView expected = scores;
  expected.at<float>({0, 2}) *= penalty;
  expected.at<float>({1, 1}) /= penalty;
  expected.at<float>({1, 2}) *= penalty;
  scores = scores.to(device);
  DEVICE_DISPATCH(device, ctranslate2::primitives<D>::penalize_previous_tokens(scores.data<float>(),
                                                                  previous_scores.data<float>(),
                                                                  previous_ids.data<int32_t>(),
                                                                  penalty,
                                                                  scores.dim(0),
                                                                  previous_ids.dim(1),
                                                                  scores.dim(1)));
  expect_storage_eq(scores, expected);
}

INSTANTIATE_TEST_SUITE_P(CPU, PrimitiveTest, ::testing::Values(ctranslate2::Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PrimitiveTest, ::testing::Values(ctranslate2::Device::CUDA));
#endif

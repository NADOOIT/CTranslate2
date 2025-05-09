#include "test_utils.h"
#include "ctranslate2/storage_view.h"

TEST(StorageViewTest, ZeroDim) {
  ctranslate2::StorageView a({2, 0, 2});
  EXPECT_EQ(a.size(), 0);
  EXPECT_EQ(a.rank(), 3);
  EXPECT_EQ(a.dim(0), 2);
  EXPECT_EQ(a.dim(1), 0);
  EXPECT_EQ(a.dim(2), 2);

  ctranslate2::StorageView b(a);
  EXPECT_EQ(b.size(), 0);
  EXPECT_EQ(b.rank(), 3);
  EXPECT_EQ(b.dim(0), 2);
  EXPECT_EQ(b.dim(1), 0);
  EXPECT_EQ(b.dim(2), 2);
}

TEST(StorageViewTest, BoolOperator) {
  ctranslate2::StorageView a;
  EXPECT_FALSE(bool(a));
  a.resize({4});
  EXPECT_TRUE(bool(a));
}

TEST(StorageViewTest, Reshape) {
  ctranslate2::StorageView a(ctranslate2::Shape{16});
  assert_vector_eq(a.shape(), ctranslate2::Shape{16});
  a.reshape({4, 4});
  assert_vector_eq(a.shape(), ctranslate2::Shape{4, 4});
  a.reshape({2, -1});
  assert_vector_eq(a.shape(), ctranslate2::Shape{2, 8});
  a.reshape({-1, 1});
  assert_vector_eq(a.shape(), ctranslate2::Shape{16, 1});
  a.reshape({2, -1, 2});
  assert_vector_eq(a.shape(), ctranslate2::Shape{2, 4, 2});
  a.reshape({-1});
  assert_vector_eq(a.shape(), ctranslate2::Shape{16});
}

TEST(StorageViewTest, ExpandDimsAndSqueeze) {
  {
    ctranslate2::StorageView a(ctranslate2::Shape{4});
    a.expand_dims(0);
    assert_vector_eq(a.shape(), ctranslate2::Shape{1, 4});
    a.expand_dims(-1);
    assert_vector_eq(a.shape(), ctranslate2::Shape{1, 4, 1});
    a.squeeze(0);
    assert_vector_eq(a.shape(), ctranslate2::Shape{4, 1});
    a.squeeze(1);
    assert_vector_eq(a.shape(), ctranslate2::Shape{4});
  }

  {
    ctranslate2::StorageView a(ctranslate2::Shape{4, 2});
    a.expand_dims(1);
    assert_vector_eq(a.shape(), ctranslate2::Shape{4, 1, 2});
    a.expand_dims(3);
    assert_vector_eq(a.shape(), ctranslate2::Shape{4, 1, 2, 1});
  }
}

class StorageViewDeviceTest : public ::testing::TestWithParam<ctranslate2::Device> {
};

TEST_P(StorageViewDeviceTest, HalfConversion) {
  const ctranslate2::Device device = GetParam();
  const ctranslate2::StorageView a({4}, std::vector<float>{1, 2, 3, 4}, device);
  EXPECT_EQ(a.reserved_memory(), 4 * 4);
  const ctranslate2::StorageView b = a.to_float16();
  EXPECT_EQ(b.dtype(), ctranslate2::DataType::FLOAT16);
  EXPECT_EQ(b.reserved_memory(), 4 * 2);
  expect_storage_eq(b.to_float32(), a);
}

INSTANTIATE_TEST_SUITE_P(CPU, StorageViewDeviceTest, ::testing::Values(ctranslate2::Device::CPU));
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, StorageViewDeviceTest, ::testing::Values(ctranslate2::Device::CUDA));
#endif

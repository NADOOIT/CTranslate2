#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "metal/metal_device.h"
#include "metal/metal_allocator.h"

using namespace ctranslate2;

class MetalMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        device = new MetalDevice();
        allocator = new MetalAllocator(*device);
    }

    void TearDown() override {
        delete allocator;
        delete device;
    }

    MetalDevice* device;
    MetalAllocator* allocator;
};

TEST_F(MetalMemoryTest, AllocateAndFree) {
    const size_t size = 1024;
    void* ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);
    allocator->free(ptr);
}

TEST_F(MetalMemoryTest, AllocateMultiple) {
    const size_t size1 = 1024;
    const size_t size2 = 2048;
    
    void* ptr1 = allocator->allocate(size1);
    void* ptr2 = allocator->allocate(size2);
    
    ASSERT_NE(ptr1, nullptr);
    ASSERT_NE(ptr2, nullptr);
    ASSERT_NE(ptr1, ptr2);
    
    allocator->free(ptr1);
    allocator->free(ptr2);
}

TEST_F(MetalMemoryTest, AllocateZeroSize) {
    void* ptr = allocator->allocate(0);
    ASSERT_EQ(ptr, nullptr);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

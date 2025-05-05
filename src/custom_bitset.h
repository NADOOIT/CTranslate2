#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ctranslate2 {

template <size_t N>
class Bitset {
private:
    static constexpr size_t kBitsPerWord = 64;
    static constexpr size_t kNumWords = (N + kBitsPerWord - 1) / kBitsPerWord;
    std::array<std::uint64_t, kNumWords> _data{};

    constexpr size_t word_index(size_t pos) const { return pos / kBitsPerWord; }
    constexpr size_t bit_index(size_t pos) const { return pos % kBitsPerWord; }

public:
    constexpr Bitset() noexcept = default;

    bool test(size_t pos) const {
        if (pos >= N) {
            throw std::out_of_range("Bitset index out of range");
        }
        return (_data[word_index(pos)] & (std::uint64_t(1) << bit_index(pos))) != 0;
    }

    bool operator[](size_t pos) const {
        return (_data[word_index(pos)] & (std::uint64_t(1) << bit_index(pos))) != 0;
    }

    void reset() noexcept {
        _data.fill(0);
    }
};

} // namespace ctranslate2

#ifndef CTRANSLATE2_STD_NAMESPACE_FIX_H
#define CTRANSLATE2_STD_NAMESPACE_FIX_H

// Disable library warnings and annotations
#define _LIBCPP_DISABLE_AVAILABILITY 1
#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS 1
#define _LIBCPP_DISABLE_VISIBILITY_ANNOTATIONS 1

// Comprehensive standard library header inclusion
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <locale>
#include <codecvt>

// Ensure global namespace types are used
#define ios_base ::std::ios_base
#define streamsize ::std::streamsize
#define basic_streambuf ::std::basic_streambuf
#define basic_filebuf ::std::basic_filebuf

// Namespace resolution
namespace ctranslate2 {
    // Explicitly use fully qualified types
    template <typename T>
    using string = ::std::basic_string<T, ::std::char_traits<T>, ::std::allocator<T>>;

    template <typename T, typename Allocator = ::std::allocator<T>>
    using vector = ::std::vector<T, Allocator>;

    using size_t = ::std::size_t;
    using uint8_t = ::std::uint8_t;
    using uint32_t = ::std::uint32_t;
}

// Debugging macro to help track type resolution
#define CTRANSLATE2_STRINGIFY(x) #x
#define CTRANSLATE2_TOSTRING(x) CTRANSLATE2_STRINGIFY(x)

#endif // CTRANSLATE2_STD_NAMESPACE_FIX_H

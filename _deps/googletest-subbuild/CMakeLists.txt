# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 4.0.1)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(googletest-populate NONE)



include(ExternalProject)
ExternalProject_Add(googletest-populate
                     "UPDATE_DISCONNECTED" "False" "DOWNLOAD_EXTRACT_TIMESTAMP" "TRUE" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/google/googletest/archive/refs/tags/v1.12.1.tar.gz" "URL_HASH" "SHA256=81964fe918a76fffa0b14183b5aba5f31abc9c60ab7b9b5e9f6e4cddeb9a1d90" "PATCH_COMMAND" "/opt/homebrew/bin/cmake" "-E" "remove_directory" "/Users/christophbackhaus/Documents/GitHub/CTranslate2/tests/third_party/_deps/googletest-src/googlemock"
                    SOURCE_DIR          "/Users/christophbackhaus/Documents/GitHub/CTranslate2/_deps/googletest-src"
                    BINARY_DIR          "/Users/christophbackhaus/Documents/GitHub/CTranslate2/_deps/googletest-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)



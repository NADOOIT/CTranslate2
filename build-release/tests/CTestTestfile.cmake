# CMake generated Testfile for 
# Source directory: /Users/christophbackhaus/Documents/GitHub/CTranslate2/tests
# Build directory: /Users/christophbackhaus/Documents/GitHub/CTranslate2/build-release/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[ctranslate2_test]=] "/Users/christophbackhaus/Documents/GitHub/CTranslate2/build-release/tests/ctranslate2_test" "/Users/christophbackhaus/Documents/GitHub/CTranslate2/tests/data")
set_tests_properties([=[ctranslate2_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/christophbackhaus/Documents/GitHub/CTranslate2/tests/CMakeLists.txt;35;add_test;/Users/christophbackhaus/Documents/GitHub/CTranslate2/tests/CMakeLists.txt;0;")
subdirs("googletest")

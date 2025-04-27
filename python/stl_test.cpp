#include <bitset>
#include <fstream>
#include <iostream>

int main() {
    std::bitset<8> b(42);
    std::cout << b << std::endl;
    std::ofstream ofs("test.txt");
    ofs << "Hello, STL!" << std::endl;
    return 0;
}

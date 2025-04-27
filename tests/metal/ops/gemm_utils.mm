#include "gemm_utils.h"
#include <random>
#include <fstream>
#include <ctime>
#include <iomanip>

void fill_random(std::vector<float>& v, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1, 1);
    for (auto& x : v) x = dist(gen);
}

bool file_exists(const std::string& filename) {
    std::ifstream f(filename);
    return f.good();
}

void write_csv_header(std::ofstream& csv) {
    csv << "timestamp,commit,operator,mode,device,size,batch,avg_ms\n";
}

std::string current_time_string() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%FT%T%z", std::localtime(&t));
    return std::string(buf);
}

double max_diff(const float* a, const float* b, size_t size) {
    double maxd = 0.0;
    for (size_t i = 0; i < size; ++i)
        maxd = std::max(maxd, static_cast<double>(std::abs(a[i] - b[i])));
    return maxd;
}

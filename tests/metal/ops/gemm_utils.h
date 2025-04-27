#pragma once
#include <vector>
#include <string>
#include <fstream>

void fill_random(std::vector<float>& v, int seed = 42);
bool file_exists(const std::string& filename);
void write_csv_header(std::ofstream& csv);
std::string current_time_string();
double max_diff(const float* a, const float* b, size_t size);

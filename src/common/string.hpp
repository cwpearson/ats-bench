#pragma once

#include <string>
#include <vector>

template <typename T>
std::string to_string(const std::vector<T> vec, const std::string sep = ",") {
  std::string result;
  for (size_t i = 0; i < vec.size(); ++i) {
    result += fmt::format("{:.2e}", vec[i]);
    if (i + 1 < vec.size()) {
      result += sep;
    }
  }
  return result;
}
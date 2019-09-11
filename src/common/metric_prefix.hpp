#pragma once

// parse numbers with metric prefix symbols

#include <algorithm>
#include <iostream>
#include <string>

/*!

Parse a string like "1e10KB" to 1e13

 */
int parse_double(double &d, const std::string &s) {
  size_t idx = 0;
  double result = std::stod(s, &idx);

  // start looking for prefix at idx
  const std::string symbol = s.substr(idx);
  if (symbol.empty()) {
    // okay
  } else if (std::all_of(symbol.begin(), symbol.end(), isspace)) {
    // okay
  } else if ("G" == symbol || "GB" == symbol) {
    result *= 1e9;
  } else if ("Gi" == symbol || "GiB" == symbol) {
    result *= 1024.0 * 1024.0 * 1024.0;
  } else if ("M" == symbol || "MB" == symbol) {
    result *= 1e6;
  } else if ("Mi" == symbol || "MiB" == symbol) {
    result *= 1024.0 * 1024.0;
  } else if ("K" == symbol || "KB" == symbol) {
    result *= 1e3;
  } else if ("Ki" == symbol || "KiB" == symbol) {
    result *= 1024;
  } else if ("B" == symbol) {
    // okay
  } else {
    return 1;
  }
  d = result;
  return 0;
}

int parse_u64(uint64_t &u64, const std::string &s) {
  double d;
  // int ret = parse_double(d, s);
  if (int ret = parse_double(d, s)) {
    return ret;
  }
  if (d < 0) {
    return 2;
  }
  u64 = d;
  return 0;
}
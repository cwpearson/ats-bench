// disable turbo boost
// https://wiki.archlinux.org/index.php/CPU_frequency_scaling
// for intel_pstate (INTEL?)
// echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
// acpi-cpufreq (AMD?)
// echo 0 > /sys/devices/system/cpu/cpufreq/boost

// https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux

/*
for i in /sys/devices/system/cpu/cpuX/cpufreq/scaling_governor
do
  echo performance > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor
done
*/

#include <cassert>
#include <fstream>
#include <map>

#include <unistd.h>

#include "logger.hpp"

std::string get_governor(const int cpu) {
  std::string path("/sys/devices/system/cpu/cpu");
  path += std::to_string(cpu);
  path += "/cpufreq/scaling_governor";
  std::ifstream ifs(path, std::ifstream::in);
  std::string result;
  std::getline(ifs, result);
  return result;
}

int set_governor(const int cpu, const std::string &governor) {
  std::string path("/sys/devices/system/cpu/cpu");
  path += std::to_string(cpu);
  path += "/cpufreq/scaling_governor";
  SPDLOG_LOGGER_DEBUG(logger::console(), "writing '{}' to {}", governor, path);
  std::ofstream ofs(path, std::ofstream::out);
  ofs << governor;
  ofs.close();
  if (ofs.fail()) {
    SPDLOG_LOGGER_DEBUG(logger::console(), "fail after write to {}", path);
    return 1;
  }
  return 0;
}

int set_governor_performance(const int cpu) {
  return set_governor(cpu, "performance");
}

bool has_intel_pstate_no_turbo() {
  return bool(std::ifstream("/sys/devices/system/cpu/intel_pstate/no_turbo"));
}

bool has_acpi_cpufreq_boost() {
  return bool(std::ifstream("/sys/devices/system/cpu/cpufreq/boost"));
}

int write_intel_pstate_no_turbo(const std::string &s) {
  assert(has_intel_pstate_no_turbo());
  std::string path("/sys/devices/system/cpu/intel_pstate/no_turbo");
  SPDLOG_LOGGER_DEBUG(logger::console(), "writing {} to {}", s, path);
  std::ofstream ofs(path, std::ofstream::out);
  ofs << s;
  ofs.close();
  if (ofs.fail()) {
    SPDLOG_LOGGER_DEBUG(logger::console(), "error writing {} to {}", s, path);
    return 1;
  }
  return 0;
}

std::string read_intel_pstate_no_turbo() {
  assert(has_intel_pstate_no_turbo());
  std::string path("/sys/devices/system/cpu/intel_pstate/no_turbo");
  SPDLOG_LOGGER_TRACE(logger::console(), "reading {}", path);
  std::ifstream ifs(path, std::ifstream::in);
  std::string result;
  std::getline(ifs, result);
  return result;
}

int write_acpi_cpufreq_boost(const std::string &s) {
  assert(has_acpi_cpufreq_boost());
  std::string path("/sys/devices/system/cpu/cpufreq/boost");
  SPDLOG_LOGGER_TRACE(logger::console(), "writing to {}", path);
  std::ofstream ofs(path, std::ofstream::out);
  ofs << s;
  ofs.close();
  if (ofs.fail()) {
    SPDLOG_LOGGER_TRACE(logger::console(), "error writing to {}", path);
    return 1;
  }
  return 0;
}

std::string readAcpiCpufreqBoost() {
  assert(has_intel_pstate_no_turbo());
  std::string path("/sys/devices/system/cpu/cpufreq/boost");
  SPDLOG_LOGGER_TRACE(logger::console(), "reading {}", path);
  std::ifstream ifs(path, std::ifstream::in);
  std::string result;
  std::getline(ifs, result);
  return result;
}

/*! Enable CPU boost

    \return 0 if success, 1 otherwise
*/
int enable_boost() {
  if (has_intel_pstate_no_turbo()) {
    write_intel_pstate_no_turbo("0");
    return 0;
  } else if (has_acpi_cpufreq_boost()) {
    write_acpi_cpufreq_boost("1");
    return 0;
  }
  LOG(error, "unsupported system");
  return 1;
}

/*! return 0 if success
 */
bool disable_boost() {
  if (has_intel_pstate_no_turbo()) {
    LOG(debug, "disable boost with intel_pstate");
    return write_intel_pstate_no_turbo("1");
  } else if (has_acpi_cpufreq_boost()) {
    LOG(debug, "disable boost with acpi cpufreq");
    return write_acpi_cpufreq_boost("0");
  }
  LOG(error, "unsupported system");
  return 1;
}

bool boost_enabled() {
  if (has_intel_pstate_no_turbo()) {
    return "0" == read_intel_pstate_no_turbo();
  } else if (has_acpi_cpufreq_boost()) {
    return "1" == readAcpiCpufreqBoost();
  } else {
    LOG(error, "unsupported system (boost)");
    return false;
  }
}

int nProcessorsOnln() { return sysconf(_SC_NPROCESSORS_ONLN); }

struct WithPerformance {
  std::map<int, std::string> governors;
  std::map<int, bool> needsRestore;
  bool strict_;

  WithPerformance(bool strict = false) : strict_(strict) {
    // upon construction, save the current governor to be stored at destuction
    for (int cpu = 0; cpu < nProcessorsOnln(); ++cpu) {
      const std::string current = get_governor(cpu);
      governors[cpu] = current;
      if ("performance" == current) {
        LOG(debug, "cpu {} already set to performance", cpu);
        needsRestore[cpu] = false;
      } else {
        if (set_governor_performance(cpu)) {
          if (strict_) {
            LOG(critical, "couldn't set governor for cpu {}", cpu);
            exit(EXIT_FAILURE);
          } else {
            LOG(warn, "couldn't set governor for cpu {}", cpu);
            needsRestore[cpu] = false;
          }
        } else {
          needsRestore[cpu] = true;
        }
      }
    }
  }

  ~WithPerformance() {
    // restore all governors we modified
    for (auto p : needsRestore) {
      auto cpu = p.first;
      auto need = p.second;
      if (need) {
        if (set_governor(cpu, governors[cpu])) {
          if (strict_) {
            LOG(critical, "couldn't restore governor for cpu {}", cpu);
            exit(EXIT_FAILURE);
          }
          LOG(error, "couldn't restore governor for cpu {}", cpu);
        }
      }
    }
  }
};

struct WithoutBoost {
  bool enabled;
  bool restore;
  bool strict_;

  WithoutBoost(bool strict = false) : strict_(strict) {
    // upon construction, save the current governor to be stored at destuction
    enabled = boost_enabled();
    if (enabled) {
      if (disable_boost()) {
        if (strict_) {
          LOG(critical, "unable to disable boost");
          exit(EXIT_FAILURE);
        } else {
          LOG(warn, "unable to disable boost");
        }
        restore = false;
      } else {
        restore = true;
      }
    } else {
      LOG(debug, "CPU boost already disabled");
      restore = false;
    }
  }

  ~WithoutBoost() {
    if (restore) {
      if (enabled) {
        if (enable_boost()) {
          if (strict_) {
            LOG(critical, "unable to re-enable boost");
          } else {
            LOG(error, "unable to re-enable boost");
          }
        }
      }
    }
  }
};
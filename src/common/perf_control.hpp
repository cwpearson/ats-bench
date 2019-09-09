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

#include <fstream>
#include <map>
#include <unistd.h>

#include "logger.hpp"

std::string getGovernor(const int cpu) {
  std::string path("/sys/devices/system/cpu/cpu");
  path += std::to_string(cpu);
  path += "/cpufreq/scaling_governor";
  std::ifstream ifs(path, std::ifstream::in);
  std::string result;
  std::getline(ifs, result);
  return result;
}

int setGovernor(const int cpu, const std::string &governor) {
  std::string path("/sys/devices/system/cpu/cpu");
  path += std::to_string(cpu);
  path += "/cpufreq/scaling_governor";
  SPDLOG_DEBUG(logger::console(), "writing to {}", path);
  std::ofstream ofs(path, std::ofstream::out);
  ofs << governor;

  ofs.close();

  if (ofs.fail()) {
    SPDLOG_DEBUG(logger::console(), "error writing to {}", path);
    return 1;
  }

  return 0;
}

int setGovernorPerformance(const int cpu) {
  return setGovernor(cpu, "performance");
}

int nProcessorsOnln() { return sysconf(_SC_NPROCESSORS_ONLN); }

struct WithPerformance {
  std::map<int, std::string> governors;
  std::map<int, bool> needsRestore;

  WithPerformance() {

    // upon construction, save the current governor to be stored at destuction
    for (int cpu = 0; cpu < nProcessorsOnln(); ++cpu) {
      governors[cpu] = getGovernor(cpu);
      if (setGovernorPerformance(cpu)) {
        needsRestore[cpu] = false;
      } else {
        needsRestore[cpu] = true;
      }
    }
  }

  ~WithPerformance() {
    // restore all governors we modified
    for (auto p : needsRestore) {
      auto cpu = p.first;
      auto need = p.second;
      if (need) {
        setGovernor(cpu, governors[cpu]);
      }
    }
  }
};
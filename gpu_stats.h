#ifndef GPU_STATS_H
#define GPU_STATS_H

#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <wordexp.h>
#include <mutex>
#include <thread>
#include <condition_variable>

enum jetson_version {TK1, TX1, TX2};

struct tegrastats {
  int mem_usage;
  int mem_max;

  std::vector<int> cpu_usage;
  std::vector<int> cpu_freq;

  int gpu_usage;
  int gpu_freq;

  jetson_version version;
};

typedef std::vector<std::string> vec_str;

vec_str tokenize(const std::string &, const char);

bool file_exists(const std::string &);

const int STATS_BUFFER_SIZE = 256;

const std::string TEGRASTATS_PATH     = "/home/nvidia/tegrastats";
void stop_gpu_stats();
int get_gpu_usage();
int get_gpu_cores();
inline int _ConvertSMVer2Cores(int major, int minor);
void read_tegrastats();
tegrastats parse_smistats(const char *);
tegrastats parse_tegrastats(const char *);

void get_cpu_stats_tx1(tegrastats &, const std::string &);
void get_cpu_stats_tx2(tegrastats &, const std::string &);
void get_gpu_stats(tegrastats &, const std::string &);
void get_mem_stats(tegrastats &, const std::string &);
void print_stats(tegrastats &);
#endif // GTOP_HH_

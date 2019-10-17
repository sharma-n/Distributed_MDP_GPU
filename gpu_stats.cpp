#include "gpu_stats.h"
#include <cuda_runtime.h>
#include <cuda.h>

std::mutex m;
std::condition_variable cv;
tegrastats t_stats;
float ema_use = 0;
bool processed = false;
bool ready     = false;
bool finished  = false;

vec_str tokenize(const std::string & str, const char delim) {
  vec_str tokens;
  std::stringstream ss(str);

  while (ss.good()) {
    std::string substr;
    getline(ss, substr, delim);
    tokens.push_back(substr);
  }

  return tokens;
}

bool file_exists(const std::string & name) {
  struct stat buffer;
  std::string full_name;
  wordexp_t expanded_name;

  wordexp(name.c_str(), &expanded_name, 0);
  full_name = expanded_name.we_wordv[0];
  wordfree(&expanded_name);

  return (stat(full_name.c_str(), &buffer) == 0);
}

void stop_gpu_stats(){
  // std::lock_guard<std::mutex> lk(m);
  finished = true;
}

int get_gpu_usage(){
  {
    std::lock_guard<std::mutex> lk(m);
    ready = true;
  }
  cv.notify_one();
  std::unique_lock<std::mutex> lk(m);
  cv.wait(lk, []{ return processed; });
  //processed = false;
  int gpu_use = (int)ema_use;
  lk.unlock();
  return gpu_use;
}

void read_tegrastats() {
  std::array<char, STATS_BUFFER_SIZE> buffer;
  if (file_exists(TEGRASTATS_PATH)) {   //TX2 found

    std::shared_ptr<FILE> pipe(popen(TEGRASTATS_PATH.c_str(), "r"), pclose);

    if (!pipe)
    throw std::runtime_error ("popen() failed!");

    while (!feof(pipe.get())) {

        if (fgets(buffer.data(), STATS_BUFFER_SIZE, pipe.get()) != NULL) {
          std::unique_lock<std::mutex> lk(m);

          //terminate reading tegrastats
          if (finished) {
            lk.unlock();
            break;
          }

          cv.wait(lk, []{ return ready; });
          ready = false;
          t_stats = parse_tegrastats(buffer.data());
          processed = true;
          lk.unlock();
          cv.notify_one();
        }
    }
  } else {    // Assume nvidia-smi functional
      std::string command = "nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1";
      std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);

      if (!pipe)
      throw std::runtime_error ("popen() failed!");

      if (fgets(buffer.data(), STATS_BUFFER_SIZE, pipe.get()) != NULL) {
        while (!feof(pipe.get())) {
          if (fgets(buffer.data(), STATS_BUFFER_SIZE, pipe.get()) != NULL) {
            // std::unique_lock<std::mutex> lk(m);

            //terminate reading tegrastats
            if (finished) {
              // lk.unlock();
              break;
            }

            // cv.wait(lk, []{ return ready; });
            // ready = false;
            t_stats = parse_smistats(buffer.data());
            ema_use = 0.1*t_stats.gpu_usage + 0.9*ema_use;
            processed = true;
            // lk.unlock();
            // cv.notify_one();
          }
        }
      }
  }


}

tegrastats parse_smistats(const char * buffer){
  tegrastats ts;
  auto stats = tokenize(buffer, ' ');
  ts.gpu_usage = std::stoi(stats.at(0));
  return ts;
}

tegrastats parse_tegrastats(const char * buffer) {
  tegrastats ts;
  auto stats = tokenize(buffer, ' ');

  if (stats.size() >= 15)
    ts.version = TX1;
  else
    ts.version = TX2;

  get_mem_stats(ts, stats.at(1));
  get_cpu_stats_tx2(ts, stats.at(5));
  get_gpu_stats(ts, stats.at(9));
  return ts;
}

void get_cpu_stats_tx1(tegrastats & ts, const std::string & str) {
  auto cpu_stats = tokenize(str, '@');
  auto cpu_usage_all = cpu_stats.at(0);
  ts.cpu_freq.push_back(std::stoi(cpu_stats.at(1)));
  auto cpu_usage = tokenize(cpu_usage_all.substr(1, cpu_usage_all.size()-2), ',');

  for (const auto & u: cpu_usage) {
    if (u != "off")
      ts.cpu_usage.push_back(std::stoi(u.substr(0, u.size()-1)));
    else
      ts.cpu_usage.push_back(0);
  }
}

void get_cpu_stats_tx2(tegrastats & ts, const std::string & str) {
  const auto cpu_stats = tokenize(str.substr(1, str.size()-1), ',');
  const auto at = std::string("@");

  for (const auto & u: cpu_stats) {
    if (u != "off") {
      std::size_t found = at.find(u);
      if (found == std::string::npos) {
        auto cpu_info = tokenize(u, at.c_str()[0]);
        ts.cpu_usage.push_back(std::stoi(cpu_info.at(0).substr(0, cpu_info.at(0).size()-1)));
        ts.cpu_freq.push_back(std::stoi(cpu_info.at(1)));
      }
    }
    else {
      ts.cpu_usage.push_back(0);
      ts.cpu_freq.push_back(0);
    }
  }
}

int get_gpu_cores(){
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  return _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
      deviceProp.multiProcessorCount;
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}

void get_gpu_stats(tegrastats & ts, const std::string & str) {
  const auto gpu_stats = tokenize(str, '@');
  const auto gpu_usage = gpu_stats.at(0);
  ts.gpu_usage = std::stoi(gpu_usage.substr(0, gpu_usage.size()-1));
  ts.gpu_freq = std::stoi(gpu_stats.at(1));
}

void get_mem_stats(tegrastats & ts, const std::string & str) {
  const auto mem_stats = tokenize(str, '/');
  const auto mem_max = mem_stats.at(1);
  ts.mem_usage = std::stoi(mem_stats.at(0));
  ts.mem_max = std::stoi(mem_max.substr(0, mem_max.size()-2));
}

void print_stats(tegrastats & ts){
  std::cout << "Memory Usage: " << ts.mem_usage << '\n';
  std::cout << "Maximum Memory: " << ts.mem_max << '\n';
  std::cout << "CPU Usage: ";
  for(std::vector<int>::const_iterator i = ts.cpu_usage.begin(); i!=ts.cpu_usage.end(); ++i){
    std::cout << *i << ' ';
  }
  std::cout << "\nCPU Frequency: ";
  for(std::vector<int>::const_iterator i = ts.cpu_freq.begin(); i!=ts.cpu_freq.end(); ++i){
    std::cout << *i << ' ';
  }
  std::cout << "\nGPU Usage: " << ts.gpu_usage << '\n';
  std::cout << "GPU Frequency: " << ts.gpu_freq << '\n';
}

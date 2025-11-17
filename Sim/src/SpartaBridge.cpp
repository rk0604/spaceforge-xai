#include "SpartaBridge.hpp"
#include "library.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <unistd.h>

namespace fs = std::filesystem;

static void set_gpu_env_defaults() {
  // Environment defaults (still fine even if we stay on CPU)
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  setenv("OMP_NUM_THREADS",      "1", 1);
  unsetenv("DISPLAY");
  unsetenv("XAUTHORITY");
}

// Very simple whitespace splitter: assumes no quoted args.
static std::vector<std::string> split_args(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream iss(s);
  std::string token;
  while (iss >> token) {
    out.push_back(token);
  }
  return out;
}

SpartaBridge::SpartaBridge(MPI_Comm comm) : comm_(comm) {
  set_gpu_env_defaults();

  // Base argv: log to log.capi
  std::vector<std::string> args_str = {
    "sparta",          // argv[0] placeholder
    "-log", "log.capi"
  };

  // If run.sh exported SPARTA_EXTRA_ARGS, append them
  if (const char* extra = std::getenv("SPARTA_EXTRA_ARGS")) {
    if (*extra) {
      auto extras = split_args(std::string(extra));
      args_str.insert(args_str.end(), extras.begin(), extras.end());
    }
  }

  // Build argv[]
  std::vector<char*> args;
  args.reserve(args_str.size() + 1);
  for (auto &s : args_str) {
    args.push_back(const_cast<char*>(s.c_str()));
  }
  args.push_back(nullptr);

  int   argc = static_cast<int>(args_str.size());
  char** argv = args.data();

  sparta_open(argc, argv, comm_, &spa_);
  if (!spa_) {
    int r = 0;
    if (comm_ != MPI_COMM_NULL) {
      MPI_Comm_rank(comm_, &r);
    }
    if (r == 0) {
      std::cerr << "Failed to open SPARTA library\n";
    }
    throw std::runtime_error("sparta_open failed");
  }
}

void SpartaBridge::runDeck(const std::string& deck,
                           const std::string& subdir) {
  // Allow both relative *and* absolute input_subdir.
  // - If subdir is relative: interpret as PROJECT_SOURCE_DIR / subdir
  // - If subdir is absolute: use it as-is (this matches run.sh behavior)
  fs::path input_dir = fs::path(subdir);
  if (!input_dir.is_absolute()) {
    input_dir = fs::path(PROJECT_SOURCE_DIR) / input_dir;
  }

  // Make species/surf paths like "data/o.species" resolve relative to input/
  std::error_code ec;
  fs::current_path(input_dir, ec);
  if (ec) {
    std::cerr << "SpartaBridge: chdir to " << input_dir
              << " failed: " << ec.message() << "\n";
    throw std::runtime_error("SpartaBridge: cannot change to input directory");
  }

  // Run the deck file
  sparta_file(spa_, const_cast<char*>(deck.c_str()));
}

void SpartaBridge::command(const char* cmd) {
  if (!spa_) {
    throw std::runtime_error("SpartaBridge::command: SPARTA not open");
  }
  sparta_command(spa_, const_cast<char*>(cmd));
}

void SpartaBridge::runSteps(int n) {
  if (n <= 0) return;
  std::string s = "run " + std::to_string(n);
  command(s.c_str());
}

void SpartaBridge::clear() {
  command("clear");
}

SpartaBridge::~SpartaBridge() {
  if (spa_) {
    sparta_close(spa_);
  }
}

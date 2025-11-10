#include "SpartaBridge.hpp"
#include "library.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>

namespace fs = std::filesystem;

static void set_gpu_env_defaults() {
  // Match your working CLI environment
  setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  setenv("OMP_NUM_THREADS",      "1", 1);
  unsetenv("DISPLAY");
  unsetenv("XAUTHORITY");
}

SpartaBridge::SpartaBridge(MPI_Comm comm) : comm_(comm) {
  set_gpu_env_defaults();

  // ---- IMPORTANT: each flag is its own argv token ----
  // Equivalent to: spa_ -k on g 1 -sf kk -log log.capi
  std::vector<std::string> args_str = {
    "sparta",        // argv[0] (program name placeholder)
    "-k", "on",      // turn Kokkos on
    "g", "1",        // request 1 GPU per node
    "-sf", "kk",     // prefer Kokkos styles by default
    "-log", "log.capi"
  };
  std::vector<char*> args; args.reserve(args_str.size() + 1);
  for (auto& s : args_str) args.push_back(const_cast<char*>(s.c_str()));
  args.push_back(nullptr);

  int argc = static_cast<int>(args_str.size());
  char** argv = args.data();

  sparta_open(argc, argv, comm_, &spa_);
  if (!spa_) {
    int r = 0; if (comm_ != MPI_COMM_NULL) MPI_Comm_rank(comm_, &r);
    if (r == 0) std::cerr << "Failed to open SPARTA library\n";
    throw std::runtime_error("sparta_open failed");
  }
}

void SpartaBridge::runDeck(const std::string& deck, const std::string& subdir) {
  fs::path input_dir = fs::path(PROJECT_SOURCE_DIR) / subdir;

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
  if (!spa_) throw std::runtime_error("SpartaBridge::command: SPARTA not open");
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
  if (spa_) sparta_close(spa_);
}

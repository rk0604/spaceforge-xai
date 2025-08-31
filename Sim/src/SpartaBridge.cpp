#include "SpartaBridge.hpp"
#include "library.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

SpartaBridge::SpartaBridge(MPI_Comm comm) : comm_(comm) {
  static char prog[] = "sparta";
  static char opt1[] = "-log";
  static char opt2[] = "log.capi";
  char* argv[] = { prog, opt1, opt2 };
  int   argc   = 3;

  sparta_open(argc, argv, comm_, &spa_);
  if (!spa_) {
    int r = 0; if (comm_ != MPI_COMM_NULL) MPI_Comm_rank(comm_, &r);
    if (r == 0) std::cerr << "Failed to open SPARTA library\n";
    throw std::runtime_error("sparta_open failed");
  }
}

void SpartaBridge::runDeck(const std::string& deck, const std::string& subdir) {
  fs::path input_dir = fs::path(PROJECT_SOURCE_DIR) / subdir;
  fs::current_path(input_dir);                // relative includes resolve here
  sparta_file(spa_, const_cast<char*>(deck.c_str()));
}

// --- NEW: wrappers used by WakeChamber ---
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
// -----------------------------------------

SpartaBridge::~SpartaBridge() {
  if (spa_) sparta_close(spa_);
}

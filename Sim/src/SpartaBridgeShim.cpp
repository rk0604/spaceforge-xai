#include "SpartaBridge.hpp"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <mpi.h>

namespace {

std::string envOr(const char* key, const std::string& def) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : def;
}

std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (b.empty()) return a;
  if (a.back() == '/' || a.back() == '\\') return a + b;
  return a + "/" + b;
}

bool isAbsolutePath(const std::string& p) {
  return !p.empty() && (p[0] == '/' || p.rfind("://", 0) == 0);
}

std::string dirname(const std::string& p) {
  std::string::size_type pos = p.find_last_of('/');
  if (pos == std::string::npos) return ".";
  if (pos == 0) return "/";
  return p.substr(0, pos);
}

std::string basename(const std::string& p) {
  std::string::size_type pos = p.find_last_of('/');
  if (pos == std::string::npos) return p;
  return p.substr(pos + 1);
}

} // namespace

SpartaBridge::SpartaBridge(MPI_Comm comm) : spa_(nullptr), comm_(comm) {}
SpartaBridge::~SpartaBridge() = default;

void SpartaBridge::runDeck(const std::string& deck_basename,
                           const std::string& input_subdir) {
  int rank = 0;
  MPI_Comm_rank(comm_, &rank);
  if (rank != 0) return;

  // Resolve absolute deck path
  std::string deck_path = deck_basename;
  const bool has_slash = deck_basename.find('/') != std::string::npos;
  if (!isAbsolutePath(deck_path)) {
    if (!has_slash && !input_subdir.empty()) {
      deck_path = joinPath(input_subdir, deck_basename);
    }
  }

  // Only launch wake deck
  static bool warned_nonwake = false;
  if (deck_path.find("in.wake") == std::string::npos) {
    if (!warned_nonwake) {
      std::cerr << "[SpartaBridgeShim] Skipping non-wake deck (" << deck_path << ")\n";
      warned_nonwake = true;
    }
    return;
  }

  static std::set<std::string> launched;
  if (launched.count(deck_path)) {
    std::cerr << "[SpartaBridgeShim] Deck already launched, skipping: "
              << deck_path << "\n";
    return;
  }
  launched.insert(deck_path);

  const std::string home = envOr("HOME", "");
  const std::string sparta_exe =
      envOr("SPARTA_EXE", home + "/opt/sparta/build-gpu/src/spa_");
const std::string extra_args =
    envOr("SPARTA_EXTRA_ARGS",
          "-echo both -log log.sparta -k on g 1 -sf kk");
  const std::string sparta_np  = envOr("SPARTA_NP", "1");

  // Log & PID files live in the harness build dir (current working directory)
  const char* pwd_env = std::getenv("PWD");
  const std::string build_dir = (pwd_env && *pwd_env) ? std::string(pwd_env) : ".";
  const std::string log_path  = joinPath(build_dir, "run_spa.log");
  const std::string pid_path  = joinPath(build_dir, "run_spa.pid");

  const std::string deck_dir  = dirname(deck_path);
  const std::string deck_file = basename(deck_path);

  // Construct a simple, robust launch command:
  //
  //   cd "<deck_dir>" &&
  //   nohup env -u DISPLAY -u XAUTHORITY CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  //     mpirun -np <sparta_np> "<sparta_exe>" -in "<deck_file>" <extra_args> \
  //     >> "<log_path>" 2>&1 & echo $! > "<pid_path>"
  //
  std::ostringstream cmd;

  cmd << "cd " << std::quoted(deck_dir)
      << " && env -u DISPLAY -u XAUTHORITY "
      << "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 "
      << "mpirun -np " << sparta_np << " "
      << std::quoted(sparta_exe)
      << " -in " << std::quoted(deck_file) << " ";

  if (!extra_args.empty()) {
    cmd << extra_args << " ";
  }

  // Send all output to both the terminal and the log file
  cmd << "2>&1 | tee " << std::quoted(log_path);

  const std::string cmd_str = cmd.str();

  std::cerr << "[SpartaBridgeShim] Launching external SPARTA:\n  "
            << cmd_str << "\n";

  const int rc = std::system(cmd_str.c_str());

  if (rc != 0) {
    std::cerr << "[SpartaBridgeShim] ERROR: system() returned " << rc
              << " while launching SPARTA. "
              << "See " << log_path << " for details (if created).\n";
  } else {
    std::cerr << "[SpartaBridgeShim] SPARTA command completed (rc=0). "
              << "See " << log_path << " for SPARTA output.\n";
  }

  // if (rc != 0) {
  //   std::cerr << "[SpartaBridgeShim] ERROR: system() returned " << rc
  //             << " while launching SPARTA. "
  //             << "See " << log_path << " for details (if created).\n";
  // } else {
  //   std::cerr << "[SpartaBridgeShim] SPARTA launch dispatched. "
  //             << "PID -> " << pid_path << "; logs -> " << log_path << "\n";
  // }
}

void SpartaBridge::command(const char*) {}
void SpartaBridge::runSteps(int) {}
void SpartaBridge::clear() {}

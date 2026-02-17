// Sim/src/App.hpp
#pragma once

class App {
public:
  App(int argc, char** argv);
  int run();

private:
  int argc_;
  char** argv_;
};

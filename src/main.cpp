#include <iostream>
#include "ReplayMemory.h"

int main() {
  ReplayMemory person("Chandler Scott", 27);

  std::cout << person.getTensor() << std::endl;
}
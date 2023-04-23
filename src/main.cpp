#include <iostream>
#include "ReplayMemory.h"

int main() {
  ReplayMemory memory(8);
  
  memory.push(1);
  memory.push(2);
  memory.push(3);
  memory.push(4);
  memory.push(5);
  
  std::cout << memory.sample(1) << std::endl;
}
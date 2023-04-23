#include <iostream>
#include "ReplayMemory.h"
#include "DQN.h"

int main() {
  DQN policyNetwork = DQN(1, 2);
  DQN targetNetwork = DQN(1, 2);
  ReplayMemory memory(8);
  
  memory.push(1);
  memory.push(2);
  memory.push(3);
  memory.push(4);
  memory.push(5);
  
  std::cout << memory.sample(1) << std::endl;
}
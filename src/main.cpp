#include <iostream>
#include "ReplayMemory.h"
#include "DQN.h"

#include "Environment.h"
#include "TicTacToeEnvironment.h"


int main() {
  DQN policyNetwork = DQN(1, 2);
  DQN targetNetwork = DQN(1, 2);
  ReplayMemory memory(8);
  TicTacToeEnvironment env = TicTacToeEnvironment();
  
  
  memory.push(1);
  memory.push(2);
  memory.push(3);
  memory.push(4);
  memory.push(5);
  env.render();
  
}
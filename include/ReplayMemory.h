#ifndef REPLAYMEMORY_H
#define REPLAYMEMORY_H

#include <string>
#include <ostream>
#include <deque>
#include <random>
#include <algorithm>
#include <torch/torch.h>
#include "Transition.h"

/**
 * Replay Memory
 * 
 * Keep track of state, next state, action, and reward
*/
class ReplayMemory
{
public:
    /**
     * Replay Memory constructor
     * 
     * @param capacity how many memories to remember
    */
    ReplayMemory(int capacity);
    /**
     * Get memory capacity
     * 
     * @return capacity of memory
     */
    int getCapacity() const;

    /**
     * Add a Transition to memory
     *
     * @param value Transition to add
     */
    void push(Transition value);

    /**
     * Get a random sample of memory
     * of size batchSize
     *
     * @param batchSize number of Transitions to sample
     * @return a batch of memories
     */
    std::vector<Transition> sample(int batchSize);

    /**
     * Get number of elements in Memory
     * 
     * @return number of elements
     */
    int size() const;

private:
    // Data structure to hold memory
    std::deque<Transition> memory;
    // Memory capacity; how many transitions we keep track of
    int capacity;
};

#endif // REPLAYMEMORY_H
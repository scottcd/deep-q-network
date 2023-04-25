#include "ReplayMemory.h"

ReplayMemory::ReplayMemory(int capacity) : capacity(capacity) {}

void ReplayMemory::push(Transition value) {
    memory.emplace_back(value);
    if (memory.size() > capacity) {
        memory.pop_front();
    }
}

std::vector<Transition> ReplayMemory::sample(int batchSize) {
    // Create a vector to hold the sampled integers
    std::vector<Transition> memorySample(batchSize);

    // Use std::sample to sample 5 integers from nums
    std::sample(memory.begin(), memory.end(), memorySample.begin(), batchSize, std::mt19937{std::random_device{}()});

    return memorySample;
}

int ReplayMemory::getCapacity() const {
    return capacity;
}

int ReplayMemory::size() const {
    return memory.size();
}
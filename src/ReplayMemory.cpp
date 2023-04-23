#include "ReplayMemory.h"

ReplayMemory::ReplayMemory(int capacity) : capacity(capacity) {}

void ReplayMemory::push(int value) {
    memory.emplace_back(value);
    if (memory.size() > capacity) {
        memory.pop_front();
    }
}

std::vector<int> ReplayMemory::sample(int batchSize) {
    // Create a vector to hold the sampled integers
    std::vector<int> sampled_nums(batchSize);

    // Use std::sample to sample 5 integers from nums
    std::sample(memory.begin(), memory.end(), sampled_nums.begin(), batchSize, std::mt19937{std::random_device{}()});

    return sampled_nums;
}

int ReplayMemory::getCapacity() const {
    return capacity;
}

int ReplayMemory::size() const {
    return memory.size();
}
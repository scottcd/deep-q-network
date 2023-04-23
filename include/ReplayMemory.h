#ifndef REPLAYMEMORY_H
#define REPLAYMEMORY_H

#include <string>
#include <ostream>
#include <deque>
#include <random>
#include <algorithm>
#include <torch/torch.h>

using namespace std;


class ReplayMemory {
public:
    ReplayMemory(int capacity);
    int getCapacity() const;
    void push(int value);
    std::vector<int> sample(int batchSize);
    int size() const;

    friend std::ostream& operator<<(std::ostream& os, const ReplayMemory& replayMemory) {
        os << "ReplayMemory(Capacity=" << replayMemory.capacity << ")";
        return os;
    }

private:
    std::deque<int> memory;
    int capacity;
};

#endif // REPLAYMEMORY_H
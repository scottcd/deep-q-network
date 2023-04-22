#ifndef REPLAYMEMORY_H
#define REPLAYMEMORY_H

#include <string>
#include <ostream>
#include <torch/torch.h>

class ReplayMemory {
public:
    ReplayMemory(const std::string& name, int age);
    std::string getName() const;
    int getAge() const;
    torch::Tensor getTensor() const;

    friend std::ostream& operator<<(std::ostream& os, const ReplayMemory& replayMemory) {
        os << "ReplayMemory(Name=" << replayMemory.mName << ", Age=" << replayMemory.mAge << ")";
        return os;
    }

private:
    std::string mName;
    int mAge;
};

#endif // REPLAYMEMORY_H
#include "ReplayMemory.h"


ReplayMemory::ReplayMemory(const std::string& name, int age) : mName(name), mAge(age) {}

std::string ReplayMemory::getName() const {
    return mName;
}

int ReplayMemory::getAge() const {
    return mAge;
}

torch::Tensor ReplayMemory::getTensor() const {
    torch::Tensor tensor = torch::rand({2, 3});
    return tensor;
}

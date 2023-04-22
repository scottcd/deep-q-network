#include "Person.h"

Person::Person(const std::string& name, int age) : mName(name), mAge(age) {}

std::string Person::getName() const {
    return mName;
}

int Person::getAge() const {
    return mAge;
}
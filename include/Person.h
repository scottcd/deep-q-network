#ifndef PERSON_H
#define PERSON_H

#include <string>
#include <ostream>


class Person {
public:
    Person(const std::string& name, int age);
    std::string getName() const;
    int getAge() const;

    friend std::ostream& operator<<(std::ostream& os, const Person& person) {
        os << "Person(Name=" << person.mName << ", Age=" << person.mAge << ")";
        return os;
    }

private:
    std::string mName;
    int mAge;
};

#endif // PERSON_H
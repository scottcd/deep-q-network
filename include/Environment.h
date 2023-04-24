#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <string>
#include <ostream>
#include <vector>
#include <tuple>

class Environment {
public:
    Environment(int numObservations, int numActions);
    virtual ~Environment();
    virtual std::tuple<std::vector<double>, float, bool> step(int action); 
    virtual std::vector<double> reset();
    virtual void render();
    virtual void close();


private:
    // structure for each possible action
    std::vector<int> actionSpace;
    // structure for current state of the environment
    std::vector<double> observationSpace;
};

#endif // ENVIRONMENT_H
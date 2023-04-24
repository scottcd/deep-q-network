#include "Environment.h"
#include "TicTacToeEnvironment.h"

Environment::Environment(int numObservations, int numActions)
: observationSpace(numObservations), actionSpace(numActions)
{}

Environment::~Environment()
{}

std::vector<double> Environment::reset()
{
    return observationSpace;
}

std::tuple<std::vector<double>, float, bool> Environment::step(int action)
{
        
        return std::make_tuple(observationSpace, 0.0f, false);
}

void Environment::render()
{
    return;
}

void Environment::close()
{
    return;
}
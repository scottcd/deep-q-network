#include "TicTacToeAgent.h"

void displayHelp()
{
  std::cout << "Usage: program_name [numEpisodes] [cleanStart]\n"
            << "Arguments:\n"
            << "  --numEpisodes: Number of episodes (optional, default = 10)\n"
            << "  --cleanStart: Whether to load from file or start anew (optional, default = false)\n"
            << "  --illegalMoveReward: Reward for illegal move (optional, default = -50)\n"
            << "  --legalMoveReward: Reward for legal move (optional, default = 50)\n"
            << "  --winReward: Reward for win (optional, default = 500)\n"
            << "  --lossReward: Reward for loss (optional, default = -500)\n"
            << "  --drawReward: Reward for draw (optional, default = 0)\n"
            << "  --epsilonStart: Value to start epsilon threshold (optional, default = 0.9)\n"
            << "  --epsilonEnd: Value to end epsilon threshold (optional, default = 0.05)\n"
            << "  --epsilonDecay: Rate at which we lower epsilon threshold (optional, default = 1000)\n"
            << "  --batchSize: Number of transitions required before learning (optional, default = 128)\n"
            << "  --gamma: Weight for future states in Q value calculation (optional, default = 0.99)\n"
            << "  --tau: Weight for target network's soft update; that is, how much we update target from policy (optional, default = 0.005)\n"
            << "  --learningRate: Optimizer's learning rate (optional, default = 1e-4)\n"
            << "  --statsFilePath: File path for statistics output (optional, default = out/tic-tac-toe-results.csv)\n";
}

int main(int argc, char *argv[])
{

  TicTacToeAgent a = TicTacToeAgent();

  // parse command line
  for (int i = 1; i < argc; i++)
  {
    std::string arg = std::string(argv[i]);

    if (arg == "-h" || arg == "--help")
    {
      displayHelp();
      return 0;
    }
    else if (arg == "--numEpisodes" && i + 1 < argc)
    {
      a.setNumberEpisodes(std::stoi(argv[++i]));
    }
    else if (arg == "--cleanStart" && i + 1 < argc)
    {
      a.setCleanStart(std::stoi(argv[++i]));
    }
    else if (arg == "--illegalMoveReward" && i + 1 < argc)
    {
      a.setIllegalMoveReward(std::stof(argv[++i]));
    }
    else if (arg == "--legalMoveReward" && i + 1 < argc)
    {
      a.setLegalMoveReward(std::stof(argv[++i]));
    }
    else if (arg == "--winReward" && i + 1 < argc)
    {
      a.setWinReward(std::stof(argv[++i]));
    }
    else if (arg == "--lossReward" && i + 1 < argc)
    {
      a.setLossReward(std::stof(argv[++i]));
    }
    else if (arg == "--drawReward" && i + 1 < argc)
    {
      a.setDrawReward(std::stof(argv[++i]));
    }
    else if (arg == "--epsilonStart" && i + 1 < argc)
    {
      a.setEpsilonStart(std::stof(argv[++i]));
    }
    else if (arg == "--epsilonEnd" && i + 1 < argc)
    {
      a.setEpsilonEnd(std::stof(argv[++i]));
    }
    else if (arg == "--epsilonDecay" && i + 1 < argc)
    {
      a.setEpsilonDecay(std::stof(argv[++i]));
    }
    else if (arg == "--batchSize" && i + 1 < argc)
    {
      a.setBatchSize(std::stoi(argv[++i]));
    }
    else if (arg == "--gamma" && i + 1 < argc)
    {
      a.setGamma(std::stof(argv[++i]));
    }
    else if (arg == "--tau" && i + 1 < argc)
    {
      a.setTau(std::stof(argv[++i]));
    }
    else if (arg == "--learningRate" && i + 1 < argc)
    {
      a.setLearningRate(std::stof(argv[++i]));
    }
    else if (arg == "--statsFilePath" && i + 1 < argc)
    {
      a.setStatsFilePath(argv[++i]);
    }
    else
    {
      std::cout << "Invalid command-line argument: " << arg << std::endl;
      return -1;
    }
  }

  a.train();
}
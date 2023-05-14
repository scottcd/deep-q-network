from agent import Agent

agent = Agent(9, 9, statistics_output='../out/stats.csv',
              policy_output='../out/policy1.pth',
              policy_input='../out/policy.pth')
agent.train()

from agent import Agent

agent = Agent(9, 9, policy_output='../out/policy1.pth', policy_input='../out/policy.pth')
agent.train()
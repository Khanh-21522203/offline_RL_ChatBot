import torch
import torch.nn.functional as F
import torch.distributions as pyd

class BRAC():
    def __init__(self, actor, critic, rewarder, discount, optimizer, lr, device) -> None:
        self.actor = actor
        self.critic = critic
        self.rewarder = rewarder
        self.discount = discount
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        
    def update_critic(self, state, action, next_state):
        with torch.no_grad():
            # Generate 5 action
            generated_actions = self.actor.generate(samples=5)
            generated_actions_Q = []
            max_generated_action_Q = None
            
            # Calculate reward for pair (s,a)
            current_reward = self.rewarder.reward(state, action)
            
            # Calculate target Q
            target_Q = current_reward + self.discount*max_generated_action_Q
            
        # Compute Q1-value, Q2-value    
        current_Q1, current_Q2 = self.critic.double_q(state, action)
        # Compute critic loss
        critic_loss = 0.5 * F.mse_loss(current_Q1, target_Q) + 0.5 * F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 
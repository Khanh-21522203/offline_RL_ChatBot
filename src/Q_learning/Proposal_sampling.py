import torch
from torch.nn import functional as F

class ProposalSampling:
    
    def __init__(self, actor, critic, rewarder, discount, optimizer) -> None:
        self.actor = actor
        self.critic = critic
        self.rewarder = rewarder
        self.discount = discount
        self.optimizer = optimizer

    def update_critic(self, states, actions, next_states):
        with torch.no_grad():
            # dialog_history
            dialog_history = states + actions + next_states
            
            # Generate 5 actions
            generated_actions = self.actor.sampling(prompt=dialog_history, n_samples=5)
            generated_actions_encoded = [self.actor.tokenizer.encode(gen_action, max_length=50, truncation=True, padding="max_length", return_tensors="pt")[0]\
                                        for gen_action in generated_actions]
            
            # Calculate the maximun Q-value of 5 generated actions
            next_state_encoded = self.actor.tokenizer.encode(next_states, max_length=50, truncation=True, padding="max_length", return_tensors="pt")[0]
            
            max_generated_action_Q = max([self.critic(gen_action, next_state_encoded).item() for gen_action in generated_actions_encoded])
            
            # Calculate reward for pair (s,a)
            current_reward = self.rewarder.reward(states, actions)
            
            # Calculate target Q
            target_Q = current_reward + self.discount*max_generated_action_Q
            
        # Compute Q1-value, Q2-value    
        current_Q1, current_Q2 = self.critic.double_q(states, actions)
        # Compute critic loss
        critic_loss = 0.5 * F.mse_loss(current_Q1, target_Q) + 0.5 * F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 

    def train(self, dataloader, learning_rate, device, epochs):
        pass
        
        

        
        
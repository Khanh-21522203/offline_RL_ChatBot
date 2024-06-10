import torch
from torch.nn import functional as F
from Actor import Actor
from Critic import DoubleQCritic

class CQL:
    
    def __init__(self, actor, critic, state_dim, action_dim, discount, optimizer, alpha_prime, temperature) -> None:
        
        self.actor = actor
        self.critic = critic
        # self.rewarder = rewarder
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.discount = discount
        self.optimizer = optimizer
        self.temperature = temperature
        self.alpha_prime = alpha_prime

    def update_critic(self, states, actions, next_states, n_samples):
        dialog_history = [state + action + next_state for state, action, next_state in zip(states, actions, next_states)]
        
        ## Tokenize
        states_tokenized = self.actor.tokenizer(states, max_length=self.state_dim, truncation=True,\
                            padding="max_length", return_tensors="pt")
        actions_tokenized = self.actor.tokenizer(actions, max_length=self.action_dim, truncation=True,\
                            padding="max_length", return_tensors="pt")
        dialog_history_tokenized = self.actor.tokenizer(dialog_history, max_length=None, truncation=True,\
                            padding=True, return_tensors="pt")
        ## Q(state, action)
        q_pred = self.critic(states_tokenized['input_ids'].to(torch.float32), actions_tokenized['input_ids'].to(torch.float32))

        ## Q(next_state, next_action)
        next_actions = self.actor.sampling_highest(prompt=dialog_history_tokenized)
        next_actions_encoded = self.actor.tokenizer(next_actions, max_length=self.action_dim, truncation=True,\
                                padding="max_length", return_tensors="pt")
        next_states_encoded = self.actor.tokenizer(next_states, max_length=self.state_dim, truncation=True,\
                                padding="max_length", return_tensors="pt")
        # return
        next_states_q_value = self.critic(next_states_encoded['input_ids'].to(torch.float32), next_actions_encoded['input_ids'].to(torch.float32))
        
        ## Q_target
        current_reward = self.rewarder.reward(states, actions)
        target_Q = current_reward + self.discount*next_states_q_value
        # target_Q = self.discount*next_states_q_value
        
        bellman_loss = F.mse_loss(q_pred, target_Q)
        ## add CQL regularization
        
        cql_losses = torch.tensor([])
        for i in range(len(states)):
            duplicated_states = dict()
            duplicated_states['input_ids'] = states_tokenized['input_ids'][i].repeat(n_samples, 1)
            duplicated_states['attention_mask'] = states_tokenized['attention_mask'][i].repeat(n_samples, 1)
            
            new_actions = self.actor.sampling(prompt=duplicated_states, n_samples=n_samples)
            new_actions_encoded = self.actor.tokenizer(new_actions, max_length=self.action_dim, truncation=True,\
                                    padding="max_length", return_tensors="pt")
            
            concatinated_q = self.critic(duplicated_states['input_ids'].to(torch.float32), new_actions_encoded['input_ids'].to(torch.float32))
            
            # concatinated_q = torch.cat()
            
            cql_loss = torch.logsumexp(concatinated_q, dim=1) - q_pred.mean()
            cql_losses = torch.cat([cql_losses, cql_loss], dim=-1)
            
        critic_loss = self.alpha_prime*cql_losses.mean() + bellman_loss
        print('done q1')
        ## Update network
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()
        
        return critic_loss

    def train(self, dataloader, epochs=10):
        for epoch in range(epochs):
            for iter, (states, actions, next_states) in enumerate(dataloader):
                loss_value = self.update_critic(states, actions, next_states, 8)
                print(f'iter: {iter}, loss: {loss_value}')
            break
        
        
if __name__=='__main__':
    li = [
        [["Hi, I'm looking to order a pizza from KinPizza. Can you tell me what types of pizzas you have?", "Hi, I'm looking to order a pizza from KinPizza. Can you tell me what types of pizzas you have?"], 
         ["Absolutely! We have a wide variety of pizzas to choose from. Here are some of our most popular options:\n Cheese Pizza: A classic choice with cheese and cream sauce, served with honey.\n Chicken BBQ Pizza: A flavorful combination of cheese, tomato sauce, chicken, onion, bell pepper, and BBQ sauce.\n Veggie Pizza: A vegetarian delight with cheese, tomato sauce, onion, bell pepper, mushroom, and black olive.\n Sausage Pizza: A meaty option with cheese, tomato sauce, and sausage.\n Hawaiian Pizza: A sweet and savory mix of cheese, tomato sauce, bacon, ham, pineapple, and honey.\n Pepperoni Pizza: A crowd-pleaser with cheese, tomato sauce, and pepperoni.",
          "Hi, I'm looking to order a pizza from KinPizza. Can you tell me what types of pizzas you have?"],
         ["Those sound great! I think I'll go with the Chicken BBQ Pizza.",
          "Hi, I'm looking to order a pizza from KinPizza. Can you tell me what types of pizzas you have?"]],
    ]
    dataloader = None
    
    actor = Actor()
    critic = DoubleQCritic(state_dim=100, action_dim=10, hidden_dim=[16,32,64])
    
    agent = CQL(
        actor=actor,
        critic=critic,
        state_dim=100,
        action_dim=10,
        discount=0.9,
        optimizer=torch.optim.Adam(critic.parameters()),
        alpha_prime=0.3,
        temperature=0
    )
    agent.train(li)
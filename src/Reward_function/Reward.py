import torch
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction
import copy

class Reward():
    
    def __init__(self, mlm_path=None, nup_path=None):
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_tokens(["<Customer:>", "<Seller:>"])
        
        if mlm_path:
            self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.load_checkpoint(self.mlm_model, mlm_path)
            self.mlm_model.resize_token_embeddings(len(self.tokenizer))
            
        if nup_path:
            self.nup_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
            self.load_checkpoint(self.nup_model, nup_path)
            self.nup_model.resize_token_embeddings(len(self.tokenizer))
        
    def load_checkpoint(self, model, path):
        checkpoint = torch.load(path,  map_location='cuda:0') if torch.cuda.is_available() else \
                     torch.load(path,  map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        
    def mlm_reward(self, states, actions, max_length):
        combined_sentences = [state + " " + action for state, action in zip(states, actions)]
        tokenized_inputs = self.tokenizer(combined_sentences, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=True)

        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']

        # Tạo tensor lưu xác suất của các token được thay thế trong mỗi câu
        batch_size = len(states)
        masked_lm_probs = []

        for i in range(batch_size):
            tokens_b = self.tokenizer.tokenize(actions[i])
            token_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
            
            sentence_prob = []
            for j in range(len(token_ids_b)):
                masked_input = input_ids[i].clone()
                # Thay thế token của câu B bằng [MASK]
                mask_idx = len(self.tokenizer.tokenize(states[i])) + j + 1  # +1 để bỏ qua token [CLS]
                masked_input[mask_idx] = self.tokenizer.mask_token_id
                
                with torch.no_grad():
                    outputs = self.mlm_model(masked_input.unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
                    prediction_scores = outputs.logits
                    softmax = torch.nn.functional.softmax(prediction_scores[0, mask_idx], dim=-1)
                    masked_token_prob = softmax[token_ids_b[j]].item()
                    sentence_prob.append(masked_token_prob)
            
            masked_lm_probs.append(sum(sentence_prob)/batch_size)
        return masked_lm_probs
            
    def nup_reward(self, states, actions, max_length):
        inputs = self.tokenizer(states, actions, return_tensors='pt', padding=True, truncation=True)

        # Chuyển đầu vào thành tensor
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # Tính xác suất câu B là câu nối tiếp sau câu A
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits

        # Áp dụng softmax để có xác suất
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Lấy xác suất cho nhãn 'is_next' (nhãn 0)
        next_sentence_probs = probs[:, 0].tolist()
        return next_sentence_probs
    
    def reward(self, states, actions):
        mlm_rewards = self.mlm_reward(states, actions, max_length=512)
        nup_rewards = self.nup_reward(self, states, actions, max_length=512)
        rewards = [(mlm_r + nup_r)/2 for mlm_r, nup_r in zip(mlm_rewards, nup_rewards)]
        
        return rewards        
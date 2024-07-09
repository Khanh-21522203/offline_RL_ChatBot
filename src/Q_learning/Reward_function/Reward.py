import torch
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction
import copy

class Reward():
    
    def __init__(self, mlm_path=None, nup_path=None):
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        SPECIAL_TOKENS_DICT = {"pad_token": "<pad>",
                           "eos_token": "<|END|>"}
        self.tokenizer.add_tokens(["<|Assistant|>", "<|User|>"])
        self.tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        
        self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mlm_model.resize_token_embeddings(len(self.tokenizer))
        if mlm_path:
            self.load_checkpoint(self.mlm_model, mlm_path)
            
        self.nup_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.nup_model.resize_token_embeddings(len(self.tokenizer))
        if nup_path:
            self.load_checkpoint(self.nup_model, nup_path)
        
    def load_checkpoint(self, model, path):
        checkpoint = torch.load(path,  map_location='cuda:0') if torch.cuda.is_available() else \
                     torch.load(path,  map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        
    def mlm_reward(self, dialog_histories, utterances, max_length=512):
        # combined_sentences = [dialog_history + " " + utterance for dialog_history, utterance in zip(dialog_histories, utterances)]
        # tokenized_inputs = self.tokenizer(combined_sentences, return_tensors="pt", add_special_tokens=True)
        
        input_ids_list = []
        attention_mask_list = []

        for dialog_historie, utterance in zip(dialog_histories, utterances):
            inputs = self.tokenizer(dialog_historie + " " + utterance, add_special_tokens=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            # Nếu số lượng token vượt quá max_length, cắt bớt từ phía bên trái
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]

            # Thêm padding nếu cần thiết
            if len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        input_ids = torch.tensor(input_ids_list)
        attention_mask = torch.tensor(attention_mask_list)

        # Tạo tensor lưu xác suất của các token được thay thế trong mỗi câu
        batch_size = len(dialog_histories)
        masked_lm_probs = []

        for i in range(batch_size):
            tokens_b = self.tokenizer.tokenize(utterances[i])
            token_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
            
            sentence_prob = []
            for j in range(len(token_ids_b)):
                masked_input = input_ids[i].clone()
                # Thay thế token của câu B bằng [MASK]
                mask_idx = len(self.tokenizer.tokenize(dialog_histories[i])) + j + 1  # +1 để bỏ qua token [CLS]
                masked_input[mask_idx] = self.tokenizer.mask_token_id
                
                with torch.no_grad():
                    outputs = self.mlm_model(masked_input.unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0))
                    prediction_scores = outputs.logits
                    softmax = torch.nn.functional.softmax(prediction_scores[0, mask_idx], dim=-1)
                    masked_token_prob = softmax[token_ids_b[j]].item()
                    sentence_prob.append(masked_token_prob)
            
            masked_lm_probs.append(sum(sentence_prob)/batch_size)
        return masked_lm_probs
            
    def nup_reward(self, dialog_histories, utterances, max_length=512):
        # inputs = self.tokenizer(dialog_histories, utterances, return_tensors='pt', padding=True, truncation=True)

        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []

        for dialog_historie, utterance in zip(dialog_histories, utterances):
            inputs = self.tokenizer(dialog_historie, utterance, add_special_tokens=True)
            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            attention_mask = inputs['attention_mask']
            
            # Nếu số lượng token vượt quá max_length, cắt bớt từ phía bên trái
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                token_type_ids = token_type_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]

            # Thêm padding nếu cần thiết
            if len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids.extend([self.tokenizer.pad_token_id] * pad_length)
                token_type_ids.extend([0] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)
        
        # Chuyển đầu vào thành tensor
        # input_ids = inputs['input_ids']
        # attention_mask = inputs['attention_mask']
        # token_type_ids = inputs['token_type_ids']
        input_ids_list = torch.tensor(input_ids_list)
        token_type_ids_list = torch.tensor(token_type_ids_list)
        attention_mask_list = torch.tensor(attention_mask_list)

        # Tính xác suất câu B là câu nối tiếp sau câu A
        with torch.no_grad():
            outputs = self.nup_model(input_ids_list, attention_mask=attention_mask_list, token_type_ids=token_type_ids_list)
            logits = outputs.logits

        # Áp dụng softmax để có xác suất
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Lấy xác suất cho nhãn 'is_next' (nhãn 0)
        next_sentence_probs = probs[:, 0].tolist()
        return next_sentence_probs
    
    def reward(self, dialog_histories, utterances):
        mlm_rewards = self.mlm_reward(dialog_histories, utterances)
        nup_rewards = self.nup_reward(dialog_histories, utterances, max_length=512)
        rewards = [(mlm_r + nup_r)/2 for mlm_r, nup_r in zip(mlm_rewards, nup_rewards)]
        
        return torch.tensor(rewards)        
    
    
if __name__=="__main__":
    rewarder = Reward()
    print(rewarder.reward(["Hi, I'm interested in ordering a pizza from KinPizza. What types of pizzas do you offer?", "Hi, I'm interested in ordering a pizza from KinPizza. What types of pizzas do you offer?"],\
        ["Welcome to KinPizza! We have a wide variety of pizzas to choose from. Here are some of our most popular options:", "Welcome to KinPizza! We have a wide variety of pizzas to choose from. Here are some of our most popular options:"]))
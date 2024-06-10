from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class Actor():
    
    def __init__(self, model_path=None):
        self.tokenizer = self.define_tokenizer()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.resize_token_embeddings(len(self.tokenizer))
        if model_path:
            self.load_checkpoint()
    
    def define_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        SPECIAL_TOKENS_DICT = {"pad_token": "<pad>",
                            "bos_token": "<startofstring>",
                            "eos_token": "<endofstring>"}
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
        tokenizer.add_tokens(["<Customer:>", "<Seller:>"])
        tokenizer.padding_side = "left"
        return tokenizer
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path,  map_location='cuda:0') if torch.cuda.is_available() else \
                    torch.load(path,  map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=True)
    
    def sampling(self, prompt, n_samples=5, max_length=50):
        # Tokenize prompt
        # responses = [[] for _ in range(len(prompt['input_ids']))]
        # for n in range(n_samples):
            # Generate text with different random seeds for diversity
        output = self.model.generate(
            prompt['input_ids'],
            attention_mask=prompt['attention_mask'],
            max_length=max_length + len(prompt['input_ids'][0]), 
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )
        responses = [self.tokenizer.decode(output[i][len(prompt['input_ids'][i]):], skip_special_tokens=True) for i in range(len(output))]
            # Decode and append response
            # for i in range(len(output)):
            #     responses[i].append(self.tokenizer.decode(output[i][len(prompt['input_ids'][i]):], skip_special_tokens=True).strip())
                
                
            # response = [self.tokenizer.decode(output[i][len(prompt['input_ids'][i]):], skip_special_tokens=True).strip() for i in range(len(output))]
            # responses.append(response)
        
        return responses
    
    def sampling_highest(self, prompt, max_length=50):
        # Generate text using greedy decoding
        output = self.model.generate(
            prompt['input_ids'],
            attention_mask=prompt['attention_mask'],
            max_length=max_length + len(prompt['input_ids'][0]),
            do_sample=False,  # Turn off sampling
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode and return response
        responses = [self.tokenizer.decode(output[i][len(prompt['input_ids'][i]):], skip_special_tokens=True) for i in range(len(output))]
        return responses
    

    
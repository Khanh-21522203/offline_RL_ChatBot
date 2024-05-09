from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import os
import json

class ChatData(Dataset):
    def __init__(self, root:str, tokenizer, max_length=1024):
        self.X = []
        for json_file in os.listdir(root):
            json_path = os.path.join(root, json_file)
            datas = json.load(open(json_path, 'rb'))
        
            for data in datas:
                dialog = data['dialog']
                lines = []
                for idx in range(len(dialog)):
                    if dialog[idx]['role'] == 'customer':
                        lines.append("<Customer:>" + dialog[idx]['text'])
                    if idx == 0:
                        continue
                    if dialog[idx]['role'] == 'seller':
                        # self.X.append("<startofstring> " + dialog[idx-1]['text'] + " <bot>: "+ dialog[idx]['text'] +" <endofstring>")
                        # self.X.append("<startofstring> " + ' '.join(lines) +" <endofstring>")
                        self.X.append("Context:" + '\n'.join(lines) + '\n<Bot:>' + dialog[idx]['text'] + '<End>')
                        lines.append("<Seller:>" + dialog[idx]['text'])
        
        self.X_encoded = tokenizer(self.X, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=True)
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.input_ids[idx])
    
if __name__=="__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    SPECIAL_TOKENS_DICT = {"pad_token": "<pad>",
                        #    "bos_token": "<startofstring>",
                        #    "eos_token": "<endofstring>",
                           "eos_token": "<End>"}
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    tokenizer.add_tokens(["<Customer:>", "<Seller:>", "<Bot:>"])
    
    dataset = ChatData("Dataset/Data", tokenizer)
    print(dataset[10])
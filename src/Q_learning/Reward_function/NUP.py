from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
import json
import os
import numpy as np
from datasets import load_dataset

class NUP_Dataset(Dataset):
    def __init__(self, root:str, tokenizer, max_length=512):
        self.context = []
        self.next_utterances = []
        self.labels = []
        self.Daily_utterances = self.load_Daily_Dialog_Dataset()
        
        for json_file in os.listdir(root):
            json_path = os.path.join(root, json_file)
            datas = json.load(open(json_path, 'rb'))
        
            for data in datas:
                dialog = data['dialog']
                utterances = []
                for idx in range(len(dialog)):
                    if dialog[idx]['role'] == 'customer':
                        utterances.append("<Customer:>" + dialog[idx]['text'])
                    if dialog[idx]['role'] == 'seller':
                        utterances.append("<Seller:>" + dialog[idx]['text'])
                
                for idx, utterance in enumerate(utterances):
                    if idx == 0:
                        continue
                    if utterance.startswith('<Seller:>'):
                        self.context.append(' '.join(utterances[:idx]))
                        # Positive utterance
                        self.next_utterances.append(utterance)
                        self.labels.append(1)
                        
                        # Negative utterance
                        if np.random.rand() < 0.3:
                            # Pick random utterance from same dialog
                            arr = np.arange(len(utterances))
                            arr = arr[arr != idx]
                            random_idx = np.random.choice(arr, 1)
                            random_utterance = utterances[random_idx].replace('<Customer:>', '<Seller:>')
                        else:
                            # Pick random utterance from Daily Dialog Dataset
                            random_utterance = np.random.choice(self.Daily_utterances, 1)[0]
                            random_utterance = '<Seller:>' + random_utterance
                        self.next_utterances.append(random_utterance)
                        self.labels.append(0)
                    
        
        self.X_encoded = tokenizer(self.context, self.next_utterances, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=True)
        self.input_ids = self.X_encoded['input_ids']
        self.token_type_ids = self.X_encoded['token_type_ids']
        self.attention_mask = self.X_encoded['attention_mask']
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.token_type_ids[idx], self.attention_mask[idx], self.labels[idx])
    
    def load_Daily_Dialog_Dataset(self):
        dataset = load_dataset("daily_dialog", split='train')
        DaiLy_utterances = []
        for data in dataset:
            DaiLy_utterances.extend(data['dialog'])
        return DaiLy_utterances
        

def checkDir(args):
    if not os.path.exists(args['checkpoint_dir']):
        print(f"* {args['checkpoint_dir']} does not exist, creating...")
        os.makedirs(args['checkpoint_dir'])

def load_checkpoint(model, optimizer, checkpoint_dir):
    path = os.path.join(checkpoint_dir, 'NUP.pt')
    checkpoint = torch.load(path,  map_location='cuda:0') if torch.cuda.is_available() else \
        torch.load(path,  map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    return epoch

def train(args):
    model = args['model']
    model.resize_token_embeddings(len(args['tokenizer']))
    model.to(args['device'])
    
    optimizer = args['optimizer']
    num_iters = len(args['dataloader'])
    
    checkDir(args)
    if args['continue_training']:
        start_epoch = load_checkpoint(model, optimizer, args['checkpoint_dir'])
        print('Continue training from epoch:', start_epoch)
    else:
        start_epoch = 0
    
    for epoch in range(args['num_epochs']):
        progress_bar = tqdm(args['dataloader'], colour='green')
        for iter, (inputs, token_type_ids, attention_masks, labels) in enumerate(progress_bar):
            inputs = inputs.to(args['device'])
            token_type_ids = token_type_ids.to(args['device'])
            attention_masks = attention_masks.to(args['device'])
            labels = labels.to(args['device'])
            # Forward
            outputs = model(input_ids=inputs, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
            loss_value = outputs.loss
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.5f}".format(epoch+1, args['num_epochs'], iter+1, num_iters, loss_value))
            # Backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        if args['checkpoint_dir']:
            checkpoint = {
                "epoch": epoch+1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, "{}/NUP.pt".format(args['checkpoint_dir']))


if __name__=="__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    
    tokenizer.add_tokens(["<Customer:>", "<Seller:>"])
    
    mlm_dataset = NUP_Dataset("", tokenizer)
    mlm_DataLoader =  DataLoader(mlm_dataset, 
                           batch_size=2, 
                           shuffle=True,
                           num_workers=2,
                           drop_last=True)
    
    optim = Adam(model.parameters(), lr=1e-5)

    args = {'tokenizer': tokenizer,
            'dataloader': mlm_DataLoader,
            'model': model,
            'optimizer': optim,
            'num_epochs': 100,
            'checkpoint_dir': None,
            'device': 'cpu'}
    
    train(args)
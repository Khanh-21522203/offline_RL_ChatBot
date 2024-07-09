from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam
from tqdm import tqdm
import json
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

class MLM_Dataset(Dataset):
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
                        self.X.append("Context:" + '\n'.join(lines) + '\n<Bot:>' + dialog[idx]['text'] + '<End>')
                        lines.append("<Seller:>" + dialog[idx]['text'])
        
        self.X_encoded = tokenizer(self.X, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=True)
        # self.X_encoded['labels'] = self.X_encoded.input_ids.detach().clone()
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']
        
    def len(self):
        pass
    
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx], self.input_ids[idx])
    
def mask_input(inputs):
    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs != 101) * (inputs != 102) * (inputs != 0)
    # replace selected token by [MASK] token
    selection = [torch.flatten(mask_arr[i].nonzero()).tolist() for i in range(len(mask_arr))]
    for i in range(inputs.shape[0]):
        inputs[i, selection[i]] = 103

def checkDir(args):
    if not os.path.exists(args['checkpoint_dir']):
        print(f"* {args['checkpoint_dir']} does not exist, creating...")
        os.makedirs(args['checkpoint_dir'])

def load_checkpoint(model, optimizer, checkpoint_dir):
    path = os.path.join(checkpoint_dir, 'model-1e4.pt')
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
        for iter, (inputs, attention_masks, labels) in enumerate(progress_bar):
            mask_input(inputs)
            inputs = inputs.to(args['device'])
            attention_masks = attention_masks.to(args['device'])
            labels = labels.to(args['device'])
            # Forward
            outputs = model(input_ids=inputs, attention_mask=attention_masks, labels=labels)
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
            torch.save(checkpoint, "{}/model.pt".format(args['checkpoint_dir']))


if __name__=="__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(["<Customer:>", "<Seller:>"])
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    mlm_dataset = MLM_Dataset("", tokenizer)
    mlm_DataLoader =  DataLoader(mlm_dataset, 
                           batch_size=16, 
                           shuffle=True,
                           num_workers=2,
                           drop_last=True)
    
    optim = Adam(model.parameters(), lr=1e-4)

    args = {'tokenizer': tokenizer,
            'dataloader': mlm_DataLoader,
            'model': model,
            'optimizer': optim,
            'num_epochs': 100,
            'checkpoint_dir': None,
            'device': 'cpu'}
    
    train(args)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from Dataset import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

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

def define_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    SPECIAL_TOKENS_DICT = {"pad_token": "<pad>",
                           "bos_token": "<startofstring>",
                           "eos_token": "<endofstring>"}
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    tokenizer.add_tokens(["<Customer:>", "<Seller:>"])
    
    return tokenizer

# def infer(inputs):
#     inp = "<startofstring> "+inp+" <bot>: "
#     inp = tokenizer(inp, return_tensors="pt")
#     X = inp["input_ids"].to(device)
#     a = inp["attention_mask"].to(device)
#     output = model.generate(X, attention_mask=a )
#     output = tokenizer.decode(output[0])
#     return output


if __name__=='__main__':
    
    tokenizer = define_tokenizer()
    
    chatData = ChatData("Dataset/Data", tokenizer)
    chatDataLoader =  DataLoader(chatData, 
                           batch_size=16, 
                           shuffle=True,
                           num_workers=2,
                           drop_last=True)
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    optim = Adam(model.parameters(), lr=1e-4)

    args = {'tokenizer': tokenizer,
            'dataloader': chatDataLoader,
            'model': model,
            'optimizer': optim,
            'num_epochs': 100,
            'checkpoint_dir': None,
            'device': 'cpu'}
    
    train(args)
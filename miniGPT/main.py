import torch
import os.path
from miniGPT import GPTLanguageModel


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, device, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    # initialize parameters
    # hyperparameters
    batch_size = 64 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 500
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    # ------------
    print(device)

    filename = "model_weights.pth"

    # load data
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    # Define training parameters
    model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)
    m = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if not os.path.exists(filename):
        for iter in range(max_iters):
         # every once in a while evaluate the loss on train and val sets
            X, Y = get_batch('train')
            optimizer.zero_grad()
            logits, loss = model(X, device, Y)
            loss.backward()
            optimizer.step()

            if iter % 100 == 0:
                out = estimate_loss(device)
                print(f"{iter}: {out}")

        torch.save(model.state_dict(), filename)

        # try to load the model
        model.load_state_dict(torch.load(filename))    
            

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500, block_size=block_size, device=device)[0].tolist()))

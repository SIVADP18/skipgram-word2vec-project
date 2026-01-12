import re 
from collections import Counter 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
 
 
file_path = "enwik9.txt" 
 
with open(file_path, "r", encoding="utf-8") as f: 
    text = f.read().lower() 
 
 
words = re.findall(r"[a-z]+", text) 
 
print("Total words:", len(words)) 
print("Sample words:", words[:20]) 
 
word_counts = Counter(words) 
 
 
vocab = {w:c for w,c in word_counts.items() if c >= 5} 
 
print("Vocabulary size:", len(vocab)) 
 
idx2word = list(vocab.keys()) 
word2idx = {w:i for i,w in enumerate(idx2word)} 
 
print("Index of 'king':", word2idx.get("king")) 
print("Word at index 100:", idx2word[100]) 
 
 
words = words[:100_000]   
 
 
window_size = 2   
 
training_pairs = [] 
 
 
for i, word in enumerate(words): 
    if word not in word2idx: 
        continue 
 
    center = word2idx[word] 
 
    start = max(0, i - window_size) 
    end   = min(len(words), i + window_size + 1) 
 
    for j in range(start, end): 
        if i == j: 
            continue 
        context_word = words[j] 
        if context_word in word2idx: 
            training_pairs.append((center, word2idx[context_word])) 
training_pairs = training_pairs[:200_000] 
 
print("Total training pairs:", len(training_pairs)) 
print("Sample pairs:", training_pairs[:10]) 
 
freqs = np.array([vocab[w] for w in idx2word], dtype=np.float32) 
unigram_dist = freqs ** 0.75 
unigram_dist = unigram_dist / unigram_dist.sum() 
 
def get_negative_samples(pos_index, k=2): 
    negatives = [] 
    while len(negatives) < k: 
        neg = np.random.choice(len(idx2word), p=unigram_dist) 
        if neg != pos_index: 
            negatives.append(neg) 
    return negatives 
 
vocab_size = len(idx2word) 
embed_size = 50   
 
input_emb  = nn.Embedding(vocab_size, embed_size) 
output_emb = nn.Embedding(vocab_size, embed_size) 
 
optimizer = optim.Adam(list(input_emb.parameters()) + 
                       list(output_emb.parameters()), lr=0.001) 
 
criterion = nn.BCEWithLogitsLoss() 
 
 
def train_skipgram(pairs, epochs=1, batch_size=512): 
 
    for epoch in range(epochs): 
        total_loss = 0 
 
        for i in range(0, len(pairs), batch_size): 
            batch = pairs[i:i+batch_size] 
            optimizer.zero_grad() 
 
            loss = 0 
 
            for center, context in batch: 
 
                
                center_vec = input_emb(torch.tensor([center])) 
                pos_vec    = output_emb(torch.tensor([context])) 
 
                pos_score = torch.matmul(center_vec, pos_vec.T) 
                pos_loss  = criterion(pos_score, 
torch.ones_like(pos_score)) 
 
                
                neg_samples = get_negative_samples(context, k=2) 
                neg_vecs = output_emb(torch.tensor(neg_samples)) 
 
                neg_score = torch.matmul(center_vec, neg_vecs.T) 
                neg_loss  = criterion(neg_score, 
torch.zeros_like(neg_score)) 
 
                loss += pos_loss + neg_loss 
 
            loss.backward() 
            optimizer.step() 
            total_loss += loss.item() 
 
        print(f"Epoch {epoch+1} completed | Loss: {total_loss:.2f}") 
 
print("Starting training...") 
train_skipgram(training_pairs, epochs=1) 
print("Training finished!")

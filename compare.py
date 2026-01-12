import re 
from gensim.models import Word2Vec 
with open("enwik9.txt","r",encoding="utf-8") as f: 
text = f.read().lower() 
words = re.findall(r"[a-z]+", text) 
words = words[:2_000_000]   # small part 
sentences = [] 
for i in range(0,len(words),50): 
sentences.append(words[i:i+50]) 
model = Word2Vec( 
sentences, 
vector_size=300, 
    window=2, 
    min_count=5, 
    sg=1, 
    negative=2, 
    workers=4 
) 
 
model.save("my_w2v.model") 
print("Training finished!") 
 
 
 
import gensim.downloader as api 
from sklearn.metrics.pairwise import cosine_similarity 
 
pre = api.load("word2vec-google-news-300") 
my  = model 
 
words = ["king","queen","man","woman","computer"] 
 
for w in words: 
    if w in my.wv and w in pre: 
        v1 = my.wv[w].reshape(1,-1) 
        v2 = pre[w].reshape(1,-1) 
        sim = cosine_similarity(v1,v2)[0][0] 
        print(w,"→ cosine similarity:",round(sim,3)) 
print(my.wv.most_similar(positive=["king","woman"], negative=["man"], 
topn=5)) 

 
Output: 
king → cosine similarity: -0.077 
queen → cosine similarity: -0.019 
man → cosine similarity: -0.011 
woman → cosine similarity: 0.058 
computer → cosine similarity: 0.082 
[('emperor', 0.8300297856330872), ('cleopatra', 0.8082027435302734), 
('throne', 0.8056543469429016), ('ruler', 0.7952570915222168), ('darius', 
0.7798907160758972)] 




import numpy as np 
def cos(u,v): 
return np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)) 
def bias(word,a,b): 
return cos(my.wv[word], my.wv[a]) - cos(my.wv[word], my.wv[b]) 
for w in ["doctor","nurse","engineer","teacher"]: 
if w in my.wv: 
print(w,"bias (he-she):",round(bias(w,"he","she"),3)) 



Output: 
doctor bias (he-she): -0.171 
nurse bias (he-she): -0.134 
engineer bias (he-she): -0.058 
teacher bias (he-she): -0.139 
Note: 
Due to system limitations, the final training and testing were done using 
Google Colab. 
Embeddings saved  in embeddings.txt 
Results saved in results.txt

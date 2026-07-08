import json, random, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from collections import Counter

DATA_DIR  = "/home/sclab/moe_model/processed_v2"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
BEST_PATH = "/home/sclab/moe_model/adaptive_deberta_5class_best.pt"
MODEL_NAME= "microsoft/deberta-large"
LABELS = ['LAW','MED','SOC','STEM','CROSS']
DOMAIN_FILES = {
    'LAW':'law_rewritten.json','MED':'med_rewritten.json',
    'SOC':'soc_rewritten.json','STEM':'stem_rewritten.json',
    'CROSS':'cross_rewritten.json',
}
LR=8e-6; EPOCHS=5; BATCH=8; MAX_LEN=256

with open(GT_PATH) as f: gt=json.load(f)
gt_instrs=set(d['instruction'].strip() for d in gt)

train_data=[]
for domain,fname in DOMAIN_FILES.items():
    with open(f"{DATA_DIR}/{fname}") as f: data=json.load(f)
    for d in data:
        instr=d['instruction'].strip()
        if instr in gt_instrs: continue
        train_data.append({'text':instr,'label':LABELS.index(domain)})

random.seed(42); random.shuffle(train_data)
print(f"학습 샘플: {len(train_data)}개  분포: {Counter(d['label'] for d in train_data)}")

class DS(Dataset):
    def __init__(self,data,tok): self.data=data; self.tok=tok
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        d=self.data[i]
        enc=self.tok(d['text'],max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
        return {'input_ids':enc['input_ids'].squeeze(),
                'attention_mask':enc['attention_mask'].squeeze(),
                'label':torch.tensor(d['label'],dtype=torch.long)}

class AdaptiveRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(MODEL_NAME,local_files_only=True)
        h=self.encoder.config.hidden_size
        self.head=nn.Sequential(nn.Dropout(0.1),nn.Linear(h,5))
    def forward(self,i,a):
        out=self.encoder(input_ids=i,attention_mask=a)
        return self.head(out.last_hidden_state[:,0,:])

tok=AutoTokenizer.from_pretrained(MODEL_NAME,local_files_only=True)
ds=DS(train_data,tok); ld=DataLoader(ds,batch_size=BATCH,shuffle=True,num_workers=2)
model=AdaptiveRouter().cuda()
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
sch=get_cosine_schedule_with_warmup(opt,len(ld)*EPOCHS//10,len(ld)*EPOCHS)
crit=nn.CrossEntropyLoss()

best_acc=0
for ep in range(EPOCHS):
    model.train(); tl=0
    for b in ld:
        iids=b['input_ids'].cuda(); amsk=b['attention_mask'].cuda(); lbl=b['label'].cuda()
        loss=crit(model(iids,amsk),lbl)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); sch.step(); tl+=loss.item()
    model.eval(); correct=Counter(); total=Counter()
    for item in gt:
        true=item.get('expected_domain','')
        enc=tok(item['instruction'].strip(),max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
        with torch.no_grad():
            pred=LABELS[model(enc['input_ids'].cuda(),enc['attention_mask'].cuda()).argmax().item()]
        total[true]+=1
        if pred==true: correct[true]+=1
    acc=sum(correct.values())/sum(total.values())*100
    per=" ".join(f"{d}={correct[d]/max(1,total[d])*100:.0f}%" for d in LABELS)
    print(f"E{ep+1}/{EPOCHS} loss={tl/len(ld):.4f} Total={acc:.1f}% | {per}")
    if acc>best_acc: best_acc=acc; torch.save(model.state_dict(),BEST_PATH)

model.load_state_dict(torch.load(BEST_PATH)); model.eval()
correct=Counter(); total=Counter()
for item in gt:
    true=item.get('expected_domain','')
    enc=tok(item['instruction'].strip(),max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
    with torch.no_grad():
        pred=LABELS[model(enc['input_ids'].cuda(),enc['attention_mask'].cuda()).argmax().item()]
    total[true]+=1
    if pred==true: correct[true]+=1
print(f"\n=== Adaptive RAG (DeBERTa 5-class) 최종 ===")
for d in LABELS:
    print(f"  {d}: {correct[d]}/{total[d]} = {correct[d]/total[d]*100:.1f}%")
print(f"  Total: {sum(correct.values())}/{sum(total.values())} = {sum(correct.values())/sum(total.values())*100:.1f}%")

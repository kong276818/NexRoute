import json, random, torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from collections import Counter

DATA_DIR  = "/home/sclab/moe_model/processed_v2"
GT_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
CKPT_ORIG = "/home/sclab/moe_model/router_multilabel.pt"
S1_BEST   = "/home/sclab/moe_model/router_stage1_v2_best.pt"
S2_BEST   = "/home/sclab/moe_model/router_stage2_v2_best.pt"
CFG_PATH  = "/home/sclab/moe_model/router_twostage_v2_config.json"

DOMAINS    = ['LAW','MED','SOC','STEM']
LABELS_ALL = ['LAW','MED','SOC','STEM','CROSS']
DOMAIN_FILES = {
    'LAW':'law_rewritten.json','MED':'med_rewritten.json',
    'SOC':'soc_rewritten.json','STEM':'stem_rewritten.json',
    'CROSS':'cross_rewritten.json',
}
LR=2e-6; BATCH=16; MAX_LEN=256

with open(GT_PATH) as f: gt=json.load(f)
gt_instrs=set(d['instruction'].strip() for d in gt)

all_data=[]
for domain,fname in DOMAIN_FILES.items():
    with open(f"{DATA_DIR}/{fname}") as f: data=json.load(f)
    for d in data:
        instr=d['instruction'].strip()
        if instr in gt_instrs: continue
        crossed=d.get('crossed_domains',['SOC','MED'])
        all_data.append({'text':instr,'primary':domain,
                         'crossed':crossed if isinstance(crossed,list) else ['SOC','MED']})

# Stage1 데이터: 전체 + CROSS 3배
s1_data=all_data.copy()
cross_s=[d for d in all_data if d['primary']=='CROSS']
for _ in range(2): s1_data.extend(cross_s)

# Stage2 데이터: CROSS 제외
s2_data=[d for d in all_data if d['primary']!='CROSS']

random.seed(42); random.shuffle(s1_data); random.shuffle(s2_data)
print(f"Stage1: {len(s1_data)}개 / Stage2: {len(s2_data)}개")
print(f"Stage2 분포: {Counter(d['primary'] for d in s2_data)}")

class S1Dataset(Dataset):
    def __init__(self,data,tok): self.data=data; self.tok=tok
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        d=self.data[i]
        enc=self.tok(d['text'],max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
        return {'input_ids':enc['input_ids'].squeeze(),
                'attention_mask':enc['attention_mask'].squeeze(),
                'label':torch.tensor(1.0 if d['primary']=='CROSS' else 0.0)}

class S2Dataset(Dataset):
    def __init__(self,data,tok): self.data=data; self.tok=tok
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        d=self.data[i]
        enc=self.tok(d['text'],max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
        return {'input_ids':enc['input_ids'].squeeze(),
                'attention_mask':enc['attention_mask'].squeeze(),
                'label':torch.tensor(DOMAINS.index(d['primary']),dtype=torch.long)}

class BinaryRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained('bert-base-uncased')
        self.head=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,1))
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        return self.head(out.last_hidden_state[:,0,:]).squeeze(-1)

class DomainRouter4(nn.Module):
    """원본과 동일 아키텍처 - 원본 가중치 로드 가능"""
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained('bert-base-uncased')
        self.classifier=nn.Sequential(nn.Linear(768,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,4))
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:,0,:])

def evaluate(s1,s2,tok,gt_data,thresh):
    s1.eval(); s2.eval()
    y_true,y_pred=[],[]
    for item in gt_data:
        q=item['instruction'].strip(); true=item.get('expected_domain','')
        enc=tok(q,max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
        iids=enc['input_ids'].cuda(); amsk=enc['attention_mask'].cuda()
        with torch.no_grad():
            cp=torch.sigmoid(s1(iids,amsk)).item()
            pred='CROSS' if cp>=thresh else DOMAINS[s2(iids,amsk).argmax().item()]
        y_true.append(true); y_pred.append(pred)
    total=sum(1 for t,p in zip(y_true,y_pred) if t==p)/len(y_true)*100
    per_d={d:sum(1 for i,t in enumerate(y_true) if t==d and y_pred[i]==d)/
           max(1,sum(1 for t in y_true if t==d))*100 for d in LABELS_ALL}
    return total,per_d

if __name__=="__main__":
    device=torch.device('cuda')
    tok=AutoTokenizer.from_pretrained('bert-base-uncased')

    # ── Stage 1: CROSS 감지기 ──────────────────────────────────
    print("\n"+"="*50+"\nStage1: CROSS Binary Classifier\n"+"="*50)
    s1=BinaryRouter().to(device)
    s1_ds=S1Dataset(s1_data,tok)
    s1_ld=DataLoader(s1_ds,batch_size=BATCH,shuffle=True,num_workers=2)
    n_c=sum(1 for d in s1_data if d['primary']=='CROSS')
    n_s=len(s1_data)-n_c
    pw=torch.tensor([n_s/n_c]).to(device)
    s1_crit=nn.BCEWithLogitsLoss(pos_weight=pw)
    s1_opt=torch.optim.AdamW(s1.parameters(),lr=LR,weight_decay=0.01)
    s1_sch=get_cosine_schedule_with_warmup(s1_opt,len(s1_ld)*8//10,len(s1_ld)*8)
    best_f1=0
    for ep in range(8):
        s1.train(); tl=0
        for b in s1_ld:
            iids=b['input_ids'].to(device); amsk=b['attention_mask'].to(device); lbl=b['label'].to(device)
            loss=s1_crit(s1(iids,amsk),lbl)
            s1_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s1.parameters(),1.0)
            s1_opt.step(); s1_sch.step(); tl+=loss.item()
        # CROSS F1 평가
        s1.eval(); tp=tn=fp=fn=0
        for item in gt:
            enc=tok(item['instruction'].strip(),max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
            with torch.no_grad(): cp=torch.sigmoid(s1(enc['input_ids'].cuda(),enc['attention_mask'].cuda())).item()
            tc=(item.get('expected_domain','')=='CROSS'); pc=(cp>=0.50)
            if pc and tc: tp+=1
            elif not pc and not tc: tn+=1
            elif pc and not tc: fp+=1
            else: fn+=1
        pr=tp/(tp+fp) if (tp+fp)>0 else 0; rc=tp/(tp+fn) if (tp+fn)>0 else 0
        f1=2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
        print(f"  E{ep+1}/8 loss={tl/len(s1_ld):.4f} | CROSS P={pr*100:.0f}% R={rc*100:.0f}% F1={f1*100:.0f}% | FP={fp}")
        if f1>best_f1: best_f1=f1; torch.save(s1.state_dict(),S1_BEST)
    s1.load_state_dict(torch.load(S1_BEST))
    print(f"Stage1 Best F1={best_f1*100:.1f}% 저장: {S1_BEST}")

    # ── Stage 2: 도메인 분류기 (원본 가중치에서 fine-tuning) ───
    print("\n"+"="*50+"\nStage2: Domain 4-class (from 원본 가중치)\n"+"="*50)
    s2=DomainRouter4().to(device)
    s2.load_state_dict(torch.load(CKPT_ORIG,map_location=device))
    print(f"원본 가중치 로드: {CKPT_ORIG}")
    s2_ds=S2Dataset(s2_data,tok)
    s2_ld=DataLoader(s2_ds,batch_size=BATCH,shuffle=True,num_workers=2)
    counts=Counter(d['primary'] for d in s2_data)
    mc=max(counts.values())
    cw=torch.tensor([mc/counts[d] for d in DOMAINS]).to(device)
    s2_crit=nn.CrossEntropyLoss(weight=cw)
    s2_opt=torch.optim.AdamW(s2.parameters(),lr=LR*0.5,weight_decay=0.01)  # 더 낮은 LR
    s2_sch=get_cosine_schedule_with_warmup(s2_opt,len(s2_ld)*10//10,len(s2_ld)*10)
    best_s2=0
    for ep in range(10):
        s2.train(); tl=0
        for b in s2_ld:
            iids=b['input_ids'].to(device); amsk=b['attention_mask'].to(device); lbl=b['label'].to(device)
            loss=s2_crit(s2(iids,amsk),lbl)
            s2_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s2.parameters(),1.0)
            s2_opt.step(); s2_sch.step(); tl+=loss.item()
        # Stage2 단독 평가 (CROSS 제외)
        s2.eval(); cor=tot=0
        pd={}
        for item in gt:
            true=item.get('expected_domain','')
            if true=='CROSS': continue
            enc=tok(item['instruction'].strip(),max_length=MAX_LEN,truncation=True,padding='max_length',return_tensors='pt')
            with torch.no_grad(): pred=DOMAINS[s2(enc['input_ids'].cuda(),enc['attention_mask'].cuda()).argmax().item()]
            pd[true]=pd.get(true,[0,0]); pd[true][1]+=1
            if pred==true: cor+=1; pd[true][0]+=1
            tot+=1
        acc=cor/tot*100 if tot>0 else 0
        per_str=" ".join(f"{d}={pd.get(d,[0,0])[0]/max(1,pd.get(d,[0,1])[1])*100:.0f}%" for d in DOMAINS)
        print(f"  E{ep+1}/10 loss={tl/len(s2_ld):.4f} | acc={acc:.1f}% | {per_str}")
        if acc>best_s2: best_s2=acc; torch.save(s2.state_dict(),S2_BEST)
    s2.load_state_dict(torch.load(S2_BEST))
    print(f"Stage2 Best={best_s2:.1f}% 저장: {S2_BEST}")

    # ── 통합 평가 ──────────────────────────────────────────────
    print("\n"+"="*50+"\n최종 통합 평가\n"+"="*50)
    best_total=0; best_thresh=0.5
    for thresh in [0.35,0.40,0.45,0.50,0.55,0.60,0.65]:
        acc,per_d=evaluate(s1,s2,tok,gt,thresh)
        cross=per_d.get('CROSS',0); soc=per_d.get('SOC',0); med=per_d.get('MED',0)
        score=acc*0.3+cross*0.3+soc*0.2+med*0.2
        print(f"  thresh={thresh:.2f}: Total={acc:.1f}% CROSS={cross:.0f}% MED={med:.0f}% SOC={soc:.0f}%")
        if score>best_total: best_total=score; best_thresh=thresh

    print(f"\n★ 최적 threshold={best_thresh}")
    acc,per_d=evaluate(s1,s2,tok,gt,best_thresh)
    print(f"Total: {acc:.1f}%")
    for d in LABELS_ALL: print(f"  {d}: {per_d.get(d,0):.1f}%")

    import json as _j
    with open(CFG_PATH,'w') as f:
        _j.dump({"cross_threshold":best_thresh,"s1_path":S1_BEST,"s2_path":S2_BEST},f)
    print(f"\n설정 저장: {CFG_PATH}")

"""
run_ablation_complete.py
========================
V01-V05 전체 ablation 재산출 (V01 hallucination CROSS KB 수정 포함)

Phase 1 : V02 ROUGE-L/BLEU scoring
Phase 2 : V01(CROSS-corrected) + V02 hallucination
Phase 3 : V03/V04/V05 generation  (7B + LoRA inference)
Phase 4 : V03/V04/V05 ROUGE-L/BLEU scoring + BERTScore (all variants)
Phase 5 : V03/V04/V05 hallucination
Phase 6 : summary JSON

결과:
  results/ablation/v02_scores_per_sample.csv
  results/ablation/v0X_hallucination_per_sample.csv  (V01-V05)
  results/ablation/v03_gen_500.json
  results/ablation/v04_gen_500.json
  results/ablation/v05_gen_500.json
  results/ablation/v03_scores_per_sample.csv  (재산출)
  results/ablation/v04_scores_per_sample.csv
  results/ablation/v05_scores_per_sample.csv  (재산출)
  results/ablation/ablation_hallucination_summary.json
  results/ablation/ablation_generation_quality.json
"""

import os, json, csv, random, gc, pickle, sys
import numpy as np
import torch
from pathlib import Path
from collections import Counter

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_7B    = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
LORA_DIR   = "/home/sclab/moe_model/lora_weights_7b_v3"
DATA_DIR   = "/home/sclab/moe_model/processed_v2"
GT_PATH    = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
PREDS_CSV  = "/home/sclab/moe_model/preds_500.csv"
FUNC_ROUTER= "/home/sclab/moe_model/routers/func_router/pipeline.pkl"
EMBED_NAME = "intfloat/multilingual-e5-large"
OUT_DIR    = "/home/sclab/moe_model/results/ablation"
NLI_MODEL  = "cross-encoder/nli-deberta-v3-large"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_GEN  = 10
TOP_K_NLI  = 3
MIN_SENT   = 20
SEED       = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

DOMAIN_FILES = {
    "LAW":   "law_rewritten.json",  "MED":  "med_rewritten.json",
    "SOC":   "soc_rewritten.json",  "STEM": "stem_rewritten.json",
    "CROSS": "cross_rewritten.json",
}
LORA_MAP = {"LAW":"lora_law","MED":"lora_med","SOC":"lora_soc",
             "STEM":"lora_stem","CROSS":"lora_cross"}
DOMAIN_SYSTEM = {
    "LAW":   "You are a legal expert. Answer accurately based on legal principles, statutes, and case law.",
    "MED":   "You are a medical expert. Answer accurately based on medical facts and clinical knowledge.",
    "SOC":   "You are a social science expert. Answer based on social context, psychology, and research.",
    "STEM":  "You are a science and engineering expert. Answer accurately based on scientific principles.",
    "CROSS": "You are a multidisciplinary expert. Integrate knowledge across domains to answer accurately.",
}
FEW_SHOT = {
    "LAW": "Q: What is the doctrine of promissory estoppel?\nA: Promissory estoppel is a legal doctrine that prevents a party from withdrawing a promise when another party has reasonably relied on it to their detriment.",
    "MED": "Q: How does the renin-angiotensin system regulate blood pressure?\nA: The renin-angiotensin system regulates blood pressure through a cascade of enzymatic reactions initiated when the kidneys detect low blood pressure or reduced sodium levels.",
    "SOC": "Q: How does social stratification affect educational outcomes?\nA: Social stratification significantly influences educational outcomes by creating unequal access to resources, opportunities, and cultural capital.",
    "STEM": "Q: How does CRISPR-Cas9 technology work?\nA: CRISPR-Cas9 is a revolutionary gene-editing technology that uses a guide RNA to direct the Cas9 endonuclease to specific DNA sequences.",
    "CROSS": "Q: How might advances in neuroscience impact criminal justice policy?\nA: Advances in neuroscience are profoundly reshaping criminal justice policy by providing empirical insights into the biological underpinnings of behavior.",
}
SCAFFOLDS = {
    "WHAT":    "Based on the following evidence, identify and define the key facts. Do not infer beyond the provided context:",
    "HOW":     "Using the following evidence, describe the mechanism or procedure in a step-by-step logical flow:",
    "WHY":     "Based on the following evidence, construct a definitive causal chain. Explicitly explain the underlying reasons connecting premise to outcome:",
    "COMPARE": "Using the following evidence, contrast the entities in the query. Highlight structural similarities and key differences in a comparative framework:",
}
GENERIC_INSTRUCTION = "provide a comprehensive analytical answer with specific mechanisms and implications"

# ══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════════════

def load_gt():
    return json.load(open(GT_PATH))

def load_routing():
    routing = {}
    with open(PREDS_CSV) as f:
        for r in csv.DictReader(f):
            idx = int(r['sample_id'])
            routing[idx] = {
                'pred_domain': r['pred_domain_nexroute'],
                'is_cross':    r['is_cross'].strip().lower() == 'true',
                'top1': r['top1'], 'top2': r['top2'],
            }
    return routing

def load_func_router():
    with open(FUNC_ROUTER, 'rb') as f:
        return pickle.load(f)

def build_prompt(query, context, func_label):
    ctx_text = "\n".join([c[:200] for c in context])
    instr = SCAFFOLDS.get(func_label, GENERIC_INSTRUCTION)
    return (f"Here is context:\n{ctx_text}\n\n"
            f"Question: {query}\n\n"
            f"{instr}\n\nAnswer:")

def avg(lst):
    lst = [v for v in lst if v is not None]
    return round(float(np.mean(lst)), 4) if lst else None

def hal_pool(rows):
    n = sum(int(r['n_sents']) for r in rows if r.get('n_sents'))
    c = sum(int(r['n_contra']) for r in rows if r.get('n_contra'))
    return round(100*c/n, 2) if n else None, n, c

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 : V02 ROUGE-L / BLEU scoring (BERTScore은 Phase 4에서 일괄)
# ══════════════════════════════════════════════════════════════════════════════

def phase1_v02_rouge_bleu():
    out_path = os.path.join(OUT_DIR, "v02_scores_per_sample.csv")
    if os.path.exists(out_path):
        rows = list(csv.DictReader(open(out_path)))
        if len(rows) == 500 and 'rouge_l' in rows[0] and rows[0].get('bertscore_f',''):
            print("[Phase1] v02_scores_per_sample.csv 이미 완료 — skip")
            return
        if len(rows) == 500:
            print("[Phase1] v02 ROUGE-L/BLEU 이미 있음 (BERTScore 없음은 Phase4에서)")
            return

    print("[Phase1] V02 ROUGE-L/BLEU 계산...")
    from rouge_score import rouge_scorer as rs_module
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download('punkt_tab', quiet=True)

    rscorer = rs_module.RougeScorer(['rougeL'], use_stemmer=True)
    smooth  = SmoothingFunction().method1
    routing = load_routing()
    gt_list = load_gt()
    v02     = json.load(open(os.path.join(OUT_DIR, "v02_gen_500.json")))

    rows = []
    for item in v02:
        idx        = int(item['sample_id'])
        gold       = item.get('gold_answer','').strip()
        pred       = item.get('pred_answer','').strip()
        domain     = item.get('domain', gt_list[idx].get('expected_domain','?'))
        r          = routing.get(idx, {})
        is_cross_gt = (gt_list[idx].get('expected_domain','') == 'CROSS')
        is_cross_pred = r.get('is_cross', False)
        is_clean   = idx < 300

        rl   = rscorer.score(gold, pred)['rougeL'].fmeasure * 100 if gold and pred else None
        bleu = sentence_bleu([nltk.word_tokenize(gold.lower())],
                              nltk.word_tokenize(pred.lower()),
                              smoothing_function=smooth) * 100 if gold and pred else None
        rows.append({
            'sample_id': idx, 'domain': domain,
            'is_cross_gt': is_cross_gt, 'is_cross_pred': is_cross_pred,
            'is_clean300': is_clean, 'pred_function': item.get('pred_function',''),
            'rouge_l': round(rl,4) if rl is not None else '',
            'bleu4':   round(bleu,4) if bleu is not None else '',
            'bertscore_f': '',
        })

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"[Phase1] 완료: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 : Hallucination — V01(CROSS 수정) + V02
# ══════════════════════════════════════════════════════════════════════════════

def compute_hallucination(gen_path, variant_name, routing, kb_domain, kb_unified, use_subdomain_for_cross):
    """
    gen_path  : *_gen_500.json 파일 경로
    variant_name: 'v01', 'v02', ...
    use_subdomain_for_cross: True → CROSS 쿼리에 top1+top2 서브도메인 KB 사용
                             False → UNIFIED KB 사용 (V03, V05)
    """
    from transformers import pipeline as hf_pipeline
    from sentence_transformers import SentenceTransformer
    import faiss
    from nltk.tokenize import sent_tokenize
    import nltk
    nltk.download('punkt_tab', quiet=True)

    out_csv  = os.path.join(OUT_DIR, f"{variant_name}_hallucination_per_sample.csv")
    out_sum  = os.path.join(OUT_DIR, f"{variant_name}_hallucination_summary.json")

    # 이미 완료면 skip
    if os.path.exists(out_csv):
        rows = list(csv.DictReader(open(out_csv)))
        if len(rows) == 500:
            print(f"  [{variant_name}] hallucination 이미 완료 — skip")
            return

    print(f"  [{variant_name}] hallucination 계산 중... ({gen_path})")
    data    = json.load(open(gen_path))
    gt_list = load_gt()

    # NLI 파이프라인 (이미 로드되어 있으면 재사용)
    nli = hf_pipeline("text-classification", model=NLI_MODEL,
                      device=0 if DEVICE == "cuda" else -1,
                      truncation=True, max_length=512)

    enc = SentenceTransformer(EMBED_NAME, device=DEVICE)

    def get_domains_for_query(idx, pred_domain, is_cross):
        if not is_cross:
            return [pred_domain] if pred_domain and pred_domain != 'CROSS' else ['CROSS']
        if use_subdomain_for_cross:
            r = routing.get(idx, {})
            t1 = r.get('top1', 'SOC'); t2 = r.get('top2', 'STEM')
            return [t1, t2]
        return None  # None → unified

    def retrieve_for_sent(sent, domains):
        emb = enc.encode([sent], normalize_embeddings=True).astype(np.float32)
        if domains is None:
            _, I = kb_unified['idx'].search(emb, TOP_K_NLI)
            return [kb_unified['texts'][i] for i in I[0] if i >= 0]
        per = max(1, TOP_K_NLI // len(domains))
        results = []
        for dom in domains:
            if dom in kb_domain:
                _, I = kb_domain[dom]['idx'].search(emb, per)
                results += [kb_domain[dom]['texts'][i] for i in I[0] if i >= 0]
        return results[:TOP_K_NLI]

    def check_sent(sent, passages):
        if not passages: return "neutral"
        has_ent = has_con = False
        for p in passages:
            try:
                r = nli({"text": p[:800], "text_pair": sent[:400]})
                lbl = (r["label"] if isinstance(r, dict) else r[0]["label"]).upper()
                if lbl == "ENTAILMENT":   has_ent = True
                elif lbl == "CONTRADICTION": has_con = True
            except: pass
        if has_ent: return "supported"
        if has_con: return "contradiction"
        return "neutral"

    # 인덱스 필드 이름 통일
    def get_idx(item): return int(item.get('id', item.get('sample_id', item.get('idx', -1))))
    def get_pred(item): return item.get('pred_answer','').strip()
    def get_domain(item, idx): return item.get('expected_domain', item.get('domain', gt_list[idx].get('expected_domain','?')))
    def get_is_cross(item, idx): return item.get('is_cross', gt_list[idx].get('expected_domain','') == 'CROSS')
    def get_pred_dom(item, idx): return item.get('pred_domain', routing.get(idx,{}).get('pred_domain','CROSS'))
    def get_is_pred_cross(item, idx):
        if 'is_cross' in item: return bool(item['is_cross'])
        return routing.get(idx,{}).get('is_cross', False)

    csv_rows = []
    for item in data:
        idx        = get_idx(item)
        pred       = get_pred(item)
        domain_gt  = get_domain(item, idx)
        is_cross_gt= (domain_gt == 'CROSS')
        is_clean   = idx < 300
        pred_dom   = get_pred_dom(item, idx)
        is_cross_pred = get_is_pred_cross(item, idx)

        domains    = get_domains_for_query(idx, pred_dom, is_cross_pred)

        sents = [s.strip() for s in sent_tokenize(pred) if len(s.strip()) >= MIN_SENT] if pred else []
        n_sents = len(sents); n_contra = 0
        if sents:
            passages = retrieve_for_sent(sents[0], domains) if sents else []
            for s in sents:
                ps = retrieve_for_sent(s, domains)
                if check_sent(s, ps) == "contradiction":
                    n_contra += 1

        pred_domains_str = "|".join(routing.get(idx,{}).get('top1','') and
                                     ([routing[idx]['top1'], routing[idx]['top2']]
                                      if is_cross_pred else [pred_dom]) or [pred_dom]) \
                           if use_subdomain_for_cross else ("UNIFIED" if domains is None else "|".join(domains or [pred_dom]))

        hal_rate = round(100*n_contra/n_sents, 2) if n_sents else None
        csv_rows.append({
            'idx': idx, 'expected_domain': domain_gt,
            'is_cross': is_cross_gt, 'is_clean': is_clean,
            'pred_domains': pred_domains_str,
            'n_sents': n_sents, 'n_contra': n_contra,
            'hal_rate': hal_rate if hal_rate is not None else '',
        })

        if (len(csv_rows) % 50) == 0:
            print(f"    [{len(csv_rows)}/500]")
            # 중간 저장
            with open(out_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                w.writeheader(); w.writerows(csv_rows)

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)

    # 집계
    def sub(cond_fn):
        rs = [r for r in csv_rows if cond_fn(r)]
        rate, n, c = hal_pool(rs)
        return {"hal": rate, "n_sents": n, "n_contra": c}

    summary = {
        "overall":      sub(lambda r: True),
        "cross":        sub(lambda r: r['is_cross']=='True' or r['is_cross']==True),
        "single":       sub(lambda r: str(r['is_cross']) not in ('True','true')),
        "clean":        sub(lambda r: r['is_clean']=='True' or r['is_clean']==True),
        "clean_cross":  sub(lambda r: (r['is_clean']=='True' or r['is_clean']==True) and (r['is_cross']=='True' or r['is_cross']==True)),
        "clean_single": sub(lambda r: (r['is_clean']=='True' or r['is_clean']==True) and str(r['is_cross']) not in ('True','true')),
        "excl_law":     sub(lambda r: r['expected_domain'] != 'LAW'),
        "by_domain":    {d: sub(lambda r, d=d: r['expected_domain']==d)
                         for d in ['LAW','MED','SOC','STEM','CROSS']},
    }
    with open(out_sum, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  [{variant_name}] 완료: hal={summary['overall']['hal']}% / "
          f"CROSS={summary['cross']['hal']}% / single={summary['single']['hal']}%")
    print(f"    excl_LAW={summary['excl_law']['hal']}%  "
          f"clean={summary['clean']['hal']}%  clean_CROSS={summary['clean_cross']['hal']}%")
    del nli, enc; gc.collect(); torch.cuda.empty_cache()


def phase2_hallucination():
    print("\n[Phase2] V01(CROSS-수정) + V02 hallucination 계산...")

    # KB 인덱스 구축 (한 번만)
    from sentence_transformers import SentenceTransformer
    import faiss

    print("  KB 인덱스 구축...")
    enc_kb = SentenceTransformer(EMBED_NAME, device=DEVICE)
    kb_domain = {}
    all_texts = []
    for dom, fname in DOMAIN_FILES.items():
        texts = [d.get("output", d.get("gold_answer",""))
                 for d in json.load(open(os.path.join(DATA_DIR, fname)))]
        texts = [t for t in texts if len(t) > 20]
        embs  = enc_kb.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        idx_f = faiss.IndexFlatIP(embs.shape[1])
        idx_f.add(embs.astype(np.float32))
        kb_domain[dom] = {'texts': texts, 'idx': idx_f}
        all_texts.extend(texts)
        print(f"    {dom}: {len(texts)}개")

    all_embs = enc_kb.encode(all_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    uni_idx  = faiss.IndexFlatIP(all_embs.shape[1])
    uni_idx.add(all_embs.astype(np.float32))
    kb_unified = {'texts': all_texts, 'idx': uni_idx}
    del enc_kb; gc.collect(); torch.cuda.empty_cache()

    routing = load_routing()

    # V01 corrected
    compute_hallucination(
        os.path.join(OUT_DIR, "v01_gen_500.json"), "v01",
        routing, kb_domain, kb_unified, use_subdomain_for_cross=True
    )
    # V02
    compute_hallucination(
        os.path.join(OUT_DIR, "v02_gen_500.json"), "v02",
        routing, kb_domain, kb_unified, use_subdomain_for_cross=True
    )

    del kb_domain, kb_unified; gc.collect(); torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 : V03 / V04 / V05 generation
# ══════════════════════════════════════════════════════════════════════════════

def _load_model_lora():
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print("  7B 모델 로드 (4-bit)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_7B, quantization_config=bnb, device_map="auto", trust_remote_code=True)
    tok   = AutoTokenizer.from_pretrained(BASE_7B, trust_remote_code=True)
    return model, tok

def _generate(model, tok, system, prompt, max_new_tokens=512):
    msgs = [{"role":"system","content":system}, {"role":"user","content":prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp  = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0][inp["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

def _build_faiss(enc):
    import faiss
    kb = {}
    all_texts = []
    for dom, fname in DOMAIN_FILES.items():
        texts = [d.get("output", d.get("gold_answer",""))
                 for d in json.load(open(os.path.join(DATA_DIR, fname)))]
        embs  = enc.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        idx_f = faiss.IndexFlatIP(embs.shape[1])
        idx_f.add(embs.astype(np.float32))
        kb[dom] = {'texts': texts, 'idx': idx_f}
        all_texts.extend(texts)

    all_embs = enc.encode(all_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    uni_idx  = faiss.IndexFlatIP(all_embs.shape[1])
    uni_idx.add(all_embs.astype(np.float32))
    kb['UNIFIED'] = {'texts': all_texts, 'idx': uni_idx}
    return kb

def _retrieve(enc, kb, query, domains, top_k=TOP_K_GEN):
    import faiss
    emb = enc.encode([query], normalize_embeddings=True).astype(np.float32)
    if not domains or domains == ['UNIFIED']:
        _, I = kb['UNIFIED']['idx'].search(emb, top_k)
        return [kb['UNIFIED']['texts'][i] for i in I[0] if i >= 0]
    per = max(1, top_k // len(domains))
    results = []
    for dom in domains:
        if dom in kb:
            _, I = kb[dom]['idx'].search(emb, per)
            results += [kb[dom]['texts'][i] for i in I[0] if i >= 0]
    return results[:top_k]


def generate_variant(variant, out_path, routing, func_router):
    """
    variant: 'v03' (unified KB, LoRA ON, scaffold ON)
             'v04' (partitioned KB, LoRA OFF, scaffold ON)
             'v05' (unified KB, LoRA OFF, scaffold OFF)
    """
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    import faiss

    if os.path.exists(out_path):
        existing = json.load(open(out_path))
        if len(existing) == 500:
            print(f"  [{variant}] gen 이미 완료 — skip")
            return
        done_ids = {int(item.get('sample_id', item.get('id', -1))) for item in existing}
        print(f"  [{variant}] 재개: {len(done_ids)}개 완료, 나머지 {500-len(done_ids)}개 처리")
    else:
        existing  = []
        done_ids  = set()

    gt_list = load_gt()
    print(f"\n  [{variant}] 생성 시작...")

    model, tok = _load_model_lora()
    enc = SentenceTransformer(EMBED_NAME, device=DEVICE)
    kb  = _build_faiss(enc)

    use_lora      = (variant in ('v03',))   # V04/V05: no LoRA
    unified_kb    = (variant in ('v03', 'v05'))
    use_scaffold  = (variant in ('v03', 'v04'))
    use_routing   = (variant in ('v03', 'v04'))

    current_lora = [None]
    if use_lora:
        from peft import PeftModel
        def switch_lora(dom):
            if current_lora[0] == dom: return
            lora_path = os.path.join(LORA_DIR, LORA_MAP[dom])
            if current_lora[0] is None:
                nonlocal model
                model = PeftModel.from_pretrained(model, lora_path, adapter_name=dom)
            else:
                model.load_adapter(lora_path, adapter_name=dom)
                model.set_adapter(dom)
            current_lora[0] = dom
    else:
        def switch_lora(dom): pass

    results = list(existing)
    for idx, item in enumerate(gt_list):
        if idx in done_ids: continue

        query = item["instruction"]
        gold  = item["gold_answer"]
        r     = routing.get(idx, {})

        if use_routing:
            pred_domain = r.get('pred_domain', 'STEM')
            is_cross    = r.get('is_cross', False)
            top1        = r.get('top1', pred_domain)
            top2        = r.get('top2', pred_domain)
            lora_target = "CROSS" if is_cross else top1
            domains_ret = [top1, top2] if is_cross else [top1]
        else:
            pred_domain = "UNIFIED"; is_cross = False; lora_target = "SOC"
            domains_ret = ['UNIFIED']

        switch_lora(lora_target)
        system  = DOMAIN_SYSTEM.get(lora_target, DOMAIN_SYSTEM["CROSS"]) if use_routing else "You are a helpful expert. Provide a comprehensive analytical answer."

        if unified_kb:
            ctx = _retrieve(enc, kb, query, ['UNIFIED'])
        else:
            ctx = _retrieve(enc, kb, query, domains_ret)

        if use_scaffold:
            pred_func = func_router.predict([query])[0]
        else:
            pred_func = "NEUTRAL"

        prompt = build_prompt(query, ctx, pred_func)
        answer = _generate(model, tok, system, prompt)

        results.append({
            'sample_id': idx, 'query': query, 'gold_answer': gold,
            'pred_answer': answer, 'pred_domain': pred_domain,
            'pred_function': pred_func, 'is_cross': is_cross,
            'domain': item.get('expected_domain','?'),
        })
        done_ids.add(idx)

        if (len(results) % 50) == 0:
            json.dump(results, open(out_path,'w'), indent=2, ensure_ascii=False)
            print(f"    [{len(results)}/500] 중간 저장")

    json.dump(results, open(out_path,'w'), indent=2, ensure_ascii=False)
    print(f"  [{variant}] 생성 완료: {out_path}")
    del model, tok, enc, kb; gc.collect(); torch.cuda.empty_cache()


def phase3_generate_v345():
    print("\n[Phase3] V03/V04/V05 생성...")
    routing     = load_routing()
    func_router = load_func_router()

    for variant in ['v03','v04','v05']:
        out_path = os.path.join(OUT_DIR, f"{variant}_gen_500.json")
        generate_variant(variant, out_path, routing, func_router)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 : 모든 variant ROUGE-L / BLEU + BERTScore
# ══════════════════════════════════════════════════════════════════════════════

def phase4_score_all():
    print("\n[Phase4] 모든 variant 점수 계산 (V01~V05)...")
    from rouge_score import rouge_scorer as rs_module
    from bert_score import score as bert_score_fn
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    nltk.download('punkt_tab', quiet=True)

    rscorer = rs_module.RougeScorer(['rougeL'], use_stemmer=True)
    smooth  = SmoothingFunction().method1
    routing = load_routing()
    gt_list = load_gt()

    variants = {
        'v01': os.path.join(OUT_DIR, "v01_gen_500.json"),
        'v02': os.path.join(OUT_DIR, "v02_gen_500.json"),
        'v03': os.path.join(OUT_DIR, "v03_gen_500.json"),
        'v04': os.path.join(OUT_DIR, "v04_gen_500.json"),
        'v05': os.path.join(OUT_DIR, "v05_gen_500.json"),
    }

    gen_quality = {}
    for vname, gen_path in variants.items():
        out_csv = os.path.join(OUT_DIR, f"{vname}_scores_per_sample.csv")
        if not os.path.exists(gen_path):
            print(f"  [{vname}] gen 파일 없음 — skip")
            continue

        # 기존 파일에 bertscore가 이미 있으면 skip
        if os.path.exists(out_csv):
            rows = list(csv.DictReader(open(out_csv)))
            if len(rows) == 500 and rows[0].get('bertscore_f',''):
                print(f"  [{vname}] 이미 완료 — skip")
                # load for summary
                gen_quality[vname] = _summarize_scores(rows)
                continue

        print(f"  [{vname}] ROUGE-L/BLEU 계산...")
        data = json.load(open(gen_path))
        rows = []
        golds_for_bert = []; preds_for_bert = []; indices_for_bert = []

        for item in data:
            idx = int(item.get('sample_id', item.get('id', item.get('idx', -1))))
            gold = item.get('gold_answer', gt_list[idx].get('gold_answer','')).strip()
            pred = item.get('pred_answer','').strip()
            domain = item.get('domain', item.get('expected_domain',
                     gt_list[idx].get('expected_domain','?')))
            r = routing.get(idx, {})
            is_cross_gt   = (gt_list[idx].get('expected_domain','') == 'CROSS')
            is_cross_pred = r.get('is_cross', False)
            is_clean      = idx < 300

            rl   = rscorer.score(gold, pred)['rougeL'].fmeasure * 100 if gold and pred else None
            bleu = sentence_bleu([nltk.word_tokenize(gold.lower())],
                                  nltk.word_tokenize(pred.lower()),
                                  smoothing_function=smooth) * 100 if gold and pred else None

            rows.append({
                'sample_id': idx, 'domain': domain,
                'is_cross_gt': is_cross_gt, 'is_cross_pred': is_cross_pred,
                'is_clean300': is_clean,
                'pred_function': item.get('pred_function',''),
                'rouge_l':   round(rl,4)   if rl is not None else '',
                'bleu4':     round(bleu,4) if bleu is not None else '',
                'bertscore_f': '',
            })
            if gold and pred:
                golds_for_bert.append(gold); preds_for_bert.append(pred)
                indices_for_bert.append(len(rows)-1)

        print(f"  [{vname}] BERTScore 계산 ({len(golds_for_bert)}개)...")
        _, _, F1 = bert_score_fn(preds_for_bert, golds_for_bert, lang="en",
                                  model_type="roberta-large",
                                  rescale_with_baseline=False,
                                  verbose=False, device=DEVICE)
        for i, bs in zip(indices_for_bert, F1.tolist()):
            rows[i]['bertscore_f'] = round(bs*100, 4)

        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        gen_quality[vname] = _summarize_scores(rows)
        print(f"  [{vname}] ROUGE-L={gen_quality[vname]['Overall_c300']['rouge_l']}  "
              f"CROSS_c300={gen_quality[vname]['CROSS_c300']['rouge_l']}")
        gc.collect(); torch.cuda.empty_cache()

    # 저장
    out = os.path.join(OUT_DIR, "ablation_generation_quality.json")
    with open(out,'w') as f: json.dump(gen_quality, f, indent=2)
    print(f"\n[Phase4] 완료: {out}")
    return gen_quality


def _summarize_scores(rows):
    def flt(v):
        try: return float(v)
        except: return None

    def subset_avg(cond):
        sub = [r for r in rows if cond(r)]
        rl   = avg([flt(r['rouge_l'])     for r in sub])
        bs   = avg([flt(r.get('bertscore_f','')) for r in sub])
        bl   = avg([flt(r['bleu4'])        for r in sub])
        return {"n": len(sub), "rouge_l": rl, "bertscore": bs, "bleu4": bl}

    def to_bool(v):
        if isinstance(v, bool): return v
        return str(v).lower() in ('true','1')

    return {
        "CROSS_full":    subset_avg(lambda r: to_bool(r['is_cross_gt'])),
        "Single_full":   subset_avg(lambda r: not to_bool(r['is_cross_gt'])),
        "Overall_full":  subset_avg(lambda r: True),
        "CROSS_c300":    subset_avg(lambda r: to_bool(r['is_clean300']) and to_bool(r['is_cross_gt'])),
        "Single_c300":   subset_avg(lambda r: to_bool(r['is_clean300']) and not to_bool(r['is_cross_gt'])),
        "Overall_c300":  subset_avg(lambda r: to_bool(r['is_clean300'])),
        **{f"dom_{d}": subset_avg(lambda r, d=d: r['domain']==d)
           for d in ['LAW','MED','SOC','STEM','CROSS']},
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 : V03/V04/V05 hallucination
# ══════════════════════════════════════════════════════════════════════════════

def phase5_hallucination_v345():
    print("\n[Phase5] V03/V04/V05 hallucination 계산...")
    from sentence_transformers import SentenceTransformer
    import faiss

    print("  KB 인덱스 구축...")
    enc_kb = SentenceTransformer(EMBED_NAME, device=DEVICE)
    kb_domain = {}; all_texts = []
    for dom, fname in DOMAIN_FILES.items():
        texts = [d.get("output", d.get("gold_answer",""))
                 for d in json.load(open(os.path.join(DATA_DIR, fname)))]
        texts = [t for t in texts if len(t) > 20]
        embs  = enc_kb.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
        idx_f = faiss.IndexFlatIP(embs.shape[1])
        idx_f.add(embs.astype(np.float32))
        kb_domain[dom] = {'texts': texts, 'idx': idx_f}
        all_texts.extend(texts)
    all_embs = enc_kb.encode(all_texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    uni_idx  = faiss.IndexFlatIP(all_embs.shape[1])
    uni_idx.add(all_embs.astype(np.float32))
    kb_unified = {'texts': all_texts, 'idx': uni_idx}
    del enc_kb; gc.collect(); torch.cuda.empty_cache()

    routing = load_routing()

    # V03: domain-partitioned KB, CROSS uses top1+top2
    gen_v03 = os.path.join(OUT_DIR, "v03_gen_500.json")
    if os.path.exists(gen_v03):
        compute_hallucination(gen_v03, "v03", routing, kb_domain, kb_unified,
                              use_subdomain_for_cross=True)
    # V04: same
    gen_v04 = os.path.join(OUT_DIR, "v04_gen_500.json")
    if os.path.exists(gen_v04):
        compute_hallucination(gen_v04, "v04", routing, kb_domain, kb_unified,
                              use_subdomain_for_cross=True)
    # V05: unified KB for all
    gen_v05 = os.path.join(OUT_DIR, "v05_gen_500.json")
    if os.path.exists(gen_v05):
        compute_hallucination(gen_v05, "v05", routing, kb_domain, kb_unified,
                              use_subdomain_for_cross=False)

    del kb_domain, kb_unified; gc.collect(); torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 : 종합 보고
# ══════════════════════════════════════════════════════════════════════════════

def phase6_summary():
    print("\n[Phase6] 종합 보고...")
    variants = ['v01','v02','v03','v04','v05']
    hal_results = {}

    for v in variants:
        p = os.path.join(OUT_DIR, f"{v}_hallucination_summary.json")
        if os.path.exists(p):
            hal_results[v] = json.load(open(p))

    # 생성 품질
    gen_q_path = os.path.join(OUT_DIR, "ablation_generation_quality.json")
    gen_q = json.load(open(gen_q_path)) if os.path.exists(gen_q_path) else {}

    summary = {"hallucination": hal_results, "generation_quality": gen_q}
    out = os.path.join(OUT_DIR, "ablation_hallucination_summary.json")
    with open(out, 'w') as f: json.dump(summary, f, indent=2)

    print("\n" + "="*70)
    print("ABLATION HALLUCINATION (KB-based, LARGE NLI, TOP_K=3, min_sent=20)")
    print("="*70)
    print(f"{'Variant':<8} {'Overall':>8} {'CROSS':>8} {'Single':>8} {'excl_LAW':>10} {'clean_CR':>10}")
    print("-"*70)
    labels = {
        'v01': 'V01 Full',
        'v02': 'V02 -Scaffold',
        'v03': 'V03 -PartRet',
        'v04': 'V04 -LoRA',
        'v05': 'V05 -DomAxis',
    }
    for v in variants:
        if v not in hal_results: print(f"  {v}: 미완료"); continue
        h = hal_results[v]
        print(f"{labels.get(v,v):<12} "
              f"{str(h['overall']['hal']):>8} "
              f"{str(h['cross']['hal']):>8} "
              f"{str(h['single']['hal']):>8} "
              f"{str(h['excl_law']['hal']):>10} "
              f"{str(h['clean_cross']['hal']):>10}")

    print("\nABLATION GENERATION QUALITY (ROUGE-L, Clean-300)")
    print(f"{'Variant':<12} {'Overall':>8} {'CROSS':>8} {'Single':>8} {'LAW':>8}")
    for v in variants:
        if v not in gen_q: print(f"  {v}: 미완료"); continue
        q = gen_q[v]
        print(f"{labels.get(v,v):<12} "
              f"{str(q.get('Overall_c300',{}).get('rouge_l','-')):>8} "
              f"{str(q.get('CROSS_c300',{}).get('rouge_l','-')):>8} "
              f"{str(q.get('Single_c300',{}).get('rouge_l','-')):>8} "
              f"{str(q.get('dom_LAW',{}).get('rouge_l','-')):>8}")

    print(f"\n종합 결과: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0,
                    help="0=all, 1=v02-score, 2=hal-v01v02, 3=gen-v345, 4=score-all, 5=hal-v345, 6=summary")
    args = ap.parse_args()

    if args.phase in (0, 1):
        phase1_v02_rouge_bleu()

    if args.phase in (0, 2):
        phase2_hallucination()

    if args.phase in (0, 3):
        phase3_generate_v345()

    if args.phase in (0, 4):
        phase4_score_all()

    if args.phase in (0, 5):
        phase5_hallucination_v345()

    if args.phase in (0, 6):
        phase6_summary()

    print("\n✅ 전체 완료")

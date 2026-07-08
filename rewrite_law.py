import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE = "/home/sclab/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
DATA = "/home/sclab/moe_model/processed_v2/law.json"
OUT  = "/home/sclab/moe_model/processed_v2/law_rewritten.json"

def is_bad(text):
    return any(p in text for p in [
        "The correct holding", "All holdings:",
        "The correct answer", "All answers:",
        "Choices:", "Options:", "(A)", "(B)", "Subject:"
    ])

def parse_answer(text):
    m = re.search(r"The correct holding is option \d+:\s*(.+?)(?:\nAll|\Z)", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"The correct answer is [A-D]:\s*(.+?)(?:\n|$)", text)
    if m: return m.group(1).strip()
    return None

print("[*] 72B 로드...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        device_map="auto", trust_remote_code=True, local_files_only=True,
        attn_implementation="eager")
print("[*] 로드 완료!")

with open(DATA) as f: data = json.load(f)
results, rw, keep, skip = [], 0, 0, 0

for i, d in enumerate(data):
    q   = d.get("instruction","").strip()
    out = d.get("output","").strip()
    if not q or not out: skip+=1; continue

    if is_bad(out):
        answer = parse_answer(out)
        if not answer: skip+=1; continue
        prompt = (
            f"You are a legal expert.\n\n"
            f"Question: {q}\n\n"
            f"The correct legal holding/answer is: {answer}\n\n"
            f"Write a comprehensive explanatory paragraph (3-5 sentences) explaining "
            f"this legal principle with rationale and implications. "
            f"Write in flowing paragraph form only.\n\nAnswer:"
        )
        msgs = [{"role":"user","content":prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1500).to(model.device)
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=256, temperature=0.7,
                                  do_sample=True, pad_token_id=tokenizer.eos_token_id)
        new_out = tokenizer.decode(outs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        for kw in ["The correct holding","All holdings:","Choices:","\n0:","\n1:"]:
            if kw in new_out: new_out = new_out[:new_out.index(kw)].strip()
        if len(new_out.split()) >= 15:
            results.append({"domain":"LAW","instruction":q,"output":new_out,"rewritten":True})
            rw+=1
        else:
            skip+=1
    else:
        if len(out.split()) >= 15:
            results.append({"domain":"LAW","instruction":q,"output":out,"rewritten":False})
            keep+=1

    if (i+1) % 100 == 0:
        with open(OUT,"w") as f: json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[{i+1}/5000] 변환:{rw} 유지:{keep} 스킵:{skip}")

with open(OUT,"w") as f: json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nLAW 완료: 변환:{rw} 유지:{keep} 스킵:{skip} 최종:{len(results)}개")

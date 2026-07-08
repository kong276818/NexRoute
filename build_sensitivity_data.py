import os, json, random, torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DOMAINS      = ['LAW', 'MED', 'SOC', 'STEM']
DATA_DIR     = "/home/sclab/moe_model/processed_v2"
GT500_PATH   = "/home/sclab/moe_model/processed_v2/gt_dataset_500.json"
ROUTER_PATH  = "/home/sclab/moe_model/router_multilabel.pt"
OUT_PATH     = "/home/sclab/moe_model/sensitivity_validation_scores.json"


class DomainRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 4)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


def build_validation_set(n_per_domain=40, n_cross=40, seed=42):
    """GT500과 중복 없는 검증셋: 단일도메인 4*40 + CROSS 40 = 200개"""
    random.seed(seed)
    with open(GT500_PATH) as f:
        gt500 = json.load(f)
    gt500_instr = set(d['instruction'] for d in gt500)

    domain_files = {
        'LAW': 'law_rewritten.json', 'MED': 'med_rewritten.json',
        'SOC': 'soc_rewritten.json', 'STEM': 'stem_rewritten.json',
    }
    val_samples = []
    for domain, fname in domain_files.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        candidates = [d for d in data if d['instruction'] not in gt500_instr]
        n_pick = min(n_per_domain, len(candidates))
        picked = random.sample(candidates, n_pick)
        for d in picked:
            val_samples.append({
                'instruction': d['instruction'],
                'expected_domain': domain,
                'is_cross_gt': False
            })
        print(f"  {domain}: {n_pick}개 선택 (후보 {len(candidates)}개 중)")

    cross_path = os.path.join(DATA_DIR, 'cross_rewritten.json')
    with open(cross_path) as f:
        cross_data = json.load(f)
    cross_cands = [d for d in cross_data if d['instruction'] not in gt500_instr]
    n_cross_pick = min(n_cross, len(cross_cands))
    picked_cross = random.sample(cross_cands, n_cross_pick)
    for d in picked_cross:
        val_samples.append({
            'instruction': d['instruction'],
            'expected_domain': 'CROSS',
            'is_cross_gt': True
        })
    print(f"  CROSS: {n_cross_pick}개 선택 (후보 {len(cross_cands)}개 중)")

    random.shuffle(val_samples)
    print(f"[*] 검증셋 총 {len(val_samples)}개 구성 완료 (GT500과 0% 중복)")
    return val_samples


def compute_router_scores(val_samples):
    """라우터 forward pass 1회씩 → domain_scores만 저장 (생성 없음, 빠름)"""
    print("[*] 라우터 로드...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    router = DomainRouter()
    router.load_state_dict(torch.load(ROUTER_PATH, map_location='cpu'))
    router.eval()

    results = []
    for i, item in enumerate(val_samples):
        query = item['instruction']
        enc = tokenizer(
            query, max_length=256, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        with torch.no_grad():
            logits = router(enc['input_ids'], enc['attention_mask'])
            scores = torch.sigmoid(logits).squeeze().tolist()
        results.append({
            'instruction': query,
            'expected_domain': item['expected_domain'],
            'is_cross_gt': item['is_cross_gt'],
            'domain_scores': dict(zip(DOMAINS, scores))
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(val_samples)}] 처리완료")
    return results


if __name__ == "__main__":
    print("=== Step 1: 검증셋 구성 (GT500과 분리) ===")
    val_samples = build_validation_set(n_per_domain=40, n_cross=40)

    print("\n=== Step 2: 라우터 점수 계산 ===")
    scored = compute_router_scores(val_samples)

    with open(OUT_PATH, 'w') as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)
    print(f"\n[+] 저장 완료: {OUT_PATH} ({len(scored)}개)")
    print("[+] 다음 단계: sensitivity_heatmap.py 실행")
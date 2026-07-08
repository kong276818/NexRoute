from datasets import load_dataset, concatenate_datasets

SAVE_DIR = "/home/sclab/moe_model/datasets"

print("=" * 50)
print("데이터셋 로드 시작")
print(f"저장 위치: {SAVE_DIR}")
print("=" * 50)

# MED
print("\n[1/4] MED - PubMedQA 로딩 중...")
med = load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir=SAVE_DIR)
print(f"완료! train: {len(med['train'])}개")

# STEM
print("\n[2/4] STEM - SciQ 로딩 중...")
stem = load_dataset("allenai/sciq", cache_dir=SAVE_DIR)
print(f"완료! train: {len(stem['train'])}개")

# SOC
print("\n[3/4] SOC - CommonsenseQA 로딩 중...")
soc = load_dataset("tau/commonsense_qa", cache_dir=SAVE_DIR)
print(f"완료! train: {len(soc['train'])}개")

# LAW - LegalBench 여러 태스크 합치기
print("\n[4/4] LAW - LegalBench 태스크 합치기...")
tasks = [
    "contract_nli_confidentiality_of_agreement",
    "contract_nli_explicit_identification",
    "contract_nli_limited_use",
    "contract_nli_no_licensing",
    "contract_nli_permissible_copy",
    "hearsay",
    "overruling",
    "successor_liability",
]

law_splits = []
for task in tasks:
    ds = load_dataset("nguha/legalbench", task, cache_dir=SAVE_DIR)
    law_splits.append(ds["test"])
    print(f"  {task}: {len(ds['test'])}개")

law = concatenate_datasets(law_splits)
print(f"완료! LAW 총 샘플: {len(law)}개")

print("\n" + "=" * 50)
print("모든 데이터셋 로드 완료!")
print("=" * 50)
print(f"MED  (PubMedQA)      train: {len(med['train'])}")
print(f"STEM (SciQ)          train: {len(stem['train'])}")
print(f"SOC  (CommonsenseQA) train: {len(soc['train'])}")
print(f"LAW  (LegalBench)    total: {len(law)}")
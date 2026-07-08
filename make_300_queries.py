import os
import json
import random

# 데이터가 있는 폴더와 생성할 파일 이름
DATA_DIR = "/home/sclab/moe_model/processed_v2"
OUTPUT_FILE = "eval_queries.json"

# 추출할 도메인 파일들
DOMAINS = ["med.json", "law.json", "stem.json", "soc.json", "cross.json"]
TOTAL_SAMPLES = 300

queries = []
samples_per_domain = TOTAL_SAMPLES // len(DOMAINS)

current_id = 1
for domain_file in DOMAINS:
    path = os.path.join(DATA_DIR, domain_file)
    if not os.path.exists(path):
        print(f"[-] 경고: {path} 파일이 없습니다. 건너뜁니다.")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 수정됨: instruction과 output(진짜 정답)을 함께 추출
    valid_data = [item for item in data if "instruction" in item and "output" in item]
    sampled = random.sample(valid_data, min(samples_per_domain, len(valid_data)))

    for item in sampled:
        queries.append({
            "id": current_id,
            "query": item["instruction"],
            "gold_answer": item["output"], # 여기에 진짜 정답을 저장!
            "expected_domain": domain_file.replace(".json", "").upper() # 원래 어느 도메인 출신인지도 저장
        })
        current_id += 1

random.shuffle(queries)

for idx, item in enumerate(queries):
    item["id"] = idx + 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(queries, f, ensure_ascii=False, indent=4)

print(f"\n[+] 성공! 총 {len(queries)}개의 질문과 진짜 정답(Gold Answer)이 '{OUTPUT_FILE}'에 저장되었습니다.")
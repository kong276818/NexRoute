import json
import os

def load_existing_results(folder_path):
    all_data = []
    # 결과가 들어있는 폴더 스캔
    for fname in os.listdir(folder_path):
        if fname.startswith("eval_") and fname.endswith(".json"):
            with open(os.path.join(folder_path, fname), "r") as f:
                try:
                    data = json.load(f)
                    # 리스트 형식이든 dict["results"] 형식이든 모두 처리
                    results = data.get("results", data) if isinstance(data, dict) else data
                    
                    for item in results:
                        # 키 매핑: true_domain 또는 domain 중 있는 것 사용
                        t_dom = item.get("true_domain") or item.get("domain") or "UNKNOWN"
                        # 라우팅 결과 키 매핑: routed_domain 또는 predicted_domain
                        r_dom = item.get("routed_domain") or item.get("predicted_domain") or "UNKNOWN"
                        
                        all_data.append({
                            "true_domain": t_dom,
                            "routed_domain": r_dom
                        })
                except Exception as e:
                    print(f"파일 읽기 오류 ({fname}): {e}")
    return all_data

def analyze_accuracy(data):
    total = len(data)
    if total == 0: return 0
    correct = 0
    for item in data:
        # 정답 도메인과 라우팅된 도메인이 정확히 일치하는지 확인
        if item["true_domain"] == item["routed_domain"]:
            correct += 1
    return correct / total

if __name__ == "__main__":
    folder = "/home/sclab/moe_model/results/"
    data = load_existing_results(folder)
    
    acc = analyze_accuracy(data)
    
    print(f"\n{'='*50}")
    print(f"분석 대상 쿼리 수: {len(data)}개")
    print(f"최종 라우팅 정확도 (Accuracy): {acc * 100:.2f}%")
    print(f"{'='*50}\n")
    
    # 팁: 정확도가 낮게 나온다면 어떤 도메인에서 많이 틀리는지 확인해보세요
    from collections import Counter
    print("[상세 분석: 오답이 많은 도메인]")
    errors = [d['true_domain'] for d in data if d['true_domain'] != d['routed_domain']]
    print(Counter(errors).most_common(5))
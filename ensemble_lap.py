"""
Ensemble 四個模型的輸出：qwen.jsonl, gemma3.jsonl, robertwwm.jsonl, llama.jsonl
策略：
1. 合併所有模型的 Quadruplet
2. 對於相同的 (Aspect, Category, Opinion) 組合，根據模型權重加權平均 VA 值
3. 使用加權投票機制：權重總和 >= threshold 才保留

模型權重（根據 CF1 分數）：
- qwen:     0.5848
- roberta:  0.5673
- gemma:    0.5501
- llama:    0.5413
"""

import json
from collections import defaultdict
import os

# 輸入檔案路徑與對應權重（根據 CF1 分數）
model_configs = [
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/bert_base_chinese_lap.jsonl",      "weight": 0.38, "name": "A"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/gemini_lap.jsonl",      "weight": 0.25, "name": "B"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/gemma_lap.jsonl",      "weight": 0.40, "name": "C"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/gemma2_lap.jsonl",      "weight": 0.40, "name": "D"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/gpt_lap.jsonl",      "weight": 0.30, "name": "E"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/llama_lap1.jsonl",      "weight": 0.33, "name": "F"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/llama_lap2.jsonl",      "weight": 0.33, "name": "G"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/llama_lap3.jsonl",      "weight": 0.33, "name": "H"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/qwen_lap1.jsonl",      "weight": 0.40, "name": "I"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/qwen_lap2.jsonl",      "weight": 0.40, "name": "J"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/qwen_lap3.jsonl",      "weight": 0.40, "name": "K"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/qwen_lap4.jsonl",      "weight": 0.40, "name": "L"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/qwen_lap5.jsonl",      "weight": 0.40, "name": "M"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/qwen_lap6.jsonl",      "weight": 0.40, "name": "N"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/roberta_large1_lap.jsonl",      "weight": 0.42, "name": "O"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/roberta_large2_lap.jsonl",      "weight": 0.42, "name": "P"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/roberta-wwm-ext-lap1.jsonl",      "weight": 0.39, "name": "Q"},
    {"file": "/mnt/usr1/ethan1017/NLP/new/result/roberta-wwm-ext-lap2.jsonl",      "weight": 0.39, "name": "R"},
]


# 正規化權重
total_weight = sum(cfg["weight"] for cfg in model_configs)
for cfg in model_configs:
    cfg["norm_weight"] = cfg["weight"] / total_weight

# 投票門檻：降低門檻以提高 Recall
# 0.48 = 需要至少 2 個模型同意 (最低兩模型和約為 0.486)
# 0.50 = 舊門檻
VOTE_THRESHOLD = 0.3421

# 輸出檔案
output_file = "/mnt/usr1/ethan1017/NLP/new/result/newenlap2.jsonl"

def load_jsonl(filepath):
    """載入 JSONL 檔案"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data[item['ID']] = item
    return data

def parse_va(va_str):
    """解析 VA 字串為 (valence, arousal) tuple"""
    try:
        parts = va_str.split('#')
        return float(parts[0]), float(parts[1])
    except:
        return None, None

def format_va(valence, arousal):
    """格式化 VA 值"""
    return f"{valence:.2f}#{arousal:.2f}"

def get_quad_key(quad):
    """取得 quadruplet 的 key (用於比對)"""
    aspect = quad.get('Aspect', '').strip().lower()
    category = quad.get('Category', '').strip().upper()
    opinion = quad.get('Opinion', '').strip().lower()
    return (aspect, category, opinion)

def ensemble_quadruplets_weighted(quad_lists_with_weights, vote_threshold=0.5):
    """
    合併多個模型的 quadruplets（加權版本）
    """
    quad_votes = defaultdict(list)
    quad_original = {} 
    
    for quads, weight, name in quad_lists_with_weights:
        if quads is None or not isinstance(quads, list):
            continue
            
        for quad in quads:
            # --- 新增安全性檢查 ---
            if not isinstance(quad, dict):
                print(f"警告：模型 {name} 輸出了非字典格式的 Quadruplet: {quad} (型別: {type(quad)})")
                continue # 跳過非字典格式的數據
            # ----------------------

            key = get_quad_key(quad)
            va = quad.get('VA', '')
            v, a = parse_va(va)
            if v is not None and a is not None:
                quad_votes[key].append((v, a, weight))
                if key not in quad_original:
                    quad_original[key] = {
                        'Aspect': quad.get('Aspect', ''),
                        'Category': quad.get('Category', ''),
                        'Opinion': quad.get('Opinion', ''),
                    }
    
    result = []
    for key, va_list in quad_votes.items():
        total_weight = sum(w for v, a, w in va_list)
        
        if total_weight >= vote_threshold:
            # 注意：這裡使用加權平均而非簡單平均會更精準
            # 加權平均公式：sum(value * weight) / sum(weight)
            avg_valence = sum(v * w for v, a, w in va_list) / total_weight
            avg_arousal = sum(a * w for v, a, w in va_list) / total_weight
            
            quad = quad_original[key].copy()
            quad['VA'] = format_va(avg_valence, avg_arousal)
            result.append(quad)
    
    return result

def main():
    # 載入所有模型的預測結果
    print("載入模型預測結果...")
    print("\n模型權重設定（根據 CF1 分數）：")
    all_data = []
    for cfg in model_configs:
        filepath = cfg["file"]
        if os.path.exists(filepath):
            data = load_jsonl(filepath)
            all_data.append((data, cfg["norm_weight"], cfg["name"]))
            print(f"  ✅ {cfg['name']:10s}: CF1={cfg['weight']:.4f}, 正規化權重={cfg['norm_weight']:.4f}, {len(data)} 筆")
        else:
            print(f"  ❌ {filepath}: 檔案不存在")
            all_data.append(({}, 0, cfg["name"]))
    
    print(f"\n投票門檻: {VOTE_THRESHOLD} (權重總和需 >= {VOTE_THRESHOLD})")
    
    # 取得所有 ID
    all_ids = set()
    for data, weight, name in all_data:
        all_ids.update(data.keys())
    all_ids = sorted(all_ids)
    print(f"共有 {len(all_ids)} 個樣本")
    
    # Ensemble
    print("\n開始加權 Ensemble...")
    results = []
    
    stats = {'total': 0, 'with_quads': 0, 'empty': 0}
    
    for sample_id in all_ids:
        # 收集各模型對此樣本的預測（帶權重）
        quad_lists_with_weights = []
        text = None
        for data, weight, name in all_data:
            if sample_id in data:
                quads = data[sample_id].get('Quadruplet', [])
                quad_lists_with_weights.append((quads, weight, name))
                if text is None and 'Text' in data[sample_id]:
                    text = data[sample_id]['Text']
            else:
                quad_lists_with_weights.append(([], weight, name))
        
        # 加權 Ensemble
        ensemble_quads = ensemble_quadruplets_weighted(quad_lists_with_weights, vote_threshold=VOTE_THRESHOLD)
        
        result = {
            'ID': sample_id,
            'Quadruplet': ensemble_quads
        }
        if text:
            result['Text'] = text
        
        results.append(result)
        
        stats['total'] += 1
        if len(ensemble_quads) > 0:
            stats['with_quads'] += 1
        else:
            stats['empty'] += 1
    
    # 儲存結果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 重新命名並壓縮
    import shutil
    import zipfile
    
    # 定義新的檔案名稱
    new_filename = "pred_zho_laptop.jsonl"
    new_filepath = os.path.join(os.path.dirname(output_file), new_filename)
    
    # 複製/重新命名檔案
    shutil.copy(output_file, new_filepath)
    
    # 建立壓縮檔
    zip_filename = output_file.replace('.jsonl', '.zip')
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(new_filepath, arcname=new_filename)
        
    # 刪除暫存的重新命名檔案 (選擇性，這裡保留以便檢查)
    # os.remove(new_filepath)

    print(f"\n✅ 加權 Ensemble 完成！")
    print(f"   輸出檔案: {output_file}")
    print(f"   重新命名檔案: {new_filepath}")
    print(f"   壓縮檔案: {zip_filename}")
    print(f"   總樣本數: {stats['total']}")
    print(f"   有 Quadruplet: {stats['with_quads']}")
    print(f"   空 Quadruplet: {stats['empty']}")
    
    # 顯示前幾個範例
    print("\n=== 前 3 個範例 ===")
    for i, item in enumerate(results[:3]):
        print(f"\n[{item['ID']}]")
        for q in item['Quadruplet']:
            print(f"  ({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})")

if __name__ == "__main__":
    main()

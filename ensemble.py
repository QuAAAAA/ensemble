"""
Ensemble output of four models: qwen.jsonl, gemma3.jsonl, robertwwm.jsonl, llama.jsonl
Strategy:
1. Merge Quadruplets from all models
2. For the same (Aspect, Category, Opinion) combination, calculate the weighted average VA value based on model weights
3. Use a weighted voting mechanism: retain only if the sum of weights >= threshold
"""

import json
from collections import defaultdict
import os

# Input file paths and corresponding weights (based on CF1 score)
model_configs = [
    # qwen model
    {"file": "test/qwen32B-3e.jsonl",             "weight": 0.5852, "name": "qwen32-3e"},
    {"file": "test/qwen32B_best_loss.jsonl",      "weight": 0.5795, "name": "qwen32-loss"}, 
    {"file": "test/qwen32B_best_cF1.jsonl",       "weight": 0.5723, "name": "qwen32-cF1"},  
    {"file": "test/qwen14B-3e.jsonl",             "weight": 0.5848, "name": "qwen14-3e"}, 
    {"file": "test/qwen14B_best_loss.jsonl",      "weight": 0.5748, "name": "qwen14-loss"}, 
    # roberta model
    {"file": "test/robertwwm.jsonl",              "weight": 0.5673, "name": "roberta"},
    # gemma3 model
    {"file": "test/gemma-3e.jsonl",              "weight": 0.5501, "name": "gemma-3e"},    
    {"file": "test/gemma-4e.jsonl",              "weight": 0.5379, "name": "gemma-4e"},    
    {"file": "test/gemma-5e.jsonl",               "weight": 0.5375, "name": "gemma-5e"},   
    # llama model
    {"file": "test/llama-3e.jsonl",               "weight": 0.5413, "name": "llama-3e"}, 
    {"file": "test/llama_best_loss.jsonl",        "weight": 0.5313, "name": "llama-loss"}, 
    # Other closed-source models
    {"file": "test/gpt.jsonl",                    "weight": 0.2745, "name": "gpt"},
    {"file": "test/gemini.jsonl",                 "weight": 0.3516, "name": "gemini"},
]

# Normalize weights
total_weight = sum(cfg["weight"] for cfg in model_configs)
for cfg in model_configs:
    cfg["norm_weight"] = cfg["weight"] / total_weight

# Voting threshold: Lower the threshold to improve Recall
VOTE_THRESHOLD = 0.3421

# Output file
output_file = "output/ensemble_withClose.jsonl"

def load_jsonl(filepath):
    """Load JSONL file"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data[item['ID']] = item
    return data

def parse_va(va_str):
    """Parse VA string into (valence, arousal) tuple"""
    try:
        parts = va_str.split('#')
        return float(parts[0]), float(parts[1])
    except:
        return None, None

def format_va(valence, arousal):
    """Format VA value"""
    return f"{valence:.2f}#{arousal:.2f}"

def get_quad_key(quad):
    """Get quadruplet key (for comparison)"""
    aspect = quad.get('Aspect', '').strip().lower()
    category = quad.get('Category', '').strip().upper()
    opinion = quad.get('Opinion', '').strip().lower()
    return (aspect, category, opinion)

def ensemble_quadruplets_weighted(quad_lists_with_weights, vote_threshold=0.5):
    """
    Merge quadruplets from multiple models (weighted version)
    quad_lists_with_weights: [(quads, weight, name), ...]
    vote_threshold: Retain only if the sum of weights reaches this threshold
    """
    # Collect all quadruplets along with their VA values and weights
    quad_votes = defaultdict(list)  # key -> [(va_valence, va_arousal, weight), ...]
    quad_original = {}  # key -> original quadruplet (preserve original case)
    
    for quads, weight, name in quad_lists_with_weights:
        if quads is None:
            continue
        for quad in quads:
            key = get_quad_key(quad)
            va = quad.get('VA', '')
            v, a = parse_va(va)
            if v is not None and a is not None:
                quad_votes[key].append((v, a, weight))
                # Preserve original format
                if key not in quad_original:
                    quad_original[key] = {
                        'Aspect': quad.get('Aspect', ''),
                        'Category': quad.get('Category', ''),
                        'Opinion': quad.get('Opinion', ''),
                    }
    
    # Generate final quadruplets based on weighted voting results
    result = []
    for key, va_list in quad_votes.items():
        # Calculate sum of weights
        total_weight = sum(w for v, a, w in va_list)
        
        if total_weight >= vote_threshold:
            # Weighted average VA value
            weighted_valence = sum(v * w for v, a, w in va_list) / total_weight
            weighted_arousal = sum(a * w for v, a, w in va_list) / total_weight
            
            quad = quad_original[key].copy()
            quad['VA'] = format_va(weighted_valence, weighted_arousal)
            result.append(quad)
    
    return result

def main():
    # Load prediction results from all models
    print("Loading model prediction results...")
    print("\nModel weight settings (based on CF1 score):")
    all_data = []
    for cfg in model_configs:
        filepath = cfg["file"]
        if os.path.exists(filepath):
            data = load_jsonl(filepath)
            all_data.append((data, cfg["norm_weight"], cfg["name"]))
            print(f"  ✅ {cfg['name']:10s}: CF1={cfg['weight']:.4f}, Normalized Weight={cfg['norm_weight']:.4f}, {len(data)} items")
        else:
            print(f"  ❌ {filepath}: File does not exist")
            all_data.append(({}, 0, cfg["name"]))
    
    print(f"\nVoting Threshold: {VOTE_THRESHOLD} (Sum of weights must be >= {VOTE_THRESHOLD})")
    
    # Get all IDs
    all_ids = set()
    for data, weight, name in all_data:
        all_ids.update(data.keys())
    all_ids = sorted(all_ids)
    print(f"Total {len(all_ids)} samples")
    
    # Ensemble
    print("\nStarting weighted Ensemble...")
    results = []
    
    stats = {'total': 0, 'with_quads': 0, 'empty': 0}
    
    for sample_id in all_ids:
        # Collect predictions for this sample from each model (with weights)
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
        
        # Weighted Ensemble
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
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Rename and zip
    import shutil
    import zipfile
    
    # Define new filename
    new_filename = "pred_zho_restaurant.jsonl"
    new_filepath = os.path.join(os.path.dirname(output_file), new_filename)
    
    # Copy/Rename file
    shutil.copy(output_file, new_filepath)
    
    # Create zip file
    zip_filename = output_file.replace('.jsonl', '.zip')
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(new_filepath, arcname=new_filename)
        
    # Delete temporary renamed file
    os.remove(new_filepath)

    print(f"\n✅ Weighted Ensemble completed!")
    print(f"   Output file: {output_file}")
    print(f"   Renamed file: {new_filepath}")
    print(f"   Zip file: {zip_filename}")
    print(f"   Total samples: {stats['total']}")
    print(f"   With Quadruplet: {stats['with_quads']}")
    print(f"   Empty Quadruplet: {stats['empty']}")
    

if __name__ == "__main__":
    main()

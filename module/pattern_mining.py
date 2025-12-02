from mlxtend.frequent_patterns import apriori, association_rules
from tqdm import tqdm
from itertools import combinations
import pandas as pd

def direct_pattern_mining(data, target_col, max_items=3, min_confidence=0.3, min_support=0.005):
    """
    직접 패턴 마이닝 - apriori 없이 직접 계산
    복잡도를 대폭 줄인 버전
    """
    print(f"Direct pattern mining for {target_col}")
    
    feature_cols = [col for col in data.columns if col != target_col]
    target_support = data[target_col].sum() / len(data)
    total_samples = len(data)
    
    results = []
    
    # 1개 조합부터 시작
    print("Processing single items...")
    single_results = []
    for col in tqdm(feature_cols, desc="Single items"):
        x_count = (data[col] == 1).sum()
        xy_count = ((data[col] == 1) & (data[target_col] == 1)).sum()
        
        if xy_count < min_support * total_samples:
            continue
            
        x_support = x_count / total_samples
        xy_support = xy_count / total_samples
        confidence = xy_support / x_support if x_support > 0 else 0
        
        if confidence >= min_confidence:
            single_results.append({
                'antecedents': [col],
                'support': xy_support,
                'confidence': confidence,
                'lift': confidence / target_support,
                'count': xy_count
            })
    
    results.extend(single_results)
    print(f"Found {len(single_results)} single-item rules")
    
    # 2개 조합 (가장 중요한 부분)
    if len(feature_cols) > 1 and max_items >= 2:
        print("Processing pairs...")
        # 효율성을 위해 유망한 단일 항목들만 선택
        promising_cols = [r['antecedents'][0] for r in single_results[:50]]  # 상위 50개만
        
        pair_results = []
        for col1, col2 in tqdm(list(combinations(promising_cols, 2)), desc="Pairs"):
            x_mask = (data[col1] == 1) & (data[col2] == 1)
            x_count = x_mask.sum()
            xy_count = (x_mask & (data[target_col] == 1)).sum()
            
            if xy_count < min_support * total_samples:
                continue
                
            x_support = x_count / total_samples
            xy_support = xy_count / total_samples
            confidence = xy_support / x_support if x_support > 0 else 0
            
            if confidence >= min_confidence:
                pair_results.append({
                    'antecedents': [col1, col2],
                    'support': xy_support,
                    'confidence': confidence,
                    'lift': confidence / target_support,
                    'count': xy_count
                })
        
        results.extend(pair_results)
        print(f"Found {len(pair_results)} pair rules")
    
    # 3개 조합 (선택적)
    if max_items >= 3 and len([r for r in results if len(r['antecedents']) == 2]) > 0:
        print("Processing triplets...")
        # 가장 유망한 pair들에서만 확장
        promising_pairs = sorted([r for r in results if len(r['antecedents']) == 2], 
                                key=lambda x: x['confidence'], reverse=True)[:20]
        
        triplet_results = []
        processed = set()
        
        for pair_rule in promising_pairs:
            pair_cols = pair_rule['antecedents']
            for col in promising_cols:
                if col in pair_cols:
                    continue
                    
                triplet = tuple(sorted(pair_cols + [col]))
                if triplet in processed:
                    continue
                processed.add(triplet)
                
                x_mask = (data[list(triplet)] == 1).all(axis=1)
                x_count = x_mask.sum()
                xy_count = (x_mask & (data[target_col] == 1)).sum()
                
                if xy_count < min_support * total_samples:
                    continue
                    
                x_support = x_count / total_samples
                xy_support = xy_count / total_samples
                confidence = xy_support / x_support if x_support > 0 else 0
                
                if confidence >= min_confidence:
                    triplet_results.append({
                        'antecedents': list(triplet),
                        'support': xy_support,
                        'confidence': confidence,
                        'lift': confidence / target_support,
                        'count': xy_count
                    })
        
        results.extend(triplet_results)
        print(f"Found {len(triplet_results)} triplet rules")
    
    return pd.DataFrame(results).sort_values('confidence', ascending=False)


def stratified_sampling_analysis(data, target_col, sample_ratio=0.3, min_confidence=0.3):
    print(f"Stratified sampling analysis (ratio={sample_ratio})")
    
    target_1_data = data[data[target_col] == 1]
    target_0_data = data[data[target_col] == 0].sample(frac=sample_ratio, random_state=42)
    
    sampled_data = pd.concat([target_1_data, target_0_data]).reset_index(drop=True)
    print(f"Sampled data size: {len(sampled_data)} (original: {len(data)})")
    
    results = direct_pattern_mining(
        sampled_data, 
        target_col, 
        max_items=3, 
        min_confidence=min_confidence,
        min_support=0.001
    )
    
    validated_results = []

    for _, rule in tqdm(results.iterrows(), total=len(results), desc="Validation"):
        items = rule['antecedents']
        
        x_mask = (data[items] == 1).all(axis=1)
        xy_mask = x_mask & (data[target_col] == 1)
        
        x_support = x_mask.sum() / len(data)
        xy_support = xy_mask.sum() / len(data)
        confidence = xy_support / x_support if x_support > 0 else 0
        
        if confidence >= min_confidence:
            validated_results.append({
                'antecedents': items,
                'support': xy_support,
                'confidence': confidence,
                'lift': confidence / (data[target_col].sum() / len(data)),
                'count': xy_mask.sum(),
                'sampled_confidence': rule['confidence']
            })
    
    return pd.DataFrame(validated_results).sort_values('confidence', ascending=False)
import pandas as pd
from itertools import combinations

class AprioriRulesMiner:
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rules = []
        self.frequent_itemsets = {}

    def _binarize_features(self, df, numeric_bins=3):
        df_binary = pd.DataFrame(index=df.index)

        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                top_values = df[col].value_counts().head(5).index
                for val in top_values:
                    df_binary[f"{col}={val}"] = (df[col] == val).astype(int)
            else:
                try:
                    bins = pd.qcut(df[col], q=numeric_bins, duplicates='drop', labels=False)
                    for i in range(numeric_bins):
                        df_binary[f"{col}_bin{i}"] = (bins == i).astype(int)
                except:
                    median_val = df[col].median()
                    df_binary[f"{col}_low"] = (df[col] <= median_val).astype(int)
                    df_binary[f"{col}_high"] = (df[col] > median_val).astype(int)

        return df_binary

    def _calculate_support(self, itemset, transactions):
        mask = transactions[list(itemset)].all(axis=1)
        return mask.sum() / len(transactions)

    def _generate_candidates(self, prev_itemsets, k):
        candidates = []
        n = len(prev_itemsets)

        for i in range(n):
            for j in range(i + 1, n):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == k:
                    candidates.append(union)

        return list(set([frozenset(c) for c in candidates]))

    def fit(self, df, max_itemset_size=3):
        df_binary = self._binarize_features(df)

        single_items = [set([col]) for col in df_binary.columns]
        self.frequent_itemsets[1] = {'itemsets': [], 'supports': {}}

        for itemset in single_items:
            support = self._calculate_support(itemset, df_binary)
            if support >= self.min_support:
                self.frequent_itemsets[1]['itemsets'].append(itemset)
                self.frequent_itemsets[1]['supports'][itemset] = support

        for k in range(2, max_itemset_size + 1):
            if k-1 not in self.frequent_itemsets:
                break

            candidates = self._generate_candidates(self.frequent_itemsets[k-1]['itemsets'], k)
            frequent = []
            supports = {}

            for itemset in candidates:
                support = self._calculate_support(itemset, df_binary)
                if support >= self.min_support:
                    frequent.append(itemset)
                    supports[itemset] = support

            if frequent:
                self.frequent_itemsets[k] = {'itemsets': frequent, 'supports': supports}

        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue

            for itemset in self.frequent_itemsets[k]['itemsets']:
                itemset_support = self.frequent_itemsets[k]['supports'][itemset]

                for i in range(1, k):
                    for antecedent in combinations(itemset, i):
                        antecedent = set(antecedent)
                        consequent = itemset - antecedent

                        if len(antecedent) in self.frequent_itemsets:
                            ant_support = self.frequent_itemsets[len(antecedent)]['supports'].get(antecedent, 0)

                            if ant_support > 0:
                                confidence = itemset_support / ant_support

                                if confidence >= self.min_confidence:
                                    lift = confidence / (itemset_support / ant_support) if ant_support > 0 else 0

                                    self.rules.append({
                                        'antecedent': antecedent,
                                        'consequent': consequent,
                                        'support': itemset_support,
                                        'confidence': confidence,
                                        'lift': lift
                                    })

        self.rules = sorted(self.rules, key=lambda x: x['confidence'], reverse=True)
        return self

    def get_top_rules(self, n=5):
        return self.rules[:n]

    def validate_data(self, df, rules=None):
        if rules is None:
            rules = self.get_top_rules(5)

        df_binary = self._binarize_features(df)
        violations = []
        for rule in rules:
            antecedent_cols = [col for col in df_binary.columns if any(item in col for item in rule['antecedent'])]
            consequent_cols = [col for col in df_binary.columns if any(item in col for item in rule['consequent'])]

            if antecedent_cols and consequent_cols:
                ant_mask = df_binary[antecedent_cols].all(axis=1)
                cons_mask = df_binary[consequent_cols].all(axis=1)
                violated = ant_mask & ~cons_mask
                violation_rate = violated.sum() / ant_mask.sum() if ant_mask.sum() > 0 else 0

                violations.append({
                    'rule': f"{rule['antecedent']} => {rule['consequent']}",
                    'expected_confidence': rule['confidence'],
                    'actual_confidence': 1 - violation_rate,
                    'violations': int(violated.sum()),
                    'violation_rate': violation_rate
                })

        return {'total_rules_checked': len(violations), 'violations': violations}

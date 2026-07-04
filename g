"""Apriori frequent itemset and association rule mining for transaction data."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd

FrequentItemset = tuple[frozenset, float]
AssociationRule = tuple[frozenset, frozenset, float, float, float]


class AprioriMiner:
    """Mines frequent itemsets and association rules from invoice transactions."""

    def __init__(self) -> None:
        self.transactions: list[set[str]] = []

    @property
    def num_transactions(self) -> int:
        return len(self.transactions)

    def load_csv(self, path: str) -> tuple[int, int]:
        """Load and clean invoice data, grouping descriptions per invoice."""
        data = pd.read_csv(path, encoding="ISO-8859-1")
        data = self._clean(data)
        self.transactions = data.groupby("Invoice")["Description"].apply(set).tolist()
        unique_items = {item for t in self.transactions for item in t}
        return self.num_transactions, len(unique_items)

    @staticmethod
    def _clean(data: pd.DataFrame) -> pd.DataFrame:
        data = data[~data["Invoice"].astype(str).str.startswith("C")]
        data = data.dropna(subset=["Description"])
        return data[data["Description"].str.strip() != "POSTAGE"]

    def support(self, itemset: Iterable[str]) -> float:
        """Fraction of transactions containing every item in `itemset`."""
        itemset = set(itemset)
        if not self.transactions:
            return 0.0
        hits = sum(itemset <= t for t in self.transactions)
        return hits / self.num_transactions

    def run(self, min_support: float, min_confidence: float) -> tuple[list[FrequentItemset], list[AssociationRule]]:
        """Mine frequent itemsets and derive association rules above the given thresholds."""
        if not self.transactions:
            raise ValueError("No transactions loaded. Call load_csv() first.")
        frequent = self._frequent_itemsets(min_support)
        rules = self._rules_from(frequent, min_confidence)
        return frequent, rules

    def _frequent_itemsets(self, min_support: float) -> list[FrequentItemset]:
        items = {item for t in self.transactions for item in t}
        level = [frozenset([item]) for item in items]
        frequent = self._filter_by_support(level, min_support)

        size = 2
        current = [fs for fs, _ in frequent]
        while current:
            candidates = self._next_candidates(current, size)
            level_frequent = self._filter_by_support(candidates, min_support)
            frequent.extend(level_frequent)
            current = [fs for fs, _ in level_frequent]
            size += 1
        return frequent

    def _filter_by_support(self, itemsets: list[frozenset], min_support: float) -> list[FrequentItemset]:
        scored = ((fs, self.support(fs)) for fs in itemsets)
        return [(fs, sup) for fs, sup in scored if sup >= min_support]

    @staticmethod
    def _next_candidates(current: list[frozenset], size: int) -> list[frozenset]:
        merged = {current[i] | current[j] for i in range(len(current)) for j in range(i + 1, len(current))}
        return [fs for fs in merged if len(fs) == size]

    def _rules_from(self, frequent: list[FrequentItemset], min_confidence: float) -> list[AssociationRule]:
        rules: list[AssociationRule] = []
        for itemset, sup in frequent:
            if len(itemset) < 2:
                continue
            rules.extend(self._rules_for_itemset(itemset, sup, min_confidence))
        return rules

    def _rules_for_itemset(self, itemset: frozenset, sup: float, min_confidence: float) -> list[AssociationRule]:
        rules: list[AssociationRule] = []
        for size in range(1, len(itemset)):
            for antecedent in combinations(itemset, size):
                rule = self._build_rule(itemset, frozenset(antecedent), sup, min_confidence)
                if rule:
                    rules.append(rule)
        return rules

    def _build_rule(
        self, itemset: frozenset, antecedent: frozenset, sup: float, min_confidence: float
    ) -> AssociationRule | None:
        consequent = itemset - antecedent
        confidence = sup / self.support(antecedent)
        if confidence < min_confidence:
            return None
        lift = confidence / self.support(consequent)
        return antecedent, consequent, sup, confidence, lift


def _group_by_size(frequent: list[FrequentItemset]) -> dict[int, list[FrequentItemset]]:
    groups: dict[int, list[FrequentItemset]] = {}
    for fs, sup in frequent:
        groups.setdefault(len(fs), []).append((fs, sup))
    return groups


def _print_itemsets(frequent: list[FrequentItemset], num_transactions: int, min_support: float) -> None:
    print("=" * 60)
    print(f"  FREQUENT ITEMSETS  (support >= {min_support})")
    print("=" * 60)
    for size, group in sorted(_group_by_size(frequent).items()):
        print(f"\n  -- Size {size} --")
        for fs, sup in sorted(group, key=lambda pair: pair[1], reverse=True):
            print(f"    {set(fs)}")
            print(f"      support = {sup:.4f}  ({int(sup * num_transactions)} transactions)")


def _print_rules(rules: list[AssociationRule], min_confidence: float) -> None:
    print("\n" + "=" * 60)
    print(f"  ASSOCIATION RULES  (confidence >= {min_confidence})")
    print("=" * 60)
    for antecedent, consequent, sup, conf, lift in sorted(rules, key=lambda r: r[3], reverse=True):
        print(f"\n    {set(antecedent)}  ->  {set(consequent)}")
        print(f"      conf = {conf:.4f}   sup = {sup:.4f}   lift = {lift:.4f}")


def main() -> None:
    path = input("Enter CSV file path: ").strip()
    min_support = float(input("Enter minimum support: "))
    min_confidence = float(input("Enter minimum confidence: "))

    miner = AprioriMiner()
    n_transactions, n_items = miner.load_csv(path)
    print(f"\nLoaded {n_transactions:,} transactions  |  {n_items:,} unique items\n")

    frequent, rules = miner.run(min_support, min_confidence)
    _print_itemsets(frequent, miner.num_transactions, min_support)
    _print_rules(rules, min_confidence)
    print(f"\n  Total: {len(frequent)} itemsets,  {len(rules)} rules")


if __name__ == "__main__":
    main()

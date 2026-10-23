"""Outcome-independent population rule approved before effect computation."""
from collections import defaultdict, deque
import hashlib


def choose_population(records, count):
    groups = defaultdict(list)
    for record in records:
        groups[record["template"]].append(record)
    queues = [deque(sorted(groups[template], key=lambda r: (
        hashlib.sha256(r["record_id"].encode()).hexdigest(), r["record_id"])))
        for template in sorted(groups)]
    selected = []
    while queues and len(selected) < count:
        for group in queues:
            if group and len(selected) < count:
                selected.append(group.popleft())
        queues = [group for group in queues if group]
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible records; need {count}; do not change N silently")
    return selected

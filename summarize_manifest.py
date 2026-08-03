import csv
from collections import Counter, defaultdict
from pathlib import Path

rows = list(csv.DictReader(Path('data/nist_images/manifest.csv').open(encoding='utf-8')))
print('rows', len(rows))
counts_by_split = Counter(row['split'] for row in rows)
counts_by_class = Counter(row['element'] for row in rows)
counts_by_split_class = defaultdict(Counter)
for row in rows:
    counts_by_split_class[row['split']][row['element']] += 1
print('split_counts', dict(counts_by_split))
print('class_counts', dict(counts_by_class))
for split in ['train', 'validation', 'test']:
    print(split, dict(counts_by_split_class[split]))

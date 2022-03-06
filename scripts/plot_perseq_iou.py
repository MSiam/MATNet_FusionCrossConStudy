import numpy as np
import os
import csv
import matplotlib.pyplot as plt

results_dirs = ['temp_fixclip', 'temp_attonly', 'temp_baseline']

ious = {}
seqs = []

for results_dir in results_dirs:
    with open(os.path.join(results_dir, 'MoCA_results.csv'), 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.split(',')
            sq_name, iou = tokens[:2]
            if sq_name not in seqs:
                seqs.append(sq_name)
            if results_dir not in ious:
                ious[results_dir] = []

            iou = float(iou) if float(iou) > 0 else 0
            ious[results_dir].append(iou)

colors = {'temp_fixclip': 'r', 'temp_attonly': 'b', 'temp_baseline': 'g'}
labels = {'temp_fixclip': 'Att + Warping', 'temp_attonly': 'Att', 'temp_baseline': 'Baseline'}

plt.figure(figsize=(100, 30), dpi=80)
plt.title('Per Sequence IoU MoCA')
plt.xlabel('Sequence')
plt.ylabel('IoU')

xs = np.arange(1, len(seqs)+1)
plt.xticks(xs, seqs, rotation=60)
temp = 0
for key, value in ious.items():
    plt.bar(xs+temp, value, width=0.2, color=colors[key], label=labels[key])
    temp += 0.2

plt.legend()

plt.savefig('moca_per_seq.png')


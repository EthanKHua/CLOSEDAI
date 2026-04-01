import csv
import numpy as np
from pred import predict_all

CSV_FILE = "ml_challenge_dataset.csv"

predictions = predict_all(CSV_FILE)

# true labels
true_labels = []
with open(CSV_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        true_labels.append(row["Painting"])

# accuracy
correct = sum(p == t for p, t in zip(predictions, true_labels))
total = len(true_labels)
print(f"Overall accuracy: {correct}/{total} = {correct/total:.4f}")

# per-class accuracy
for painting in ['The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond']:
    indices = [i for i, t in enumerate(true_labels) if t == painting]
    class_correct = sum(predictions[i] == true_labels[i] for i in indices)
    print(f"  {painting}: {class_correct}/{len(indices)} = {class_correct/len(indices):.4f}")
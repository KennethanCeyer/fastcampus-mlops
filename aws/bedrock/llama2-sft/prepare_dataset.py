import json
from datasets import load_dataset

dataset = load_dataset('sebastiaan/test-cefr')
output_data = []

for entry in dataset['train']:
    prompt = entry['prompt']
    completion = entry['label']
    output_data.append({
        "prompt": prompt,
        "completion": completion
    })

with open('output.jsonl', 'w') as outfile:
    for entry in output_data:
        json.dump(entry, outfile)
        outfile.write('\n')

print("Data has been converted and saved to output.jsonl")


import json
import os
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
json.dump({'data': data}, open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
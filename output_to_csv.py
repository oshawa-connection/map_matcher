import ast
import csv

input_file = "out_initial.txt"
output_file = "out.csv"

rows = []
with open(input_file, "r") as f:
    for line in f:
        if not line.strip():
            continue
        dict_part, value_part = line.strip().rsplit("},", 1)
        dict_part += "}"
        params = ast.literal_eval(dict_part)
        score = float(value_part)
        params["score"] = score
        rows.append(params)

# Get all unique keys for header
header = sorted({k for row in rows for k in row.keys()})

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
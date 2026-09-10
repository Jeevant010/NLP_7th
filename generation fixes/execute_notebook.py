import nbformat
from nbclient import NotebookClient
import sys

nb_path = r"D:\Desktop\NLP_7th\Assignment1\Lab1_NLP.ipynb"
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=60, kernel_name="python3")
try:
    client.execute()
    print("SUCCESS: notebook executed")
except Exception as e:
    print("ERROR:", type(e).__name__, e)
    tb = client.temp_output_dir
    print("temp_dir:", tb)

# Write back regardless (with or without outputs)
nbformat.write(nb, nb_path)
print("Notebook saved.")

# Report cell output summary
code_cells = [c for c in nb.cells if c.cell_type == "code"]
print(f"Total code cells: {len(code_cells)}")
for i, c in enumerate(code_cells):
    n_out = len(c.get("outputs", []))
    exec_count = c.get("execution_count")
    print(f"  Cell {i+1}: execution_count={exec_count}, outputs={n_out}")

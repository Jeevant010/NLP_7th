import nbformat

nb = nbformat.read(r"D:\Desktop\NLP_7th\Assignment1\Lab1_NLP.ipynb", as_version=4)
for i, c in enumerate(nb.cells):
    print(f"--- Cell {i+1} ({c.cell_type}) ---")
    if c.cell_type == "code":
        print("SOURCE:")
        print(c.source)
        print("OUTPUTS:")
        for out in c.get("outputs", []):
            if out.output_type == "stream":
                print(out.text)
            elif out.output_type == "error":
                print("ERROR:", out.ename, out.evalue)
        print()
    else:
        print(c.source[:400])

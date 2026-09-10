import pdfplumber
import sys

pdf_path = r"D:\Desktop\7th Sem\NLP\Labs\Lab1 _ NLP.pdf"
out_path = r"D:\Desktop\NLP_7th\Assignment1\lab1_text.txt"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Pages: {len(pdf.pages)}", file=sys.stderr)
    all_text = []
    for i, page in enumerate(pdf.pages):
        text = page.extract_text(x_tolerance=2, y_tolerance=2, keep_blank_lines=True, use_text_flow=True)
        print(f"Page {i+1}: {len(text) if text else 0} chars", file=sys.stderr)
        if text:
            all_text.append(text)
    full = "\n\n---PAGE BREAK---\n\n".join(all_text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"Written to {out_path}, total {len(full)} chars", file=sys.stderr)

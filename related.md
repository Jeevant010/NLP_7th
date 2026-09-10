some important commands

python -c "
import pdfplumber
text = ''
with pdfplumber.open(r'D:\Desktop\7th Sem\NLP\Labs/Lab1 _ NLP.pdf') as pdf:
    for page in pdf.pages:
        text += page.extract_text() or ''
        text += '\n---PAGE BREAK---\n'
print(text)
" 2>&1.


continue

python -c "
import pdfplumber
with pdfplumber.open(r'D:\Desktop\7th Sem\NLP\Labs/Lab1 _ NLP.pdf') as pdf:
    print('Pages:', len(pdf.pages))
    page = pdf.pages[0]
    print('Page 1 objects:', len(page.objects))
    text = page.extract_text(x_tolerance=2, y_tolerance=2, keep_blank_lines=True, use_text_flow=True)
    print('TEXT LENGTH:', len(text) if text else 0)
    print(text[:5000] if text else 'NO TEXT EXTRACTED')
" 2>&1.


python -c "import numpy; print('numpy', numpy.__version__)" 2>&1; python -c "import sklearn; print('sklearn', sklearn.__version__)" 
2>&1; python -c "import nbformat; print('nbformat', nbformat.__version__)" 2>&1; python -c "import nltk; print('nltk', nltk.__version__)" 2>&1; jupyter --version 
2>&1 | head -3.

python -c "import sklearn; print('sklearn', sklearn.__version__)" 2>&1; python -c "import nbformat; print('nbformat', 
nbformat.__version__)" 2>&1.

python -c "import nbformat; print('nbformat OK', nbformat.__version__)" 2>&1; python -c "import scipy; print('scipy OK')" 2>&1.



python -c "import nbclient; print('nbclient OK')" 2>&1; jupyter nbconvert --version 2>&1; pip list 2>&1 | grep -i -E 
"nbclient|nbconvert|jupyter".

python -c "
import nbformat
from nbclient import NotebookClient
nb_path = r'D:\Desktop\NLP_7th\Assignment1\Lab1_NLP.ipynb'
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=60, kernel_name='python3')
client.execute()
nbformat.write(nb, nb_path)
print('Notebook executed successfully with outputs captured.')
" 2>&1.


python -c "
import nbformat
from nbclient import NotebookClient
nb_path = r'D:\Desktop\NLP_7th\Assignment1\Lab1_NLP.ipynb'
nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(nb, timeout=60, kernel_name='python3')
try:
    client.execute()
    print('SUCCESS')
except Exception as e:
    print('ERROR:', type(e).__name__, e)
nbformat.write(nb, nb_path)
" 2>&1; echo "---EXIT CODE: $?---".


python -c "
import nbformat
nb = nbformat.read(r'D:\Desktop\NLP_7th\Assignment1\Lab1_NLP.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    print(f'--- Cell {i+1} ({c.cell_type}) ---')
    if c.cell_type == 'code':
        print('SOURCE:')
        print(c.source)
        print('OUTPUTS:')
        for out in c.get('outputs', []):
            if out.output_type == 'stream':
                print(out.text)
            elif out.output_type == 'error':
                print('ERROR:', out.ename, out.evalue)
        print()
    else:
        print(c.source[:300])
" 2>&1.



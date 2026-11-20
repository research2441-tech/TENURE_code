"""Builds TENURE_Colab.ipynb embedding the source files, so it runs stand-alone on Colab (free tier)."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
src = {f: open(os.path.join(here, f)).read() for f in
       ['tenure_sim.py', 'run_experiments.py', 'detector_bench.py', 'analyze.py', 'fig_diagrams.py']}


def md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': t.splitlines(True)}


def code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': t.splitlines(True)}


cells = [md('# TENURE: handshake-aware cluster-head tenure, rekeying and trust-weighted routing for post-quantum WSNs\n\n'
            'This notebook reproduces every figure and table of the paper. With `QUICK = True` (default) it runs a reduced set of seeds '
            'in about 20-30 minutes on the free Colab CPU; set `QUICK = False` for the full protocol (30 seeds, several hours on 2 cores).'),
         code('QUICK = True   # set False for the full experimental protocol\n'
              'import os\nos.environ["QUICK"] = "1" if QUICK else "0"\nos.environ["TENURE_OUT"] = "results"\nos.environ["TENURE_FIG"] = "figures"\n'
              '!pip -q install numpy scipy matplotlib pillow')]
for f, t in src.items():
    cells.append(md(f'## `{f}`'))
    cells.append(code(f'%%writefile {f}\n' + t))
cells += [md('## Run the experiments (E1-E5, E10) and the detector benchmark (E8)'),
          code('!python run_experiments.py\n!python detector_bench.py'),
          md('## Analysis: statistics, figures and the numeric summary used in the manuscript'),
          code('!python analyze.py\n!python fig_diagrams.py'),
          code('from IPython.display import Image, display\nimport glob\nfor f in sorted(glob.glob("figures/*.png")):\n    print(f); display(Image(f, width=900))'),
          code('import json\nR = json.load(open("results/analysis.json"))\n'
               'for m in ["TENURE","FixedTenure-PQ","DEEC-PQ","PQ-LEACH","pqDTLS-LEACH","MH-RSA"]:\n'
               '    print(f"{m:16s} benign FND={R[f\'ben_{m}_FND\']:.0f} SL={R[f\'ben_{m}_SL\']:.0f} PDR={R[f\'ben_{m}_PDR\']:.3f}")')]
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                   'language_info': {'name': 'python'}, 'colab': {'provenance': []}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open(os.path.join(here, 'TENURE_Colab.ipynb'), 'w'), indent=1)
print('notebook written')

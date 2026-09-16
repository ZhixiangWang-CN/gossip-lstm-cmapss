"""Create a compact results archive to share for analysis."""
import argparse
from pathlib import Path
from datetime import datetime
import zipfile

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=root/'output')
p.add_argument('--figures',action='store_true')
a=p.parse_args()
target=root/('PdM_results_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
if not a.output.exists():raise SystemExit('Output folder does not exist. Run experiments first.')
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(a.output.rglob('*')):
        if not f.is_file() or f.suffix in ['.pt','.tmp','.zip']:continue
        if not a.figures and f.suffix in ['.png','.svg']:continue
        z.write(f,Path('results')/f.relative_to(a.output))
print(target)

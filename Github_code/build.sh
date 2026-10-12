#!/bin/bash
# Assemble thesis.tex from parts/ and compile three times.
cd "$(dirname "$0")"
python3 - <<'PY'
order=['00_preamble','01_begin','02_title','03_front','05_lists','10_ch1_ch2','20_ch3_part1','21_ch3_part2','30_ch4','40_ch5','50_ch6','60_ch7','70_ch8','75_ch9','80_references','90_appendices']
body=''.join(open(f'parts/{o}.tex').read().rstrip('\n')+'\n' for o in order)+'\n\\end{document}\n'
open('thesis.tex','w').write(body)
PY
for i in 1 2 3; do pdflatex -interaction=nonstopmode thesis.tex > build.log 2>&1 || { grep -n -E "^! " build.log | head; exit 1; }; done
grep -E "undefined|multiply" thesis.log | head
pdfinfo thesis.pdf | grep Pages

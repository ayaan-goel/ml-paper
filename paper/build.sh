#!/bin/sh
# Build paper/main.pdf with TinyTeX (pdflatex + bibtex).
cd "$(dirname "$0")" || exit 1
export PATH="$(cygpath -u "$APPDATA")/TinyTeX/bin/windows:$PATH"
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1
bibtex main >> build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1
grep -E "^!|LaTeX Warning: (Citation|Reference)|Overfull" build.log | sort | uniq -c
grep -o "Output written on main.pdf ([0-9]* pages" build.log | tail -1

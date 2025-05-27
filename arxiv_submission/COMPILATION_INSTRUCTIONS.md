# LaTeX Compilation Instructions for arXiv Submission

## Option 1: Local Compilation (after LaTeX installation completes)

Once the LaTeX installation is complete, compile the paper using:

```bash
cd arxiv_submission
latexmk -pdf -bibtex main.tex
```

Or manually:
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Option 2: Use Overleaf (Recommended for quick compilation)

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload all files from the `arxiv_submission` directory
3. Set `main.tex` as the main document
4. Compile online

## Option 3: Windows LaTeX Installation

Since you're on WSL2, you can also install LaTeX on Windows:

1. Download and install [MiKTeX](https://miktex.org/download) or [TeX Live](https://www.tug.org/texlive/acquire-netinstall.html)
2. Use TeXworks or TeXstudio to compile `main.tex`

## Option 4: Docker Container

Use a LaTeX Docker container:

```bash
docker run --rm -v $(pwd):/data texlive/texlive:latest \
  sh -c "cd /data && latexmk -pdf -bibtex main.tex"
```

## Required LaTeX Packages

The document uses these packages:
- Standard: article class, graphicx, amsmath, amssymb
- Bibliography: biblatex with biber backend
- Tables: booktabs, tabularx
- Algorithms: algorithm, algorithmic
- Hyperlinks: hyperref
- Colors: xcolor

## arXiv Submission Checklist

1. Compile successfully to PDF
2. Check all figures are included and referenced
3. Verify bibliography compiles correctly
4. Remove any absolute paths
5. Create a zip file with all source files:
   ```bash
   zip -r arxiv_submission.zip *.tex sections/ figures/ *.bib
   ```

## Troubleshooting

- If compilation fails due to missing packages, install them individually
- Check `main.log` for detailed error messages
- Ensure all figure files exist in the `figures/` directory
- Verify all `.tex` files in `sections/` are present
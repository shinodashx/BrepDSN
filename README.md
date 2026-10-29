# BRepDSN

### A Dual-stream Network Design for B-Rep Representation Learning Leveraging the Duality of Face and Boundary Vertices

Haoxuan Song*, Bingchen Yang*, Jun Xiao, Peter Wonka, Haiyong Jiang

*Equal contribution. Haiyong Jiang is the corresponding author and project lead.*

BRepDSN combines face-adjacency and vertex-adjacency graph features for boundary-aware B-Rep representation learning. The supplied manuscript evaluates modeling operation segmentation, operation step segmentation, B-Rep classification, machining feature segmentation, and assembly mating, with task-specific decoders and training.

![BRepDSN analysis results](docs/assets/teaser.webp)

## Project homepage

The static website is in [`docs/`](docs/), following the [CADRec homepage](https://shinodashx.github.io/CADRec/). It uses the same local Noto Sans font, typography, container widths, spacing, figure layout, responsive breakpoints, result tabs, and citation-copy interaction. The only additional CSS rule styles the unavailable paper button. There is no build step, package manager, external font request, or runtime dependency.

Preview with Python 3:

```bash
python3 -m http.server 8080 --directory docs
```

The site can be served at either a domain root or a subdirectory: all page assets, figures, and section links are relative. Publishing is separate from editing the page. For GitHub Pages, select the desired branch and `/docs` as the source in the repository's Pages settings, or copy the contents of `docs/` to the chosen path of an existing static site. No Pages settings or external deployment are changed by this repository.

### Content and assets

- [`docs/index.html`](docs/index.html): manuscript content, author details, figures, result table, and provisional BibTeX.
- [`docs/style.css`](docs/style.css): CADRec's original styling, with the disabled paper-button state.
- [`docs/script.js`](docs/script.js): CADRec's result tabs (including arrow, Home, and End keys) and citation copy, with manual-selection fallback when clipboard access is denied. Both result panels remain readable without JavaScript.
- [`docs/assets/`](docs/assets/): self-hosted font, font license, favicon, and 2600-pixel-wide WebP figure previews generated from the supplied PDFs.
- [`docs/files/`](docs/files/): unchanged full-resolution PDFs for the nine displayed figures.

The text and numerical results come from the supplied `all.zip` manuscript, especially `main.tex`, `src/3-Method.tex`, and `src/4-Experiments.tex`. Figure previews use `Figs/teaser.pdf`, `pipeline.pdf`, `dualgraph.pdf`, `cl.pdf`, `comp.pdf`, `step.pdf`, `mating.pdf`, `retrieval.pdf`, and `ablationdual.pdf`. The full LaTeX source archive is not redistributed here.

The manuscript's method macro spells the name **BRepDSN**, which is used on the page. The author email addresses follow the manuscript, including `songhaoxuan241@mails.ucas.ac.cn`.

### Deliberately unavailable information

The supplied archive contains neither a compiled full-paper PDF nor a video. The paper action is marked “coming soon”; the third action links to results rather than to a nonexistent video. Add the real paper file and replace the disabled span with a link when it is available.

The manuscript explicitly marks its year, DOI, volume, and issue metadata as unconfirmed placeholders. The page therefore does not claim a conference, journal acceptance, publication year, or DOI. The blue header line is the research area, not a venue. BibTeX is a provisional manuscript citation. Once publication and deployment details are confirmed, update the citation and scholarly metadata, add the paper URL, and set the canonical and absolute Open Graph URLs for the final site address.

The main table preserves all reported values and distinguishes results quoted from original papers. Improvements are **percentage-point differences**, not relative percentages. Operation-step results use **CC3D-Ops**. Assembly mating location and type metrics are separate; type prediction is conditioned on the ground-truth location.

## Checks

Run the dependency-free resource and accessibility-relationship checks:

```bash
python3 -m unittest discover -s tests -v
node --check docs/script.js
```

For browser verification, check both result tabs with pointer and keyboard, expand the additional figures, copy the citation (including denied-clipboard fallback), and inspect narrow-screen table scrolling. Check the page without JavaScript as well. The static checks do not replace these interaction and visual checks.

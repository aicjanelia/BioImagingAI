# Contribution Guidelines

Contributions to chapters of this textbook are made by invited collaborators. If you are a reader of the textbook and find an issue with example exercises or other materials, please open a GitHub issue. General questions about this book can be addressed to the editors: [@ScientistRachel](https://github.com/ScientistRachel) and [@opp1231](https://github.com/opp1231).

## Author Contributions
Authors should create a new branch to work on their chapters rather than forking the repository. Please create a pull request to trigger review by the editors. In your pull request message, please describe what level of review you are requesting: feedback on material, review of a first draft, editing of a revised draft, etc. You may also start a draft pull request and request review by the editors.  Do not commit directly to the main branch. Pull requests against the main branch must be approved by one of the editors.

## Quarto Formatting
The textbook is a [Quarto book](https://quarto.org/docs/books/) and general Quarto book [formatting](https://quarto.org/docs/authoring/markdown-basics.html), [cross-referencing](https://quarto.org/docs/authoring/cross-references.html), etc. can be used. In your chapter, only use one first-level header at the top of the document, which will set the chapter title as used in other places on the website. Second-level and below headers can be used to create sections in the chapter. Literature references can be added to the text by first adding them to _references.bib_.

## Embedding Code and Images
The repository contains a minimal conda environment for rendering the book that can be used to create executable pieces of code (e.g., to create a simple figure) inside a chapter. However, for larger examples or more complex calculations, please create a notebook in the notebooks directory that can then be [embedded in the chapter .qmd file](https://quarto.org/docs/authoring/notebook-embed.html). Please name notebooks starting with `chapter#_` and give them short but descriptive names.

The preference for this book is to embed images via url from a published repository or similar source. If you need to include images that are not available via url, please reach out to the editors.
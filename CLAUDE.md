# BioImagingAI Textbook - Chapter 3 (LLMs) Development Guide

## Book Overview

**Title**: "AI in Microscopy: A BioImaging Guide"
**Format**: Quarto book (`.qmd` files in `docs/`)
**Editors**: Owen Puls, Rachel Lee, Teng-Leong Chew (HHMI Janelia Research Campus)
**Audience**: Life scientists and microscopists with minimal technical/coding background
**Goal**: Demystify AI for bioimaging; empower researchers especially in low-resource settings

## Chapter Progress Summary (as of Feb 2026)

| Chapter | Title | Author(s) | Status |
|---------|-------|-----------|--------|
| 1 (Preface) | AI across the Microscopy Workflow | Teng-Leong Chew | **Complete** - Full prose, figures, cross-refs |
| 2 | AI Primer | Stephan Saalfeld | **Template only** - No content written |
| **3** | **LLMs and AI Agents** | **Wei Ouyang** | **Outline only** - Section headers + descriptions, no prose |
| 4 | Architectures and Loss Models | Carsen Stringer | **Template only** - No content written |
| 5 | Training Data | Martin Weigert | **Template only** - No content written |
| 6 | Image Restoration | Yue Li, Min Guo, Hari Shroff et al. | **Complete** - ~42K chars, deeply technical, figures, citations, appendix |
| 7 | Smart Microscopy | Wesley Legant, Suliana Manley | **Detailed outline** - Bullet-point structure, no prose |
| 8 | Finding and Using Tools | Beth Cimini, Erin Weisbart | **Complete** - ~34K chars, practical, figures, callout boxes |
| 9 | Training Your Own Models | Joanna Pylvänäinen, Guillaume Jacquemet et al. | **Complete** - ~65K chars, extensive, figures, step-by-step workflows |
| 10 | Output Quality | Morgan Schwartz, Diane Adjavon | **Detailed outline** - Bullet-point structure, no prose |
| 11 | Outlook | Anna Kreshuk | **Template only** - No content written |

### Key Observation
Only 4 of 11 chapters have substantial content (Ch1, Ch6, Ch8, Ch9). Ch3 is at outline stage. Several chapters (2, 4, 5, 11) still have the default template.

## Analysis of Well-Written Chapters (Patterns to Follow)

### Chapter 6 (Image Restoration) - Gold Standard
- **Structure**: Introduction with motivation -> General concepts -> Traditional approaches -> DL approaches -> Practical guidelines -> Limitations & future
- **Writing style**: Starts with WHY it matters for biologists, then explains the science
- **Figures**: Custom schematics for each major concept (degradation model, methods comparison)
- **Equations**: Used sparingly but with clear explanation of every symbol
- **Citations**: Extensive (`@key` format in references.bib), both foundational and recent
- **Glossary links**: Heavy use of `[term](glossary.qmd#term)` linking format
- **Cross-chapter references**: Links to Ch4, Ch5, Ch9, Ch10 where relevant
- **Has an appendix**: Step-by-step RCAN tutorial in `6-image-restoration-appendix.qmd`

### Chapter 8 (Finding Tools) - Best for Accessibility
- **Structure**: Introduction -> Needs assessment questions -> Tool categories -> Case studies
- **Unique approach**: Asks the reader questions to guide their thinking
- **Callout boxes**: Uses note/tip/warning/caution effectively for supplementary content
- **Figures with captions**: Uses `icon=false` callout notes for figure+caption combos
- **Tone**: Conversational, empathetic ("know that if you found the section above overwhelming, you are not alone")
- **Practical focus**: Real-world decision-making, not just theory

### Chapter 9 (Training Models) - Most Comprehensive
- **Structure**: Challenge -> Preparation -> Implementation workflow (6 steps) -> Tools -> Resources
- **Decision trees**: Flowcharts for "Is DL the right choice?" and "Train vs fine-tune?"
- **Collapsible tips**: Step-by-step tutorials in collapsible callout boxes
- **Section numbering**: Uses `{#sec-9.3.4.5}` style for deep nesting
- **Practical**: Includes hyperparameter guidance, loss monitoring advice
- **Multiple figures**: Each major concept has an illustrative figure

### Chapter 1 (Preface) - Exemplary Introduction
- **Structure**: Big picture motivation -> Textbook outline with cross-refs to all chapters
- **Tone**: Authoritative but accessible
- **Figure**: Single overview figure of the microscopy workflow
- **Cross-references**: Links to every subsequent chapter

## Chapter 3 Current State & Revision Plan

### Current Outline Structure (in `3-llms.qmd`)
1. Foundations of LLMs (`#sec-llm-foundations`)
2. Multi-modal AI: VLMs and Generative AI (`#sec-multimodal-ai`)
3. AI Agents for Microscopy Workflows (`#sec-ai-agents`)
4. Challenges and Limitations (`#sec-challenges`)
5. Future Directions (`#sec-future`)
6. Practical Guide: Getting Started (`#sec-practical-guide`)

### What to REMOVE (per author decision)
- **Training LLM section**: The original outline mentions LLM foundations/transformer architecture - remove or drastically simplify. The audience doesn't need to know how to train LLMs.

### What to KEEP & STRENGTHEN
- **Using LLMs for microscopy tasks**: This is the core value proposition
- **Multi-modal AI / Vision-Language Models**: Highly relevant (GPT-4o, Claude vision for image analysis)
- **AI Agents**: Omega, BioImage.io chatbot, MCP-based agents - this is cutting edge and unique
- **Practical guide**: Step-by-step examples are what make Ch8 and Ch9 successful

### Recommended Revised Structure
1. **Introduction** - Why LLMs matter for microscopists (motivating examples)
2. **What Are LLMs?** - Brief, intuitive explanation (NO training details, just what they are and can do)
3. **Using LLMs as Research Assistants** - Learning concepts, literature review, troubleshooting
4. **LLMs for Code Generation** - Writing analysis scripts, ImageJ macros, Python/napari code
5. **Vision-Language Models for Microscopy** - Image understanding, annotation assistance, GPT-4o/Claude examples
6. **AI Agents for Bioimage Analysis** - Omega, BioImage.io chatbot, autonomous workflows, MCP
7. **Challenges & Limitations** - Hallucinations, reproducibility, validation needs
8. **Practical Guide** - Prompt engineering tips, example workflows, getting started resources
9. **Summary**

## Writing Conventions & Formatting Rules

### Quarto Format
- One `#` for chapter title only
- `##`, `###` for sections/subsections
- Section refs: `{#sec-llms}`, `{#sec-ai-agents}` etc.
- Citations: `@citationkey` (add to `references.bib`)
- Glossary: `[term](glossary.qmd#term)` format
- Cross-chapter refs: `@sec-primer`, `@sec-4`, `@sec-8` etc.
- Figures: prefer URL-hosted images; use `![caption](path){#fig-label}` format
- Code blocks: use `{python}` with `#| eval: false` for non-executable examples
- Callout boxes: `::: {.callout-tip}`, `::: {.callout-warning}`, etc.

### YAML Header Template
```yaml
---
author:
  - name: Wei Ouyang
    orcid: 0000-0002-0291-926X
    affiliations:
      - name: SciLifeLab | KTH Royal Institute of Technology
        country: Sweden
subtitle: "Large Language Models and AI Agents for Microscopy Imaging"
---
```

### Style Guidelines (from .cursorrules)
- **Tone**: Friendly, accessible, informative
- **Jargon**: Avoid or define clearly; link to glossary
- **Analogies**: Use microscopy/biology analogies
- **Examples**: Practical scenarios from microscopy labs
- **Section length**: 300-500 words per section
- **AI disclosure**: Include a section noting AI tool usage in writing

### What Makes Chapters Great (Takeaways)
1. **Start with WHY** - Every section opens with motivation before explanation
2. **Use figures generously** - Schematics, flowcharts, decision trees, screenshots
3. **Callout boxes** - Tips, warnings, and practical exercises break up dense text
4. **Cross-reference other chapters** - The book should feel interconnected
5. **Practical workflows** - Step-by-step instructions readers can follow
6. **Honest about limitations** - Every chapter discusses caveats and failure modes
7. **Glossary linking** - Technical terms are linked on first use
8. **Real examples** - Tools that actually exist, with citations
9. **Decision guidance** - Help readers decide WHEN to use something, not just HOW

## Key References to Add (for Ch3)
- ChatGPT / GPT-4o for microscopy assistance
- Claude (Anthropic) for code generation and image understanding
- BioImage.io chatbot and model zoo
- Omega agent for bioimage analysis
- napari + LLM integrations
- ImageJ macro generation via LLMs
- MCP (Model Context Protocol) for tool integration
- Relevant papers on LLMs in scientific research

## File Paths
- Chapter file: `docs/3-llms.qmd`
- References: `docs/references.bib`
- Glossary: `docs/glossary.yml`
- Images: prefer URLs; or `docs/3-llms_files/` if local
- Notebooks: `notebooks/chapter3_*.ipynb`
- Quarto config: `docs/_quarto.yml`
- Style guide: `.cursorrules`

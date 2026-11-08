# AGENTS.md

## Project Context
This is an academic computer science research project involving
reproduction and extension of existing research.

The priority is scientific correctness, reproducibility, readable
implementation, and clear experimental methodology.

See the Proposal as well as the original paper (Certifiably Robust RAG).
The workspace rules regarding colab are in ONBOARDING.md.

## General Principles
- Inspect the existing codebase before making changes.
- Understand existing implementations before replacing them.
- Prefer simple, readable, modular code over unnecessary abstractions.
- Follow existing project conventions.
- Do not perform broad refactoring when a targeted change is sufficient.
- Do not invent file paths, data schemas, or experimental results.
- Ask for clarification when a missing detail materially affects correctness.

## Research Integrity
- Distinguish paper reproduction experiments from novel extensions.
- Never silently change experimental parameters, prompts, models,
  datasets, random seeds, or evaluation procedures.
- Record relevant experimental configurations alongside results.
- Preserve existing results and raw data.
- Clearly distinguish observed results from hypotheses and interpretations.
- Do not claim an experiment reproduces a paper without checking
  the corresponding experimental setup and metrics.

## Implementation
- Write concise, readable Python.
- Use descriptive variable and function names.
- Avoid unnecessary boilerplate, redundant environment checks,
  and excessive defensive programming.
- Derive labels, dataset sizes, and other quantities programmatically.
- Avoid hardcoding assumptions that prevent scaling to additional datasets.
- Reuse existing cached outputs whenever possible.
- Never regenerate expensive LLM responses unless explicitly requested.
- Keep data processing, experiment execution, and analysis logically separate.

## Experimental Reproducibility
- Keep baseline and extension experiments separate.
- Make important parameters explicit and configurable.
- Preserve original experimental configurations.
- Track model, dataset, alpha, top_k, attack, defense,
  prompt configuration, and sample size where applicable.
- Make it possible to reproduce an experiment from its saved configuration.

## Data Management
- Never overwrite original datasets, caches, or experiment outputs.
- Follow the existing directory structure.
- Organize new outputs by experiment and dataset.
- Use consistent filenames and documented column names.
- Preserve missing values and abstentions rather than silently discarding them.

## Testing and Verification
- Test new functionality on a small representative sample first.
- Verify output schemas and numerical calculations.
- Run the relevant existing tests when available.
- Do not launch expensive full-scale experiments without authorization.
- Report any unverified assumptions or limitations.

## Communication
- Explain nontrivial algorithms and mathematical reasoning clearly.
- When implementing a research idea, explain the methodology
  and assumptions, not just the code.
- Report which files were changed and what was verified.
- Do not hide important implementation decisions.
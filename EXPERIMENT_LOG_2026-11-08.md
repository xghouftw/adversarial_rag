# RobustRAG reproduction and attack-profiling experiment log

Date of work: 2026-11-08 America/New_York (the UTC run timestamps cross into 2026-11-09)

Status at handoff: the released-code and attached-paper parameter configurations were run on RealtimeQA and Open NQ. A response-level promotion/suppression feasibility profile was completed only for RealtimeQA with `alpha=0.2`. No live poisoning passages have been constructed or tested.

## 1. Research context and objectives

The project extends *Certifiably Robust RAG against Retrieval Corruption* (`Certifiably Robust RAG.pdf`). The project proposal is `RAG Poisoning Proposal.pdf`.

Today's work had two separate objectives.

### 1.1 Reproduction objective

Run the authors' released Mistral-7B keyword-aggregation implementation on the same fixed 100-question subsets used for Table I, save all responses and caches, and compare:

1. no retrieval (`no_rag`);
2. ordinary RAG with ten passages (`vanilla`);
3. RobustRAG keyword aggregation and its certificate for one injected passage (`keyword`).

Two keyword-threshold configurations were kept separate rather than silently conflated:

- `legacy_v1`: `alpha=0.3`, `beta=3.0`. This is the configuration in the released repository's `run.sh` commands.
- `current_v2`: `alpha=0.2`, `beta=3.0`. This is the short-answer QA configuration stated in the attached paper and associated with the paper's Table-I values.

The reproduction hypothesis was that the released code, pinned dependencies, fixed data, and greedy decoding would recover the published or repository-era counts up to small hardware-dependent differences. A second purpose was to establish trustworthy cached isolated responses for the extension experiment.

### 1.2 Extension objective and hypotheses

The proposed extension studies whether the keyword voting mechanism can be subverted without selecting an entirely novel attacker-invented target.

The working promotion hypothesis is:

> If an incorrect phrase already occurs in an isolated response to a benign retrieved passage but is filtered for insufficient support, injected passages that reproduce that phrase may raise its vote count above the retention threshold.

The working suppression hypothesis is:

> If injected passages cause distinct nonliteral nonanswers that RobustRAG counts as answering responses, they may increase the answer count and therefore the keyword threshold, causing weakly supported correct phrases to be filtered.

These mechanisms are coupled. A promotion response is also nonabstaining and can change the threshold. With a fixed top ten, each injection also displaces a bottom-ranked benign passage, so it can remove benign votes.

The attack budget considered in the profiler was `b in {1, 2, 3}` injected passages.

The feasibility profile completed today is only a response-level oracle analysis. It assumes that injected passages can induce selected isolated responses. It does **not** establish that natural-looking passages can realize those responses, that retained keyword changes alter the final LLM answer, or that an end-to-end attack succeeds.

## 2. RobustRAG mechanism being analyzed

For these experiments, `top_k=10` and group size is one, so Mistral answers the question separately from each retrieved passage.

The upstream implementation treats a response as abstaining only when it contains the exact substring `I don't`. Let `n` be the number of responses not recognized as abstentions. Each response contributes at most one vote per extracted phrase. A phrase is retained when its vote count is at least:

```text
threshold = min(beta, alpha * n)
```

The retained phrases, rather than the retrieved passages, are then given to Mistral for the final answer.

For certification against `b=1` injected passage, the implementation removes the bottom benign passage, considers the attacker's possible influence on nonabstaining count and medium-count keywords, and checks possible final responses. The reported certified count includes only questions whose clean defended response is correct and whose certificate succeeds.

## 3. Code, data, and environment provenance

### 3.1 Notebook and upstream source

The experiment driver is:

```text
notebooks/robustrag_reproduction.ipynb
```

The notebook pins the authors' repository to:

```text
https://github.com/inspire-group/RobustRAG.git
commit 9bc35b2fa5fa7d1088383b2789ec19a512d316b1
```

The upstream repository has no license file, so its source is not vendored into this repository. The notebook fetches the pinned commit into ephemeral Colab storage and exports a runtime copy.

The only runtime source compatibility modification was deferring `litellm` and `openai` imports used only by the unused GPT code path. Mistral generation, keyword aggregation, certification, prompts, and evaluation were not intentionally changed.

### 3.2 Datasets

The notebook checksum-verifies the upstream data files before running.

| Dataset argument | File | SHA-256 | Evaluated records |
|---|---|---|---:|
| `realtimeqa` | `realtimeqa.json` | `0924fd01734f90e7910d6cb5802d387aaf5811eeba650667a8e72d6b94d7941e` | first 100 |
| `open_nq` | `open_nq.json` | `32a71eab0a71f67e8f07ca8fddf8d9ac4d18c841d92df3f77a0e0cfa11ad8f11` | first 100 |

Open NQ contains more records, but the authors' `main.py` and this reproduction evaluate only the first 100.

### 3.3 Hardware and software

The successful runs used:

- GPU: NVIDIA L4, driver `580.82.07`, 23034 MiB;
- Python: `3.11.17` in `/content/robustrag-py311`;
- PyTorch: `2.2.1`;
- Transformers: `4.40.1`;
- Tokenizers: `0.19.1`;
- Accelerate: `0.29.3`;
- spaCy: `3.7.4`;
- `en_core_web_sm`: `3.7.1`;
- NLTK: `3.8.1`;
- NumPy: `1.26.4`;
- Joblib: `1.3.2`;
- SentencePiece: `0.2.0`;
- Hugging Face Hub: `0.23.5`.

The paper used an A100. Every manifest records `hardware_matches_paper: false`. Small numerical or generation differences due to L4 versus A100 remain possible.

### 3.4 Model and common parameters

| Parameter | Value |
|---|---|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Decoding | greedy (`do_sample=False` upstream) |
| Retrieved passages | 10 |
| Isolation group size | 1 |
| Certification corruption size | 1 |
| Keyword `beta` | 3.0 |
| Legacy/code `alpha` | 0.3 |
| Attached-paper `alpha` | 0.2 |
| Saved responses | yes |
| Response cache | yes |

For `no_rag`, `top_k=0`, defense is `none`, and corruption size is zero. For `vanilla`, `top_k=10`, defense is `none`, and corruption size is zero. For `keyword`, `top_k=10`, defense is `keyword`, corruption size is one, and `--no_vanilla` avoids an unnecessary additional vanilla query inside that command.

## 4. Execution chronology

All timestamps below are UTC and come from saved manifests.

1. **Failed two-question smoke attempt**: `2026-11-08T22:54:29` to `22:54:52`.
   - It used Colab's Python 3.13, Transformers 5.22.1, and `/content/robustrag-upstream/main.py`.
   - It failed before inference because upstream imported the removed private symbol `_crop_past_key_values`.
   - Saved directory: `outputs/connor/robustrag-reproduction/table1-legacy_v1-realtimeqa-20261108T225428Z/`.
   - This failed run is excluded from all results.

2. **Successful two-question smoke test**: `2026-11-08T23:10:41` to `23:12:38`.
   - RealtimeQA, legacy/code profile, keyword only.
   - Result: one of two clean answers correct and certified.
   - Saved directory: `table1-legacy_v1-realtimeqa-20261108T231041Z/`.

3. **Full RealtimeQA legacy/code profile**: `2026-11-08T23:12:38` to `23:28:36`.
   - Saved directory: `table1-legacy_v1-realtimeqa-20261108T231238Z/`.

4. **Full RealtimeQA attached-paper profile**: `2026-11-08T23:28:36` to `2026-11-09T01:55:24`.
   - Reused the legacy run's isolated-response cache where prompts matched.
   - Saved directory: `table1-current_v2-realtimeqa-20261108T232836Z/`.

5. **Full Open NQ legacy/code profile**: `2026-11-09T01:55:24` to `02:15:01`.
   - Saved directory: `table1-legacy_v1-open_nq-20261109T015524Z/`.

6. **Full Open NQ attached-paper profile**: `2026-11-09T02:15:01` to `03:04:32`.
   - Reused the legacy run's cache where prompts matched.
   - Saved directory: `table1-current_v2-open_nq-20261109T021501Z/`.

7. **RealtimeQA `alpha=0.2` response-level attack profile**: saved at approximately `2026-11-09T03:05:04`.
   - Saved directory: `attack-profile-realtimeqa-alpha0.2-20261109T030504Z/`.

The notebook intended to profile RealtimeQA at both `alpha=0.2` and `alpha=0.3`, but only the `alpha=0.2` directory exists. The saved artifacts do not reveal why the second iteration did not complete. NQ attack profiling was never invoked; the notebook explicitly scoped the first profiler run to RealtimeQA.

## 5. Reproduction results

Counts are out of 100 questions. “Expected” is the count stored in the notebook for the corresponding profile. Deltas are observed minus expected.

| Dataset | Profile | No RAG | Vanilla | Keyword clean | Keyword certified |
|---|---|---:|---:|---:|---:|
| RealtimeQA | `alpha=0.3` | 8 / 8 (0) | 69 / 69 (0) | 59 / 59 (0) | 46 / 45 (+1) |
| RealtimeQA | `alpha=0.2` | 8 / 8 (0) | 69 / 69 (0) | 61 / 71 (-10) | 35 / 38 (-3) |
| Open NQ | `alpha=0.3` | 30 / 30 (0) | 62 / 61 (+1) | 55 / 54 (+1) | 47 / 47 (0) |
| Open NQ | `alpha=0.2` | 30 / 30 (0) | 62 / 61 (+1) | 59 / 61 (-2) | 25 / 26 (-1) |

The exact metric files are:

- `outputs/connor/robustrag-reproduction/table1-legacy_v1-realtimeqa-20261108T231238Z/metrics.json`
- `outputs/connor/robustrag-reproduction/table1-current_v2-realtimeqa-20261108T232836Z/metrics.json`
- `outputs/connor/robustrag-reproduction/table1-legacy_v1-open_nq-20261109T015524Z/metrics.json`
- `outputs/connor/robustrag-reproduction/table1-current_v2-open_nq-20261109T021501Z/metrics.json`

### 5.1 Reproduction interpretation

The released-code `alpha=0.3` configuration reproduced closely: RealtimeQA clean keyword accuracy matched exactly, its certificate differed by one question, and both Open NQ keyword counts differed by at most one question.

Merely changing the released implementation to the attached paper's `alpha=0.2` did not reproduce the paper's RealtimeQA `71/38` result. The observed result was `61/35`, although the RealtimeQA no-RAG and vanilla baselines matched exactly. This is a real reproduction discrepancy and must be reported rather than tuned away.

The authors' pinned `run.sh` uses `alpha=0.3`, `beta=3` for keyword Table-I commands, whereas the attached paper states `alpha=0.2`, `beta=3` for short-answer QA. The upstream README also warns that the repository and preprint can differ. The released implementation therefore supports a strong reproduction claim for the released-code configuration, not an exact reproduction claim for every attached-paper number.

The L4/A100 hardware difference is an unresolved confound. It could contribute to generation differences, but the exact RealtimeQA baselines and exact legacy clean keyword count make it insufficient, by itself, to explain the discrepancy without further evidence.

### 5.2 Upstream certification fallbacks

The upstream warning `added list too long... skipped` means the certification routine declined to enumerate an overly large set of medium-count keyword combinations and returned a failed certificate for that query. It is not a runtime crash.

Observed warning counts were:

| Run | `added list too long` | Clean abstention warning | Traceback |
|---|---:|---:|---:|
| RealtimeQA `alpha=0.3` | 1 | 1 | 0 |
| RealtimeQA `alpha=0.2` | 6 | 1 | 0 |
| Open NQ `alpha=0.3` | 0 | 1 | 0 |
| Open NQ `alpha=0.2` | 16 | 1 | 0 |

These fallbacks can lower certified accuracy. They do not directly explain the RealtimeQA clean keyword accuracy gap because clean final-answer generation happens before certification.

## 6. Response-level promotion/suppression profiler

### 6.1 Purpose

The profiler reused the completed RealtimeQA cache. It did not regenerate isolated LLM responses. For each question, it reconstructed the authors' isolated prompts, retrieved their cached responses, repeated the upstream short-answer keyword extraction, and simulated budgets one through three.

For a budget `b`, it removed the bottom `b` benign keyword sets before adding `b` simulated injected response sets. This models the paper's injection assumption that new documents displace the lowest-ranked benign documents in a fixed top ten.

### 6.2 Simulated mechanisms

**Promotion** selected a non-gold phrase that had actually been extracted from a benign isolated response but was not retained in the clean keyword set. The simulation added one vote for that phrase per injected passage and recorded it when it became retained.

**Threshold suppression** added `b` distinct nonempty marker sets representing semantic nonanswers that do not contain the exact literal `I don't`. Distinct markers prevent the simulation from accidentally promoting one repeated wrong answer. A case was recorded as threshold suppression only when a gold-aligned phrase survived benign passage displacement but disappeared after the injected answering responses raised the threshold.

**Vote loss** was recorded separately when displacement alone removed every retained gold-aligned phrase before the injected responses raised the threshold.

### 6.3 Per-query profile fields

The profiler wrote 100 rows to `profiles.json`. The four proposal quantities currently map to:

| Proposal quantity | Saved field | Current definition |
|---|---|---|
| Number of answering passages, `n_b` | `answering_passages` | Isolated responses not containing literal `I don't` |
| Correct-keyword vote count, `c*` | `correct_keyword_max_count` | Maximum count over phrases matched heuristically to a gold alias |
| Strongest incorrect-keyword count, `d` | `strongest_incorrect_count` | Maximum count over non-gold, nonpunctuation, non-stopword phrases |
| Abstention count | `abstaining_passages` | Ten minus answering passages |

The row also stores the gold-aligned phrase counts, strongest alleged incorrect phrase, clean threshold, clean correctness, and labels for budgets one through three.

### 6.4 Saved attack-profile results

Files:

- `outputs/connor/robustrag-reproduction/attack-profile-realtimeqa-alpha0.2-20261109T030504Z/profiles.json`
- `outputs/connor/robustrag-reproduction/attack-profile-realtimeqa-alpha0.2-20261109T030504Z/findings.json`
- `outputs/connor/robustrag-reproduction/attack-profile-realtimeqa-alpha0.2-20261109T030504Z/summary.json`

The profile contained 61 clean-correct and 39 clean-incorrect questions. The answering-passage distribution ranged from zero to ten:

```text
n=0: 1, n=1: 3, n=2: 3, n=3: 3, n=4: 6,
n=5: 4, n=6: 7, n=7: 11, n=8: 15, n=9: 19, n=10: 28
```

The saved summary was:

| Mechanism | Candidate rows | Unique questions | Clean-correct questions |
|---|---:|---:|---:|
| Promotion | 1,946 | 58 | 38 |
| Threshold suppression | 5 | 4 | 1 |
| Vote loss | 1 | 1 | 0 |

Breakdown by budget:

| Mechanism | Budget | Rows | Unique questions | Clean-correct questions |
|---|---:|---:|---:|---:|
| Promotion | 1 | 606 | 56 | 36 |
| Promotion | 2 | 670 | 58 | 38 |
| Promotion | 3 | 670 | 58 | 38 |
| Threshold suppression | 1 | 0 | 0 | 0 |
| Threshold suppression | 2 | 1 | 1 | 0 |
| Threshold suppression | 3 | 4 | 4 | 1 |
| Vote loss | 1 | 0 | 0 | 0 |
| Vote loss | 2 | 0 | 0 | 0 |
| Vote loss | 3 | 1 | 1 | 0 |

The only clean-correct threshold-suppression candidate was query 57, “Who won this year's Nobel Peace Prize?”, with gold answer `Narges Mohammadi` and budget three. Its clean profile had six answering and four abstaining passages. After removing the bottom three passages, four benign answering responses remained; adding three nonliteral nonanswers raised the answering count to seven and the threshold from `0.8` to `1.4`, removing the surviving one-vote gold-aligned phrases in the simulated survivor state.

### 6.5 What the profile does and does not show

The raw profile suggests that structural promotion opportunities are common and structural suppression opportunities are rare for RealtimeQA at `alpha=0.2`. In particular, 38 of 61 clean-correct questions had at least one mechanically promotable extracted phrase, while only one clean-correct question met the simulated threshold-suppression condition.

These are not attack success rates. The 1,946 promotion rows are substantially inflated because one isolated response generates a complete-response phrase, component keyphrases, and individual lemmas. The output includes plausible wrong answers such as `Dragonflies`, `Bob Menendez`, `191,000 books`, and `Finding Freedom`, but also generic fragments such as `context`, `year`, and `politician`.

The current `d` calculation is also not reliable enough for scientific analysis. For query 57 it labels `narge Mohammadi` as the strongest incorrect phrase even though it is a malformed lemmatized form of the gold answer `Narges Mohammadi`. Thus `strongest_incorrect_count` should be treated as `d_raw`, not as a validated wrong-answer count.

The label `safe` in the current profile means only “this heuristic profiler found no structural candidate.” It is not a safety or certification guarantee and should be renamed before publication.

## 7. Current derived populations

Aggregate metrics imply the following counts of clean-correct but uncertified questions:

- RealtimeQA `alpha=0.2`: `61 - 35 = 26`.
- Open NQ `alpha=0.2`: `59 - 25 = 34`.
- Combined attached-paper-profile pool: 60.
- RealtimeQA `alpha=0.3`: `59 - 46 = 13`.
- Open NQ `alpha=0.3`: `55 - 47 = 8`.
- Combined released-code-profile pool: 21.

These differences are valid aggregate counts because `certified_correct` is counted only when the defended answer is correct and certified. However, the current saved results do not identify which individual questions are certified. Consequently, it is not yet known which of the 38 clean-correct promotion-profile questions are uncertified.

The primary attack population should be clean-correct, uncertified questions. Clean-incorrect questions cannot demonstrate a clean-to-attacked failure. Certified clean-correct questions are useful negative controls: keyword sets may change, but an in-scope deterministic attack should not make their final responses incorrect if the certificate and implementation are correct.

## 8. Threat-model decisions still requiring agreement

The phrase “fabricates no new factual claims” needs an operational definition.

Recommended primary promotion condition:

- The wrong claim or target must be explicitly present in the original benign retrieved passage.
- The attacker may copy or paraphrase that existing claim but may not introduce a new answer target.

Recommended secondary promotion condition:

- The target occurs in the isolated model response but not necessarily in the source passage. This recycles a model error but is a weaker interpretation of “no new factual claim.”

Recommended suppression conditions:

- Primary: natural claim-free passages without embedded instructions.
- Secondary upper bound: passage-level prompt injection that forces a nonliteral nonanswer. This must be reported separately because it changes the attack subtype.

## 9. Required next work

Do not rerun the expensive reproduction pipeline. The required caches already exist.

1. **Recover exact per-question certification labels from cache.** Use a cache-only replay of the upstream keyword defense so each profile row can be labeled clean-correct/certified, clean-correct/uncertified, or clean-incorrect.
2. **Repair phrase classification.** Save both:
   - `d_raw`: strongest mechanically non-gold extracted phrase;
   - `d_answer`: strongest manually or semantically validated wrong answer.
   Also save the phrases attaining `c*` and `d`, their source isolated responses, and their source passages.
3. **Canonicalize promotion candidates.** Collapse component phrases by `(query, source_response)` so one response such as `Finding Freedom` is one candidate rather than several lemma rows.
4. **Profile missing configurations from existing caches.** Run:
   - RealtimeQA `alpha=0.3`;
   - Open NQ `alpha=0.2`;
   - Open NQ `alpha=0.3`.
5. **Record budget-specific survivor quantities.** For each budget, store answering count and keyword counts after bottom-passage displacement, because clean `n_b`, `c*`, and `d` alone do not capture ejected votes.
6. **Export a manual-audit table.** Include dataset, query, gold aliases, clean final answer, certificate, original passage, isolated response, candidate target, target vote count, minimum predicted budget, and strict/response-recycled/reject annotation.
7. **Only after audit, run a small live pilot.** Measure separately:
   - response realization: injected passage induces the intended isolated response;
   - mechanism realization: retained keywords change as predicted;
   - targeted success: final answer contains the promoted wrong target;
   - untargeted failure: final answer no longer contains a gold answer.
8. **Use controls.** Include certified promotion candidates as negative controls, clean-correct uncertified questions with no found candidate, and exact-`I don't know` suppression controls.

## 10. Claims that are currently justified

The following claims are supported by saved artifacts:

- Both datasets were reproduced under both threshold configurations with fixed 100-question samples.
- Released-code `alpha=0.3` results match the stored reproduction targets exactly or within one question.
- The released implementation with `alpha=0.2` does not reproduce the attached paper's RealtimeQA `71/38` result on the L4; it produces `61/35`.
- The RealtimeQA `alpha=0.2` response-level profiler finds many mechanically promotable extracted phrases but only one clean-correct threshold-suppression candidate.
- Raw promotion rows and raw `d` values contain semantic false positives and cannot yet be treated as validated attack candidates.

The following claims are **not** yet justified:

- that any natural injected passage realizes a profiled response;
- that promotion or suppression changes the final answer;
- that the reported structural candidate proportions are attack success rates;
- that questions labeled `safe` are robust;
- that the paper/repository discrepancy is caused by GPU hardware;
- that promotion is statistically more effective than suppression across datasets, because three profiles and per-query certification labels are still missing.

## 11. Artifact map

| Artifact | Purpose |
|---|---|
| `notebooks/robustrag_reproduction.ipynb` | Complete Colab reproduction and first profiler implementation |
| `Certifiably Robust RAG.pdf` | Attached reference paper |
| `RAG Poisoning Proposal.pdf` | Project proposal |
| `ONBOARDING.md` | Colab/Drive workflow and data-handling rules |
| `outputs/connor/robustrag-reproduction/table1-*` | Pulled manifests, metrics, logs, results, and caches for reproduction runs |
| `outputs/connor/robustrag-reproduction/attack-profile-realtimeqa-alpha0.2-20261109T030504Z/` | Only completed attack-profile configuration |

The `outputs/` directory is populated by `./pull.sh` from the shared Drive. It is not a substitute for the Drive copy. Running `./pull.sh` during a Colab experiment is remote-read-only, but an active output file may be an incomplete snapshot until Colab closes or flushes it.

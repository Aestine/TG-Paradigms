# How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms

Official code for the CVPR 2026 Workshops (ECV) paper *"How Should Video LLMs
Output Time? An Analysis of Efficient Temporal Grounding Paradigms"* (Jin et al.),
studying **how temporal predictions should be formulated in a Video-LLM's output
space**. We implement three representative
video-temporal-grounding (VTG) output paradigms under one unified framework and
compare them on compact backbones (0.5B–8B) while holding backbone, training
data, optimizer, epochs, and fine-tuning protocol fixed — so the **output
formulation is the only variable**.

| Paradigm (paper) | `--paradigm` flag | Idea | Code |
|---|---|---|---|
| Continuous Temporal Decoding (*Cont.*) | `distime` | distribution over time bins (DisTime-style) | `models/*_distime.py`, `models/time_modules.py` |
| Temporal Token Generation (*Gen.*) | `trace` | dedicated temporal tokens + causal event head (TRACE-style) | `models/*_trace.py`, `models/trace_modules.py` |
| Text Numeral (*Text*) | `text` | timestamps as plain-text numerals (VTimeLLM-style) | `models/*_text.py`, `models/text_paradigm_base.py` |

Backbones: **SmolVLM2 0.5B / 2.2B** (SigLIP), **FastVLM-1.5B** (FastViTHD),
**Molmo2 4B / 8B** (SigLIP-SO400M). Benchmarks: **Charades-STA, QVHighlights, YouCook2**.

> `README_legacy_DisTime.md` is the original single-paradigm README, kept for reference.

---

## Repository layout

Minimal, code-only release (no datasets / weights / figures — see [Data](#data)).

```
.
├── README.md  requirements.txt  LICENSE  .gitignore
├── configs/                 # accelerate + DeepSpeed ZeRO configs, train_config.yaml
├── models/
│   ├── smolvlm_distime.py  fastvlm_distime.py  molmo2_distime.py   # Cont. paradigm
│   ├── smolvlm_trace.py    fastvlm_trace.py    molmo2_trace.py     # Gen. paradigm
│   ├── smolvlm_text.py     fastvlm_text.py     molmo2_text.py      # Text paradigm
│   ├── text_paradigm_base.py  time_modules.py  trace_modules.py  manual_lora.py  merge_lora.py
│   └── molmo2/             # Molmo2 modeling / configuration / processors (code only)
├── data/
│   ├── dataset.py          # LazySupervisedDataset, collate_fn, build_datasets (3 paradigms)
│   ├── convert_internvid_to_trace.py  convert_trace_to_distime.py  filter_trace_data.py
│   └── __init__.py
├── utils/                  # args, losses, metrics, dist_utils, mm_utils
├── scripts/
│   ├── train.py                  # unified training entry point
│   ├── train_single.sh           # example: DisTime (set --model_type for the backbone)
│   ├── train_single_trace.sh     # example: TRACE
│   ├── train_single_text.sh      # example: Text numeral
│   ├── train_slurm_multinode.sh  # example: multi-node SLURM
│   └── merge_and_export.py       # merge LoRA + export a deployable checkpoint
└── eval/
    ├── evaluate_{charades,qvh,youcook2}.py   # Cont. paradigm
    ├── evaluate_{charades,qvh}_trace.py      # Gen. paradigm (needs TRACE, see below)
    ├── evaluate_text.py                      # Text paradigm (generation + regex parse)
    └── benchmarks/benchmark_efficiency.py    # params / latency / throughput / memory
```

All entry points use **absolute imports from the repo root** (`from models…`,
`from utils…`, `from data…`). Run from the repo root or add it to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

## Installation

```bash
conda create -n ecv2026 python=3.10 -y && conda activate ecv2026
pip install -r requirements.txt
# optional, faster training:
pip install flash-attn --no-build-isolation
```

---

## Data

**No datasets, annotations or videos are included** — this is a code-only
release. Download the data from the official sources below and convert it with
the scripts in `data/`.

### Training data

The paper trains on ~1.2M temporally-annotated samples (~400K videos) from 11
public sources. Get each from its official release:

| Source | Where |
|---|---|
| InternVid | https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid |
| YT-Temporal | https://rowanzellers.com/merlot/ |
| Valley | https://github.com/RupertLuo/Valley |
| DiDeMo | https://github.com/LisaAnne/TemporalLanguageRelease |
| ShareGPT4Video | https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video |
| ViTT | https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT |
| TextVR | https://github.com/callsys/TextVR |
| COIN | https://coin-dataset.github.io/ |
| ActivityNet Captions | https://cs.stanford.edu/people/ranjaykrishna/densevid/ |
| QuerYD | https://www.robots.ox.ac.uk/~vgg/data/queryd/ |
| VideoChat | https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data |

All paradigms share the same `(video, query, annotation)` triples; each applies
its own formatting. Build the per-paradigm training files with:

```bash
python data/convert_internvid_to_trace.py ...   # raw  -> TRACE (Gen.) format
python data/convert_trace_to_distime.py   ...   # TRACE -> DisTime (Cont.) format
# the Text paradigm reuses the DisTime file directly (dataset reformats targets)
```

Expected per-sample JSON schema consumed by `data/dataset.py`:

```json
{"video": "path/to/video.mp4", "query": "...", "start": 10.5, "end": 15.2, "caption": "..."}
```

### Evaluation benchmarks

| Benchmark | Annotations | Videos |
|---|---|---|
| Charades-STA | https://github.com/jiyanggao/TALL (`charades_sta_{train,test}.txt`) | https://prior.allenai.org/projects/charades |
| QVHighlights | https://github.com/jayleicn/moment_detr (`data/`) | same repo / YouTube |
| YouCook2 | http://youcook2.eecs.umich.edu/ | http://youcook2.eecs.umich.edu/ |

> Links are official project pages; verify the exact file names against each
> repo, as dataset hosting occasionally moves.

---

## Training

One example launcher per paradigm is provided; edit the paths inside, and pick
the backbone with `--model_type`:

```bash
bash scripts/train_single.sh         # DisTime (Cont.)   — set MODEL_TYPE / --model_type
bash scripts/train_single_trace.sh   # TRACE   (Gen.)
bash scripts/train_single_text.sh    # Text numeral
bash scripts/train_slurm_multinode.sh  # multi-node SLURM example
```

`scripts/train.py` dispatches on `--model_type {smolvlm,fastvlm,molmo2}` and
`--paradigm {distime,trace,text}`. The five backbones (SmolVLM2-0.5B/2.2B,
FastVLM-1.5B, Molmo2-4B/8B) are selected via `--model_type` + the corresponding
`--model_name_or_path`. Protocol (fixed across all runs): freeze vision encoder,
LoRA (r=16, α=32) on the LLM, 32 frames, 1 epoch, AdamW lr 1e-4 cosine,
DeepSpeed ZeRO-2, bf16.

**Ablations** (paper Fig. 6, SmolVLM2-2.2B):
- *Context length* (8→64 frames): vary `--max_frames`.
- *Data efficiency* (25→100%): subset via per-source sampling weights in
  `data/dataset.py` (`build_datasets`); a one-flag `--data_fraction` driver is
  not yet provided (see gaps below).

---

## Evaluation

Each evaluator is a Python entry point (run with `python eval/<file>.py --help`):

```bash
# Cont. (DisTime)
python eval/evaluate_charades.py  --model_type smolvlm --checkpoint_dir <ckpt> --data_file <charades_test> --video_root <videos>
python eval/evaluate_qvh.py       ...
python eval/evaluate_youcook2.py  ...

# Gen. (TRACE) — needs the external TRACE package (see Known gaps)
python eval/evaluate_charades_trace.py ...

# Text numeral
python eval/evaluate_text.py --task charades --model_type smolvlm --checkpoint_dir <ckpt> --data_file <charades_test> --video_root <videos>

# Efficiency profile
python eval/benchmarks/benchmark_efficiency.py
```

Metrics: Charades/QVH moment retrieval → R1@0.5, R1@0.7, mIoU; QVH highlight
detection → mAP, HIT@1 (Gen. only); YouCook2 dense captioning → CIDEr, SODA_c, F1.

---

## Code–paper alignment — known gaps

Read this before reproducing the paper tables.

1. **Text Numeral paradigm — IMPLEMENTED (was missing in the original code).**
   Run with `--paradigm text`. Added components:
   `data/dataset.py::_getitem_text` (targets formatted exactly as the paper
   Appendix: *"The event happens from X to Y seconds."*, reusing the same frames /
   prompt / label masking as the other paradigms; consumes the SAME data file as
   DisTime); `models/{smolvlm,fastvlm,molmo2}_text.py` +
   `models/text_paradigm_base.py` (base VLM + LoRA, standard next-token loss, no
   extra modules); `scripts/train.py` text branch + `scripts/train_single_text.sh`;
   `eval/evaluate_text.py` (generation + "from X to Y seconds" parsing, with the
   exact Appendix inference prompts for Charades / QVHighlights / YouCook2).
   - Format, prompts, loss and hyperparameters follow the paper Appendix exactly.
   - **Validation status:** data-format and timestamp-parse logic are unit-tested
     (the single-event target matches the Appendix verbatim; format↔parse is
     round-trip consistent). Full training and `generate()` were NOT run here
     (need GPUs, the multi-GB corpus, and real backbone weights). Molmo2 uses a
     simple greedy decode loop (no KV cache) — validate/optimize on real weights.
   - Exact reproduction of the paper's *Text* accuracy numbers is not guaranteed
     (depends on the training run / seed / data), but the methodology matches.
   - `eval/evaluate_text.py` covers **moment retrieval** (Charades-STA, QVH-MR):
     R@1 / mIoU. It does NOT compute QVHighlights highlight detection (mAP/HIT@1
     — not applicable to Text by design) nor YouCook2 dense-captioning metrics
     (CIDEr/SODA_c/F1); for the latter, route the parsed multi-event text through
     the scorer in `eval/evaluate_youcook2.py`. All three text `generate()`
     implementations return generated-only tokens and use KV cache
     (SmolVLM/FastVLM via HF `.generate`, Molmo2 via a cached manual loop that
     mirrors `molmo2_distime.generate`).

2. **TRACE (Gen.) baseline depends on the external TRACE codebase.**
   `eval/evaluate_*_trace.py` import a top-level `trace` package
   (`from trace.model.builder import ...`). Vendor it or document the dependency:
   clone `https://github.com/gyxxyg/TRACE`, add to `PYTHONPATH`, pin the commit.

3. **Qwen3-VL + TRACE files** (if you choose to include them) need the upstream
   TRACE package layout (`..multimodal_encoder.builder`, `..constants`); they do
   not resolve standalone.

4. **Data efficiency ablation** has the mechanism (per-source sampling weights)
   but no single `--data_fraction` flag/script — add one for clean reproduction.

5. **Large training corpora and source videos are not included** — provide a
   download script or external host (HF / cloud) and link them in the Data section.

6. **Before publishing:** add a `LICENSE` (template provided), scrub
   cluster-specific paths / accounts in `scripts/*.slurm` and `*.sh`, and add a
   `CITATION.cff` / BibTeX entry.

---

## Citation

```bibtex
@InProceedings{Jin_2026_CVPR,
  author    = {Jin, Shengji and Zou, Yuanhao and Zhu, Victor and Ji, Zhengping and Chen, Chen},
  title     = {How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2026},
  pages     = {3539-3548}
}
```

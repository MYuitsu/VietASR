# Rebuilding the VietASR Paper Test

This repository can train and decode VietASR models, but the public code does not ship the exact paper benchmark harness. The paper evaluates offline ASR with:

- `WER` as the metric
- `modified_beam_search`
- `beam-size 4`
- three public Vietnamese test sets:
  - `GigaSpeech 2 TEST`
  - `Common Voice 17.0 TEST`
  - `FLEURS TEST`

The helper changes in this workspace add the missing pieces to reproduce that evaluation layout.

## 1. Prepare benchmark manifests

Organize each public test set as a split under a single directory:

```text
SSL/download/paper_benchmark
├── gigaspeech2
│   └── ...
├── commonvoice
│   └── ...
└── fleurs
    └── ...
```

Each split should follow the same layout already expected by `ASR/local/prepare_manifest.py`:

```text
split_name/
├── speaker_or_subset/
│   ├── *.wav
│   └── filename.trans.txt
```

Create manifests:

```bash
python ASR/local/prepare_manifest.py \
  --corpus-dir SSL/download/paper_benchmark \
  --output-dir SSL/data/paper_manifests \
  --language vietnamese \
  --dataset-parts "gigaspeech2 commonvoice fleurs"
```

## 2. Compute fbank features

```bash
python ASR/local/compute_fbank.py \
  --manifest-dir SSL/data/paper_manifests \
  --output-dir SSL/data/fbank \
  --dataset "gigaspeech2 commonvoice fleurs"
```

This writes:

- `SSL/data/fbank/vietASR_cuts_gigaspeech2.jsonl.gz`
- `SSL/data/fbank/vietASR_cuts_commonvoice.jsonl.gz`
- `SSL/data/fbank/vietASR_cuts_fleurs.jsonl.gz`

## 3. Decode with the paper setup

For ASR checkpoints, run from `ASR/`:

```bash
GIGASPEECH2_CUTS=../SSL/data/fbank/vietASR_cuts_gigaspeech2.jsonl.gz \
COMMONVOICE_CUTS=../SSL/data/fbank/vietASR_cuts_commonvoice.jsonl.gz \
FLEURS_CUTS=../SSL/data/fbank/vietASR_cuts_fleurs.jsonl.gz \
./scripts/decode_paper_benchmark.sh 12 1 zipformer/your_exp_dir 0
```

Run from `SSL/`:

```bash
GIGASPEECH2_CUTS=../data/fbank/vietASR_cuts_gigaspeech2.jsonl.gz \
COMMONVOICE_CUTS=../data/fbank/vietASR_cuts_commonvoice.jsonl.gz \
FLEURS_CUTS=../data/fbank/vietASR_cuts_fleurs.jsonl.gz \
./scripts/decode_paper_benchmark.sh 12 1 zipformer_fbank/your_exp_dir 0
```

The helper script uses:

- `modified_beam_search`
- `beam-size 4`
- one decode run per test set
- use the `ASR/` version when your checkpoint comes from `ASR/zipformer/...`
- use the `SSL/` version when your checkpoint comes from `SSL/zipformer_fbank/...`

## 4. Summarize the weighted average

```bash
python SSL/local/summarize_paper_benchmark.py \
  --gigaspeech2-cuts SSL/data/fbank/vietASR_cuts_gigaspeech2.jsonl.gz \
  --gigaspeech2-summary SSL/zipformer_fbank/your_exp_dir/paper-benchmark/modified_beam_search/wer-summary-gigaspeech2-modified_beam_search-epoch-12-avg-1-modified_beam_search-beam-size-4.txt \
  --commonvoice-cuts SSL/data/fbank/vietASR_cuts_commonvoice.jsonl.gz \
  --commonvoice-summary SSL/zipformer_fbank/your_exp_dir/paper-benchmark/modified_beam_search/wer-summary-commonvoice-modified_beam_search-epoch-12-avg-1-modified_beam_search-beam-size-4.txt \
  --fleurs-cuts SSL/data/fbank/vietASR_cuts_fleurs.jsonl.gz \
  --fleurs-summary SSL/zipformer_fbank/your_exp_dir/paper-benchmark/modified_beam_search/wer-summary-fleurs-modified_beam_search-epoch-12-avg-1-modified_beam_search-beam-size-4.txt
```

## Notes

- The paper’s main offline table is not using the repo’s default `SSL/scripts/decode.sh`, because that script currently decodes only the repo-local `dev/test` split and defaults to `greedy_search`.
- The paper reports a weighted average by word count across the three test sets.
- The exact numeric match still depends on using the same model checkpoint, text normalization, and public test-set transcripts as the authors used.
Common Voice 17.0: 11.13% at wer-summary-commonvoice-beam_size_4-epoch-12-avg-1-modified_beam_search-beam-size-4.txt
FLEURS: 10.96% at wer-summary-fleurs-beam_size_4-epoch-12-avg-1-modified_beam_search-beam-size-4.txt
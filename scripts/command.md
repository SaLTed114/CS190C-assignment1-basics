```bash
python scripts/prepare_data.py \
  --train_txt data/tinystories/debug/train_20M.txt \
  --valid_txt data/tinystories/debug/valid_2M.txt \
  --out_dir artifacts/debug_tinystories \
  --vocab_size 4096
```

```bash
# prepare dataset
mkdir -p data/tinystories/mid

head -c 300000000 data/tinystories/TinyStoriesV2-GPT4-train.txt \
  > data/tinystories/mid/train_300M.txt

cp data/tinystories/TinyStoriesV2-GPT4-valid.txt \
  data/tinystories/mid/valid_full.txt

# prepare data
python scripts/prepare_data.py \
  --train_txt data/tinystories/mid/train_300M.txt \
  --valid_txt data/tinystories/mid/valid_full.txt \
  --out_dir artifacts/tinystories_mid_4k \
  --vocab_size 8192 \
  --save_pickle

# train
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/tinystories_train_stable.json
```

```bash
tmux new -s tinystories
tmux attach -t tinystories

# background hang
Ctrl-b d
```

## full training
```bash
mkdir -p data/tinystories/full

cp data/tinystories/TinyStoriesV2-GPT4-train.txt \
  data/tinystories/full/train_full.txt
cp data/tinystories/TinyStoriesV2-GPT4-valid.txt \
  data/tinystories/full/valid_full.txt

mkdir -p artifacts/tinystories_full_8k

python scripts/prepare_data.py \
  --train_txt data/tinystories/full/train_full.txt \
  --valid_txt data/tinystories/full/valid_full.txt \
  --out_dir artifacts/tinystories_full_8k \
  --vocab_size 8192 \
  --save_pickle

CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config configs/tinystories_full_aggressive.json

python scripts/sample.py --checkpoint artifacts/tinystories_full_aggressive/best.pt --config artifacts/tinystories_full_8k/
```

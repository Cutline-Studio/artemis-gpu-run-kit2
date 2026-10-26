"""On-instance data pipeline (runs alongside training).
FineWeb-Edu (HuggingFaceFW/fineweb-edu, ODC-By 1.0) sample/10BT parquet shards, streamed straight from the
HF CDN (no login), tokenized with the fixed tok24k BPE, written as ~100M-token uint16 chunks fw_XXXX.bin
(atomic rename; the trainer picks them up). The first 3000 docs of shard 000 -> fw_val.bin (never trained on).
Stops after --target tokens. Next shard downloads in a background thread while the current one tokenizes.
Also fetches databricks-dolly-15k (CC-BY-SA-3.0) for Stage 2. Progress -> data/fetch_progress.json."""
import argparse, json, os, threading, time, urllib.request, numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--tok", required=True); ap.add_argument("--target", type=float, default=2.3e9)
ap.add_argument("--out", default="data"); ap.add_argument("--max-shards", type=int, default=5); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); tk = Tokenizer.from_file(a.tok); EOS = tk.token_to_id("<|eos|>")
BASE = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/{:03d}_00000.parquet"
prog = {"dataset": "HuggingFaceFW/fineweb-edu sample/10BT", "license": "ODC-By 1.0 (plus Common Crawl terms of use)",
        "shards": [], "chunks": [], "train_tokens": 0, "val_tokens": 0, "docs": 0, "started": time.time()}
def save(): json.dump(prog, open(os.path.join(a.out, "fetch_progress.json"), "w"), indent=1)
def dl(i):
    p = os.path.join(a.out, f"shard{i:03d}.parquet"); t0 = time.time()
    urllib.request.urlretrieve(BASE.format(i), p + ".part"); os.replace(p + ".part", p)
    prog["shards"].append({"file": BASE.format(i).split("main/")[1], "bytes": os.path.getsize(p), "download_s": round(time.time() - t0, 1)}); save()
    return p
try:
    urllib.request.urlretrieve("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl", os.path.join(a.out, "dolly15k.jsonl"))
    prog["dolly"] = {"dataset": "databricks/databricks-dolly-15k", "license": "CC-BY-SA-3.0", "bytes": os.path.getsize(os.path.join(a.out, "dolly15k.jsonl"))}
except Exception as e:
    prog["dolly_error"] = str(e)[:200]
save()
nxt = {}; th = threading.Thread(target=lambda: nxt.__setitem__(0, dl(0))); th.start()
buf, ci = [], 0
def flush():
    global buf, ci
    if not buf: return
    arr = np.concatenate(buf).astype(np.uint16); f = os.path.join(a.out, f"fw_{ci:04d}.bin")
    arr.tofile(f + ".tmp"); os.replace(f + ".tmp", f); prog["chunks"].append({"file": os.path.basename(f), "tokens": int(len(arr))})
    prog["train_tokens"] += int(len(arr)); ci += 1; buf = []; save()
for i in range(a.max_shards):
    th.join(); p = nxt.pop(i)
    if i + 1 < a.max_shards and prog["train_tokens"] + 0.75e9 < a.target:
        th = threading.Thread(target=lambda j=i + 1: nxt.__setitem__(j, dl(j))); th.start()
    pf = pq.ParquetFile(p); first = (i == 0); n = 0
    for b in pf.iter_batches(batch_size=4000, columns=["text"]):
        texts = b.column(0).to_pylist(); prog["docs"] += len(texts)
        if first:
            enc = tk.encode_batch(texts[:3000]); v = []
            for e in enc: v.extend(e.ids); v.append(EOS)
            np.array(v, dtype=np.uint16).tofile(os.path.join(a.out, "fw_val.bin")); prog["val_tokens"] = len(v); prog["val_docs"] = 3000; texts = texts[3000:]; first = False
        enc = tk.encode_batch(texts); ids = []
        for e in enc: ids.extend(e.ids); ids.append(EOS)
        buf.append(np.array(ids, dtype=np.uint16)); n += len(ids)
        if sum(len(x) for x in buf) >= 1e8: flush()
        if prog["train_tokens"] + sum(len(x) for x in buf) >= a.target: break
    flush(); os.remove(p)
    if prog["train_tokens"] >= a.target: break
prog["finished"] = time.time(); prog["done"] = True; save()
open(os.path.join(a.out, "FETCH_DONE"), "w").write("ok")

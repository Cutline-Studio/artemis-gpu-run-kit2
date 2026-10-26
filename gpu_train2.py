"""Run 2: ~94M dense Artemis on FineWeb-Edu (streamed on the instance) + CC export + WordNet + Artemis docs x5, BPE tok24k.
bf16 autocast, SDPA, torch.compile, packed windows (random contiguous windows over the growing token pool, <|eos|>
between docs), AdamW warmup+cosine, held-out bpb (CC web_val, FineWeb-Edu val, Artemis val), periodic checkpoints,
wall-clock deadline. Data pool = pinned CPU int16 buffer filled with base bins + fw_XXXX.bin chunks as fetch_data.py
writes them (checked every 200 steps). Stage 2: SFT on Artemis chats (definitions capped at 30%, Artemis doc
dialogues x5) + Dolly-15k, assistant-only loss, early stopping on held-out SFT loss."""
import argparse, glob, json, math, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from artemis_dense import train as T
from artemis_dense.engines import transformer as tx
from artemis_dense.tok import get_codec, n_bytes
IGN = -100
P = ["define serendipity", "My name is Latoya. What can you do?", "who must approve a GPU rental?", "what is 12*13?", "fix it"]
ap = argparse.ArgumentParser()
for k, v in dict(run="m94", tok="tok24k/tokenizer.json", data="data", out="out", d_model=768, n_layers=12, n_heads=12, n_kv_heads=4,
                 d_ff=2048, context_len=1024, batch=16, accum=2, lr=6e-4, warmup=1000, tokens_per_param=20.0, stage1_budget_s=18000.0,
                 eval_every=1000, ckpt_every_s=1800.0, pool_tokens=2.6e9, deadline_epoch=0.0, sft_max_steps=2000, sft_batch=16,
                 sft_lr=1e-4, sft_patience=3, artemis_up=5, seed=567885, compile=1, min_start_tokens=1.5e8, web_val_windows=200).items():
    ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
a = ap.parse_args()
torch.set_num_threads(8); dev = "cuda"; amp = dict(device_type="cuda", dtype=torch.bfloat16)
torch.manual_seed(a.seed); random.seed(a.seed); g = torch.Generator().manual_seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
rd = os.path.join(a.out, a.run); os.makedirs(rd, exist_ok=True); logf = open(os.path.join(rd, "train_log.jsonl"), "a")
def log(r):
    r["t"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); s = json.dumps(r); logf.write(s + "\n"); logf.flush(); print(s, flush=True)
late = lambda margin=0: bool(a.deadline_epoch) and time.time() > a.deadline_epoch - margin
codec = get_codec(kind="bpe", path=a.tok); V = codec.vocab_size; assert V < 32768
model = tx.build("medium", seed=a.seed, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, n_kv_heads=a.n_kv_heads, d_ff=a.d_ff,
                 context_len=a.context_len, dropout=0.0, tokenizer="bpe", tokenizer_path=os.path.abspath(a.tok), vocab_size=V).to(dev)
NP = model.num_params()
# ---------------- token pool
pool = torch.empty(int(a.pool_tokens), dtype=torch.int16); filled = 0; loaded = set(); src = {}
def add(name, arr, rep=1):
    global filled
    t = torch.from_numpy(arr.astype(np.int16))
    for _ in range(rep):
        n = min(len(t), len(pool) - filled)
        if n <= 0: return
        pool[filled:filled + n] = t[:n]; filled += n
        src[name] = src.get(name, 0) + n
rb = lambda f: np.fromfile(f, dtype=np.uint16)
add("cc_web_train", rb("web_train.bin")); add("wordnet", rb("wordnet.bin")); add(f"artemis_train_x{a.artemis_up}", rb("artemis_train.bin"), a.artemis_up)
def refresh():
    for f in sorted(glob.glob(os.path.join(a.data, "fw_*.bin"))):
        if f not in loaded and not f.endswith(".tmp"):
            add("fineweb_edu", rb(f)); loaded.add(f)
while True:
    refresh()
    if src.get("fineweb_edu", 0) >= a.min_start_tokens or os.path.exists(os.path.join(a.data, "FETCH_DONE")): break
    time.sleep(10)
wv, fv, av = [torch.from_numpy(rb(f).astype(np.int64)) for f in ("web_val.bin", os.path.join(a.data, "fw_val.bin"), "artemis_val.bin")]
nb = lambda ids, w: n_bytes(codec.decode(ids[:w * a.context_len + 1].tolist()))
wv_b, fv_b = nb(wv, a.web_val_windows), nb(fv, a.web_val_windows)
av_b = n_bytes(open("mix/artemis_val.txt", encoding="utf-8").read())
# ---------------- SFT set
art_tr = open("mix/artemis_train.txt", encoding="utf-8").read(); dl = T.sft_dialogues(art_tr)
defs = [d for d in dl if d[0]["content"].startswith("define ")]; conv_n = len([l for l in open("artemis_dense/data/corpus/conversations.jsonl") if l.strip()])
convs = dl[:conv_n]; probs = [d for d in dl[conv_n:] if not d[0]["content"].startswith("define ")]
dolly = []
if os.path.exists(os.path.join(a.data, "dolly15k.jsonl")):
    for l in open(os.path.join(a.data, "dolly15k.jsonl")):
        r = json.loads(l); u = r["instruction"] + (("\n\nContext: " + r["context"]) if r.get("context") else "")
        dolly.append([{"role": "user", "content": u}, {"role": "assistant", "content": r["response"]}])
others = convs * 5 + probs + dolly; random.Random(a.seed).shuffle(defs)
defs = defs[:int(0.3 / 0.7 * len(others))]; allc = others + defs; random.Random(a.seed + 1).shuffle(allc)
recs = [T.encode_masked(d, codec) for d in allc]; recs = [r for r in recs if any(y != IGN for y in r[1][:a.context_len])]
sft_va = recs[:max(200, len(recs) // 20)]; sft_tr = recs[len(sft_va):]
sft_mix = {"artemis_doc_dialogues": f"{len(convs)} x5", "math_physics": len(probs), "dolly15k": len(dolly), "definitions": len(defs),
           "definition_share": round(len(defs) / max(1, len(allc)), 3), "train_records": len(sft_tr), "val_records": len(sft_va)}
fwd = model
if a.compile:
    try:
        t0 = time.time(); fwd = torch.compile(model)
        with torch.autocast(**amp): fwd(torch.zeros(a.batch, a.context_len, dtype=torch.long, device=dev))[0].float().sum().backward()
        model.zero_grad(set_to_none=True); log({"event": "compile_ok", "s": round(time.time() - t0, 1)})
    except Exception as e:
        fwd = model; log({"event": "compile_failed", "err": str(e)[:300]})
@torch.no_grad()
def bpb(ids, nbytes, maxw=None):
    model.eval(); Tn = a.context_len; nll = 0.0; w = 0
    for s in range(0, len(ids) - 1, Tn):
        c = ids[s:s + Tn + 1].to(dev)
        if len(c) < 2: break
        with torch.autocast(**amp): lg, _ = model(c[None, :-1])
        nll += F.cross_entropy(lg[0].float(), c[1:], reduction="sum").item(); w += 1
        if maxw and w >= maxw: break
    return round(nll / math.log(2) / nbytes, 4)
@torch.no_grad()
def sftv():
    model.eval(); tot = 0.0; n = 0
    for x, y in sft_va:
        x = torch.tensor([x[:a.context_len]], device=dev); y = torch.tensor([y[:a.context_len]], device=dev)
        with torch.autocast(**amp): lg, _ = model(x)
        tot += F.cross_entropy(lg[0].float(), y[0], ignore_index=IGN, reduction="sum").item(); n += int((y != IGN).sum())
    return round(tot / max(n, 1), 4)
ev = lambda: {"cc_web_val_bpb": bpb(wv, wv_b, a.web_val_windows), "fineweb_val_bpb": bpb(fv, fv_b, a.web_val_windows), "artemis_val_bpb": bpb(av, av_b), "sft_val_nll": sftv()}
def samples():
    model.to("cpu"); o = {p: T.sample(model, codec, p) for p in P}; model.to(dev); return o
def save(name, info=None):
    sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    ck = {"cfg": model.cfg.to_dict(), "model": sd}
    if info: ck["trained_info"] = info
    p = os.path.join(rd, name); torch.save(ck, p + ".tmp"); os.replace(p + ".tmp", p)
summ = {"run": a.run, "command": "python gpu_train2.py " + " ".join(sys.argv[1:]), "args": vars(a), "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "python": sys.version.split()[0], "model_card": tx.model_card(model), "params_total": NP,
        "params_non_embedding": NP - model.embed.weight.numel(), "sft_mix": sft_mix, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
log({"event": "start", **summ, "pool_at_start": dict(src)}); summ["before"] = ev(); log({"event": "before", **summ["before"]})
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
B, Tn = a.batch, a.context_len; total = 10 ** 9; best = {"bpb": 1e9, "step": 0}; seen = 0; stop = "steps"; step = 0
t0 = time.time(); lastck = time.time(); ar = torch.arange(Tn + 1)
while step < total:
    if late(1500): stop = "deadline"; break
    model.train()
    for pg in opt.param_groups: pg["lr"] = T.lr_at(step, total, a.lr, a.warmup)
    opt.zero_grad(set_to_none=True); lsum = 0.0
    for _ in range(a.accum):
        ix = torch.randint(0, filled - Tn - 2, (B,), generator=g)
        xy = pool[ix[:, None] + ar].to(dev, non_blocking=True).long()
        x, y = xy[:, :-1], xy[:, 1:]
        with torch.autocast(**amp): lg, _ = fwd(x)
        loss = F.cross_entropy(lg.float().reshape(-1, V), y.reshape(-1)) / a.accum
        loss.backward(); lsum += loss.item()
    loss = torch.tensor(lsum)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    step += 1; seen += B * a.accum * Tn
    if step == 30: torch.cuda.synchronize(); tm = time.time()
    if step == 130:
        torch.cuda.synchronize(); sps = 100 / (time.time() - tm)
        left = a.stage1_budget_s - (time.time() - t0)
        if a.deadline_epoch: left = min(left, a.deadline_epoch - 1500 - time.time())
        fit = int(sps * left * 0.92) + 130; want = int(a.tokens_per_param * NP / (B * a.accum * Tn))
        total = max(1000, min(fit, want))
        log({"event": "plan", "steps_per_s": round(sps, 3), "tokens_per_s": round(sps * B * a.accum * Tn), "stage1_steps": total, "fit_steps": fit,
             "chinchilla_steps": want, "planned_tokens": total * B * a.accum * Tn, "tokens_per_param": round(total * B * a.accum * Tn / NP, 2),
             "train_flops_6ND": 6 * NP * total * B * a.accum * Tn, "pool_tokens_now": filled})
    if step % 200 == 0: refresh()
    if step % 100 == 0 or step % a.eval_every == 0 or step == total:
        r = {"stage": 1, "step": step, "loss": round(loss.item(), 4), "tokens": seen, "lr": round(opt.param_groups[0]["lr"], 7),
             "elapsed_s": round(time.time() - t0, 1), "pool": filled}
        if step % a.eval_every == 0 or step == total:
            r["val"] = ev()
            if r["val"]["fineweb_val_bpb"] < best["bpb"]: best = {"bpb": r["val"]["fineweb_val_bpb"], "step": step}; save("stage1_best.pt")
        log(r)
    if time.time() - lastck > a.ckpt_every_s: save("stage1_last.pt", {"step": step}); lastck = time.time()
summ["pool_sources_tokens"] = dict(src); summ["pool_filled"] = filled
summ["stage1"] = {"steps_run": step, "planned_steps": total, "stop": stop, "tokens": seen, "wall_s": round(time.time() - t0, 1),
                  "epochs_over_pool": round(seen / filled, 3), "best": best}
save("stage1_final.pt", {"step": step}); summ["after_stage1"] = ev(); summ["samples_after_stage1"] = samples()
log({"event": "stage1_done", **summ["stage1"], "pool_sources_tokens": dict(src), "after": summ["after_stage1"], "samples": summ["samples_after_stage1"]})
opt = torch.optim.AdamW(model.parameters(), lr=a.sft_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
t0 = time.time(); seen2 = 0; s2 = 0; bestv = sftv(); bad = 0; save("stage2_best.pt", {"sft_step": 0}); bstep = 0
for s2 in range(1, a.sft_max_steps + 1):
    if late(600): break
    model.train(); bt = [sft_tr[i] for i in torch.randint(0, len(sft_tr), (a.sft_batch,), generator=g).tolist()]
    L = min(Tn, max(len(x) for x, _ in bt)); X = torch.zeros(len(bt), L, dtype=torch.long); Y = torch.full((len(bt), L), IGN, dtype=torch.long)
    for j, (x, y) in enumerate(bt):
        x, y = x[:L], y[:L]; X[j, :len(x)] = torch.tensor(x); Y[j, :len(y)] = torch.tensor(y)
    X, Y = X.to(dev), Y.to(dev)
    for pg in opt.param_groups: pg["lr"] = T.lr_at(s2 - 1, a.sft_max_steps, a.sft_lr, 50)
    with torch.autocast(**amp): lg, _ = model(X)
    loss = F.cross_entropy(lg.float().reshape(-1, V), Y.reshape(-1), ignore_index=IGN)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    seen2 += int((Y != IGN).sum())
    if s2 % 100 == 0:
        v = sftv(); log({"stage": 2, "step": s2, "loss": round(loss.item(), 4), "assistant_tokens": seen2, "sft_val_nll": v})
        if v < bestv - 1e-3: bestv = v; bad = 0; bstep = s2; save("stage2_best.pt", {"sft_step": s2})
        else:
            bad += 1
            if bad >= a.sft_patience: break
summ["stage2"] = {"steps_run": s2, "best_step": bstep, "assistant_tokens": seen2, "wall_s": round(time.time() - t0, 1), "best_sft_val_nll": bestv}
model.load_state_dict(torch.load(os.path.join(rd, "stage2_best.pt"), weights_only=False)["model"])
summ["after_stage2_best"] = ev(); summ["samples_after_stage2_best"] = samples(); summ["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
json.dump(summ, open(os.path.join(rd, "summary.json"), "w"), indent=2)
log({"event": "done", "stage2": summ["stage2"], "after_stage2_best": summ["after_stage2_best"], "samples": summ["samples_after_stage2_best"]})
open(os.path.join(rd, "DONE"), "w").write("ok")

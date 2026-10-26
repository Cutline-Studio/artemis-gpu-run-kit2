"""Run-2 vs run-1 comparison, run ON THE GPU INSTANCE (no box-CPU eval). Same tok24k, same held-out sets:
CC web_val, FineWeb-Edu val (fw_val.bin), Artemis val; bpb over the same first N tokens of each set for every model
(windows of each model's own context length, bf16 autocast on CUDA). Plus the 5 fixed prompts and a scripted REPL
(artemis_dense.chat --ckpt) per model. Writes out/m94/eval_compare.json and out/m94/repl_<name>.txt."""
import io, json, math, os, sys, time, numpy as np, torch, torch.nn.functional as F
os.chdir("/workspace/run"); sys.path.insert(0, "/workspace/run")
from artemis_dense import train as T
from artemis_dense.engines import transformer as tx
from artemis_dense.tok import get_codec, n_bytes
from artemis_dense import chat
TOKP = "/workspace/run/tok24k/tokenizer.json"; dev = "cuda"; NTOK = int(os.environ.get("NTOK", 200 * 1024)); OUT = os.environ.get("OUT", "out/m94")
P = ["define serendipity", "My name is Latoya. What can you do?", "who must approve a GPU rental?", "what is 12*13?", "fix it"]
REPL = ["define serendipity", "My name is Latoya. What can you do?", "is the chat server running?", "what is 12*13?", "fix it",
        "how much would a GPU run cost?", "who are you?"]
codec = get_codec(kind="bpe", path=TOKP)
rb = lambda f: np.fromfile(f, dtype=np.uint16).astype(np.int64)
sets = {"cc_web_val": rb("web_val.bin")[:NTOK + 1], "fineweb_val": rb("data/fw_val.bin")[:NTOK + 1], "artemis_val": rb("artemis_val.bin")[:NTOK + 1]}
nbytes = {k: n_bytes(codec.decode(v[1:].tolist())) for k, v in sets.items()}
def fix(src, dst):
    ck = torch.load(src, map_location="cpu", weights_only=False); ck["cfg"]["tokenizer_path"] = TOKP; torch.save(ck, dst); return dst
@torch.no_grad()
def bpb(m, ids):
    Tn = m.cfg.context_len; ids = torch.from_numpy(ids); nll = 0.0
    for s in range(0, len(ids) - 1, Tn):
        c = ids[s:s + Tn + 1].to(dev)
        if len(c) < 2: break
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): lg, _ = m(c[None, :-1])
        nll += F.cross_entropy(lg[0].float(), c[1:], reduction="sum").item()
    return nll
cks = {"run1_medium_bpe24k_stage2_final": fix("/workspace/run1_stage2_final.pt", "/workspace/run1_s2_fixed.pt")}
for n in ("stage1_best", "stage1_final", "stage2_best"):
    p = f"out/m94/{n}.pt"
    if os.path.exists(p): cks[f"run2_m94_{n}"] = fix(p, f"/workspace/run2_{n}_fixed.pt")
res = {"gpu": torch.cuda.get_device_name(0), "tokens_per_set": NTOK, "bytes_per_set": nbytes, "models": {}}
for name, path in cks.items():
    t0 = time.time(); m = tx.load(path).to(dev).eval()
    r = {"ckpt": path, "params": m.num_params(), "context_len": m.cfg.context_len}
    for k, v in sets.items(): r[k + "_bpb"] = round(bpb(m, v) / math.log(2) / nbytes[k], 4)
    m = m.to("cpu"); c = get_codec(m.cfg); r["samples"] = {p: T.sample(m, c, p, max_new=120) for p in P}
    if name.endswith("stage1_best") or name.endswith("stage1_final"): r.pop("samples")  # base models: bpb only
    out = io.StringIO()
    if not name.startswith("run2_m94_stage1"):
        import subprocess
        pr = subprocess.run([sys.executable, "-m", "artemis_dense.chat", "--ckpt", path, "--max-new", "120"], input="\n".join(REPL) + "\n",
                            capture_output=True, text=True, timeout=600, cwd="/workspace/run")
        out.write(pr.stdout + (("\n[stderr tail]\n" + pr.stderr[-800:]) if pr.returncode else ""))
        open(f"{OUT}/repl_{name}.txt", "w").write(out.getvalue())
    r["eval_s"] = round(time.time() - t0, 1); res["models"][name] = r; print(json.dumps({name: r})[:400], flush=True)
    del m; torch.cuda.empty_cache()
json.dump(res, open(f"{OUT}/eval_compare.json", "w"), indent=1)

"""On-instance: tokenize the base parts (CC export web_train/web_val, WordNet glosses, Artemis train/val) with tok24k."""
import sys, json, numpy as np
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1]); EOS = tk.token_to_id("<|eos|>"); out = {}
def w(name, ids): np.array(ids, dtype=np.uint16).tofile(name + ".bin"); out[name] = len(ids)
for nm in ("web_train", "web_val"):
    docs = [d for d in open(f"mix/{nm}.txt", encoding="utf-8").read().split("\n<|eos|>\n") if d.strip()]; ids = []
    for e in tk.encode_batch(docs): ids.extend(e.ids); ids.append(EOS)
    w(nm, ids)
lines = open("mix/wordnet.txt", encoding="utf-8").read().split("\n"); ids = []
for e in tk.encode_batch(["\n".join(lines[i:i + 2000]) + "\n" for i in range(0, len(lines), 2000)]): ids.extend(e.ids)
w("wordnet", ids)
for nm in ("artemis_train", "artemis_val"): w(nm, tk.encode(open(f"mix/{nm}.txt", encoding="utf-8").read()).ids)
json.dump(out, open("base_token_counts.json", "w"), indent=1); print(out)

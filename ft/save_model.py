"""Promote a finished LoRA training session checkpoint to a Fireworks model and download its weights locally.

Usage: ft/.venv/bin/python ft/save_model.py <checkpoints.json key>   (e.g. qwen_r2, nemotron_r2)
Needs FIREWORKS_API_KEY in env. Writes promoted model id back into checkpoints.json and files under ft/models/<model_id>/.
"""
import json, os, sys, time, urllib.request
from datetime import datetime
from pathlib import Path

from fireworks import Fireworks
from fireworks.training.sdk import FireworksClient

FT = Path(__file__).resolve().parent
key = sys.argv[1]
ck_path = FT / "checkpoints.json"
ck = json.load(open(ck_path))
entry = ck[key]
api_key = os.environ["FIREWORKS_API_KEY"]

fw = FireworksClient(api_key=api_key)
model_name = entry.get("promoted_model")
if not model_name:
    rows = fw.list_training_session_checkpoints(entry["session"])
    promotable = [r for r in rows if r.get("promotable")]
    if not promotable:
        sys.exit(f"no promotable checkpoints in {entry['session']}: {rows}")
    target = max(promotable, key=lambda r: datetime.fromisoformat(r["createTime"].replace("Z", "+00:00")))
    out_id = f"holo-ab-{key.replace('_', '-')}-{time.strftime('%m%d%H%M')}"
    model = fw.promote_session_checkpoint(name=target["name"], output_model_id=out_id, base_model=entry["base_model"])
    model_name = getattr(model, "name", None) or (model.get("name") if isinstance(model, dict) else str(model))
    print("promoted", target["name"], "->", model_name)
    ck = json.load(open(ck_path))
    ck[key]["promoted_model"] = model_name
    json.dump(ck, open(ck_path, "w"), indent=2)
else:
    print("already promoted:", model_name)

model_id = model_name.split("/")[-1]
account_id = model_name.split("/")[1] if model_name.startswith("accounts/") else fw.account_id
client = Fireworks(api_key=api_key, account_id=account_id)
for attempt in range(30):
    m = client.models.get(model_id=model_id)
    state = str(getattr(m, "state", ""))
    print("model state:", state)
    if "READY" in state:
        break
    time.sleep(10)

ep = client.models.get_download_endpoint(model_id=model_id)
urls = ep.filename_to_signed_urls or {}
out = FT / "models" / model_id
out.mkdir(parents=True, exist_ok=True)
for fname, url in urls.items():
    dest = out / fname
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"downloaded {fname} ({dest.stat().st_size/1e6:.1f} MB)")
print("saved to", out)

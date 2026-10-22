import nbformat as nbf
nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)
nb.cells = [
md("""# QLoRA fine-tune: request classifier (Llama 3.2 3B Instruct)
Run on **Colab / Kaggle with a GPU** (Runtime > Change runtime type > T4 GPU).
Before you start, upload `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` and `common.py`, `evaluate.py` to the Colab file panel (left sidebar).
If a cell errors because a library API changed, copy the error message and send it to me."""),
code("""%%capture
!pip install unsloth
!pip install --upgrade trl datasets"""),
md("## 0. Settings"),
code("""MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LEN = 1536
N_TRAIN = None        # None = use all training data. For the ablation run with e.g. 500, 1000, 2000
RUN_NAME = "ft_full"  # use ft_n500, ft_n1000 ... for ablation runs (the number must match N_TRAIN)
LORA_R, LORA_ALPHA = 16, 16
EPOCHS, LR = 3, 2e-4
SEED = 3407"""),
md("## 1. Load model + LoRA adapters"),
code("""from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LEN, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)"""),
md("## 2. Data (chat format -> training text)"),
code("""from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})
if N_TRAIN:
    ds["train"] = ds["train"].shuffle(seed=SEED).select(range(min(N_TRAIN, len(ds["train"]))))
fmt = lambda ex: {"text": [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in ex["messages"]]}
ds = ds.map(fmt, batched=True)
print({k: len(v) for k, v in ds.items()})
print(ds["train"][0]["text"][:900])"""),
md("## 3. Train (loss only on the assistant's JSON answer)"),
code("""import torch
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds["train"], eval_dataset=ds["validation"],
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN, dataset_num_proc=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=EPOCHS, learning_rate=LR, lr_scheduler_type="cosine", warmup_ratio=0.05,
        weight_decay=0.01, optim="adamw_8bit", logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", save_total_limit=2,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        seed=SEED, report_to="none", output_dir="outputs"))
trainer = train_on_responses_only(
    trainer, instruction_part="<|start_header_id|>user<|end_header_id|>\\n\\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\\n\\n")
stats = trainer.train()"""),
md("## 4. Loss curves (save these for the README)"),
code("""import matplotlib.pyplot as plt
log = trainer.state.log_history
tr = [(l["step"], l["loss"]) for l in log if "loss" in l]
ev = [(l["step"], l["eval_loss"]) for l in log if "eval_loss" in l]
plt.figure(figsize=(6,4))
plt.plot(*zip(*tr), label="train loss")
if ev: plt.plot(*zip(*ev), marker="o", label="val loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title(RUN_NAME); plt.tight_layout()
plt.savefig(f"loss_{RUN_NAME}.png", dpi=150); plt.show()"""),
md("## 5. Predict on the TEST set (same output format as `run_model.py`)\nRun this once per run. For `ft_full` do this only after you've finished tuning on validation data."),
code("""import json, time, torch
from common import SYSTEM_PROMPT, build_user_message
FastLanguageModel.for_inference(model)
tokenizer.padding_side = "left"
test = [json.loads(l) for l in open("test.jsonl", encoding="utf-8") if l.strip()]
preds, BATCH = [], 8
for i in range(0, len(test), BATCH):
    batch = test[i:i+BATCH]
    prompts = [tokenizer.apply_chat_template(
        [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":build_user_message(r["subject"], r["message"])}],
        tokenize=False, add_generation_prompt=True) for r in batch]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=320, do_sample=False, use_cache=True)
    dt = (time.time() - t0) / len(batch)
    for r, o in zip(batch, out):
        txt = tokenizer.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append({"id": r["id"], "raw": txt, "latency_s": dt, "prompt_tokens": int(enc["attention_mask"][0].sum())})
    print(f"{min(i+BATCH, len(test))}/{len(test)}")
import os; os.makedirs("preds", exist_ok=True)
with open(f"preds/{RUN_NAME}.jsonl", "w") as f:
    for p in preds: f.write(json.dumps(p) + "\\n")"""),
code("""!python evaluate.py --preds preds/{RUN_NAME}.jsonl --name {RUN_NAME} --test test.jsonl"""),
md("## 6. Save adapter + export to GGUF for Ollama (do this for the final `ft_full` run only)"),
code("""model.save_pretrained(f"adapter_{RUN_NAME}"); tokenizer.save_pretrained(f"adapter_{RUN_NAME}")
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
!ls -lh gguf_model"""),
md("## 7. Download everything you need"),
code("""!zip -r results_{RUN_NAME}.zip loss_{RUN_NAME}.png preds results adapter_{RUN_NAME} gguf_model/*.gguf
from google.colab import files
files.download(f"results_{RUN_NAME}.zip")  # Kaggle: find the zip in the Output tab"""),
]
nbf.write(nb, "training/train_lora.ipynb")
print("notebook written")

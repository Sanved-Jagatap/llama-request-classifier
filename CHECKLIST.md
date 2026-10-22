# YOUR CHECKLIST: what to run, WHERE, and WHEN

Everything else is already built. Do the steps in order. Rough hands-on time: 6-10 hours, mostly waiting on runs.

## Where each step runs (legend)

| Tag | Meaning |
|---|---|
| **[VS Code]** | VS Code terminal on your Windows PC, opened in the project folder (File > Open Folder > `finetuned-classifier`, then Terminal > New Terminal). The commands are for PowerShell. |
| **[Colab]** | Google Colab notebook in your browser (colab.research.google.com) |
| **[Kaggle]** | Kaggle notebook in your browser (only if you choose Kaggle instead of Colab) |
| **[Ollama]** | Ollama must be running on your PC (it usually runs in the system tray; check with `ollama list` in the VS Code terminal) |
| **[Browser]** | A website (n8n, Hugging Face, GitHub) |

---

## 0. One-time setup  [VS Code]
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```
Re-activate the venv (`.\.venv\Scripts\Activate.ps1`) every time you open a new terminal.

---

## A. Build the dataset (~2-3 h)

**A1. Test the generator on a tiny run  [VS Code] + [Ollama]**
```powershell
ollama pull mistral:7b                       # skip if you already have it
python generate_dataset.py --dry-run         # prints one prompt, no model call
python generate_dataset.py --backend ollama --model mistral:7b --calls 20 --per-call 4 --workers 2
python build_dataset.py
```
Read the printed rejection counts and ~20 rows of `data/raw/raw.jsonl` by hand. If more than ~25% are rejected, send me the output.

*If you use an Anthropic API key instead of Ollama:* set it first with `$env:ANTHROPIC_API_KEY="your-key"` and replace the generate command with `python generate_dataset.py --calls 20 --per-call 5`.

**A2. Review and extend the hand-written test examples  [VS Code]**
Open `data/seed/handwritten.jsonl` in the editor. I drafted 61 examples. Read every line, fix labels you disagree with, rewrite some in your own words, and add more until you have 80+. Keep the same JSON format (one object per line). These are TEST ONLY, never used for training.

**A3. Full dataset run  [VS Code] + [Ollama]**
```powershell
Remove-Item data\raw\raw.jsonl               # throw away the 20-call trial
python generate_dataset.py --backend ollama --model mistral:7b --calls 800 --per-call 4 --workers 2
python build_dataset.py
```
This can take hours with a local model. Leave it running. (Use a generator from a different family than Llama, which is why we use mistral.)

**A3b. Independent label check  [VS Code] + [Ollama]**  (added after the trial showed ~3 of 8 mislabeled rows)
A second model classifies every clear message without seeing our label; rows where it disagrees get dropped.
```powershell
ollama pull llama3:8b                        # skip if you already have it
python verify_labels.py                      # resumable; writes data\raw\verdicts.jsonl; can take 1-2 h
python build_dataset.py                      # now prints how many rows the judge dropped, and why
```
Run it after the full generation finishes (it uses the same GPU). If it is interrupted, just run it again; it continues where it stopped.

**A4. Manual quality review  [VS Code]**
```powershell
python review_sample.py --make               # writes data\review_sample.csv
```
Open the CSV in Excel/VS Code, fill the `ok(y/n)` column for each row (does the label fit the message?), save, then:
```powershell
python review_sample.py --tally              # prints your label-error rate
```
Write that number down for the model card. If it's above ~10%, send me a few bad rows.

---

## B. Baseline (~30 min)  [VS Code] + [Ollama]

**B1.** Done for you: `prompts/baseline_prompt.txt` now holds your exact production prompt, extracted from `incoming-request-workflow-v6.json` (no action needed).

**B2.**
```powershell
ollama pull llama3.2                         # skip if already pulled
python run_model.py --model llama3.2 --prompt baseline --out preds\baseline.jsonl
python evaluate.py --preds preds\baseline.jsonl --name baseline
```

---

## C. Training (~1-3 h)  [Colab] (or [Kaggle])

### Option 1: Google Colab (recommended)
1. **[Browser]** Go to colab.research.google.com > File > Upload notebook > choose `training/train_lora.ipynb`.
2. Runtime > Change runtime type > **T4 GPU** > Save.
3. In the left sidebar, open the Files panel (folder icon) and drag in these 5 files from your PC: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, `common.py`, `evaluate.py`. They must sit in the notebook's root folder (not in a subfolder).
4. Run the cells from the top down (Runtime > Run all works too). Leave the defaults (`RUN_NAME = "ft_full"`).
5. Watch cell 3's output: validation loss should drop and not climb at the end. If you see out-of-memory errors, lower `per_device_train_batch_size` from 4 to 2.
6. The last cell downloads `results_ft_full.zip` to your PC. Don't close the tab before it finishes.

### Option 2: Kaggle (if Colab keeps disconnecting)
1. **[Browser]** kaggle.com > Datasets > New Dataset: upload the same 5 files. Name it e.g. `req-classifier-data`.
2. kaggle.com > Code > New Notebook > File > Import Notebook > upload `training/train_lora.ipynb`. Add your dataset (right panel > Add Input).
3. Right panel > Session options: Accelerator **GPU T4 x2**, **Internet On** (needs phone verification the first time).
4. Add this as the first code cell and run it:
   ```python
   !cp /kaggle/input/req-classifier-data/* /kaggle/working/
   %cd /kaggle/working
   ```
5. Run the rest from the top. The final `files.download` cell won't work on Kaggle: skip it and, after the run, download `results_ft_full.zip` from the **Output** tab on the right.

### Optional ablation (do after the main run works)
Run the notebook again 3 times, changing only the settings cell: `N_TRAIN = 500, RUN_NAME = "ft_n500"`, then `1000 / ft_n1000`, then `2000 / ft_n2000`. Skip the GGUF export cell (section 6) for these. Download each zip and copy its `results/ft_n*.json` into your local `results\` folder.

---

## D. Deploy locally and evaluate (~1 h)  [VS Code] + [Ollama]

1. Unzip `results_ft_full.zip` on your PC. Copy these into the project folder: the `gguf_model` folder (contains the `.gguf` file), `loss_ft_full.png`, and `results\ft_full.json`.
2. Check the exact GGUF file name:
   ```powershell
   dir gguf_model
   ```
   If it isn't `unsloth.Q4_K_M.gguf`, edit the `FROM` line in `export\Modelfile` to match.
3. Register the model in Ollama, test it, and evaluate:
   ```powershell
   ollama create req-classifier -f export\Modelfile
   ollama run req-classifier "Subject: Test`nMessage: I was charged twice for invoice INV-1234"
   python run_model.py --model req-classifier --prompt short --out preds\finetuned.jsonl
   python evaluate.py --preds preds\finetuned.jsonl --name finetuned
   python report.py --names baseline finetuned
   ```
   (If you did the ablation, use `python report.py --names baseline finetuned --ablation-prefix ft_n`.)
4. Open the `results\` folder. **Send me** `comparison.md` and the PNG charts, and I'll write the analysis and finalize the README.

---

## E. Plug it into n8n (~1 h)  [Browser] + [VS Code]

Follow `docs\N8N_SWAP.md`: it tells you which n8n nodes to edit (in your browser, in the n8n editor). Then re-run your 5 original test cases from the n8n form and screen-record them. The vague message ("I need help with something.") should still go to Human Review.

---

## F. Publish (~2 h)  [Browser] + [VS Code]

1. **[Browser]** huggingface.co > New Model: upload the `.gguf` file, the `adapter_ft_full` folder contents, and `MODEL_CARD.md` (rename it `README.md` on upload; fill in the `<...>` placeholders first).
2. **[Browser]** huggingface.co > New Space > SDK: Gradio > Hardware: CPU basic (free). Upload the 2 files in `demo\`. Before uploading, edit `GGUF_REPO` in `demo\app.py` to point to the model repo you just made. The first build takes 10+ minutes.
3. **[VS Code]** Push the project to GitHub (`git init`, commit, push). Add `data/raw/`, `.venv/` and the large `.gguf` to `.gitignore`. Put the PNGs from `results\` in the README.

---

## If something breaks
Copy the full error message and tell me which step (for example "C, Colab, cell 3"). The training notebook uses Unsloth and TRL, whose APIs change often, so a small fix there is normal.
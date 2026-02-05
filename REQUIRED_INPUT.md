# 🔑 REQUIRED USER INPUT SUMMARY

## What You Need to Provide

### 1. Kaggle API Credentials (REQUIRED)

**Why needed**: To download the E. coli resistance dataset

**How to get**:
1. Create account at https://www.kaggle.com (free)
2. Go to: Profile → Settings → API
3. Click "Create New API Token"
4. Downloads `kaggle.json` file

**Where to place it**:
```
Windows: C:\Users\cagns\.kaggle\kaggle.json
```

**Quick setup (PowerShell)**:
```powershell
mkdir $env:USERPROFILE\.kaggle
Move-Item -Path $env:USERPROFILE\Downloads\kaggle.json -Destination $env:USERPROFILE\.kaggle\
```

---

### 2. Dataset Name (REQUIRED)

**Why needed**: The code needs to know which Kaggle dataset to download

**How to find**:
1. Go to https://www.kaggle.com/datasets
2. Search: "E. coli antibiotic resistance" or "E. coli MIC"
3. Find dataset with:
   - E. coli isolates
   - MIC values
   - Resistance phenotypes (S/I/R)
   - Antibiotic names

**Example datasets** (search for these):
- "E. coli AMR data"
- "Bacterial antibiotic resistance"
- "E. coli resistance phenotypes"

**Where to put it**:
Edit file: `config.py` (line 21)

```python
# Change this line:
KAGGLE_DATASET = "your-username/ecoli-resistance-dataset"

# To actual dataset (example):
KAGGLE_DATASET = "johndoe/ecoli-amr-data"
```

**Format**: Always `username/dataset-name` (from URL)

---

### 3. Nothing Else! 🎉

**BVBRC API**: No credentials needed - public access

**Python packages**: Install automatically with `pip install -r requirements.txt`

**Other settings**: Already configured with sensible defaults

---

## Quick Verification

Run this to check if everything is set up:

```powershell
cd c:\Users\cagns\Downloads\capstone_project
.\venv\Scripts\Activate.ps1
python check_setup.py
```

This will tell you exactly what's missing!

---

## Setup Time Estimate

- ✅ Kaggle account: 2 minutes
- ✅ Download API token: 30 seconds  
- ✅ Place kaggle.json: 30 seconds
- ✅ Find dataset: 5 minutes
- ✅ Update config.py: 1 minute

**Total: ~10 minutes** ⏱️

---

## What Happens After Setup

Once you provide these two things, the code will **automatically**:

1. ✅ Download resistance data from Kaggle
2. ✅ Download genomic data from BVBRC (no key needed)
3. ✅ Clean and merge datasets
4. ✅ Run PCA and clustering
5. ✅ Train 3 machine learning models
6. ✅ Perform statistical validation
7. ✅ Generate all figures and results

**No further input needed!** 🚀

---

## Alternative: Skip Kaggle API

If you **don't want** to set up Kaggle API:

1. **Manually download** dataset from Kaggle website
2. Place CSV file in: `data/raw/ecoli_resistance.csv`
3. Skip acquisition step:
   ```powershell
   python main.py --steps preprocess explore train validate
   ```

---

## Need Help?

📖 **Detailed guides**:
- [SETUP.md](SETUP.md) - Step-by-step setup instructions
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [README.md](README.md) - Full documentation

🔧 **Test your setup**:
```powershell
python check_setup.py
```

❓ **Common issues**: See "Troubleshooting" section in SETUP.md

---

## Checklist

Before running the pipeline:

- [ ] Kaggle account created
- [ ] `kaggle.json` downloaded
- [ ] `kaggle.json` placed in `C:\Users\cagns\.kaggle\`
- [ ] E. coli dataset found on Kaggle
- [ ] `config.py` updated with dataset name
- [ ] Tested with: `kaggle datasets list`
- [ ] Run: `python check_setup.py` (all checks pass)

✅ **Ready!** Run: `python main.py --steps all`

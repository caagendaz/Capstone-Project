# Setup Instructions - API Keys & Credentials

## Required Setup Before Running

### 1. Kaggle API Setup (REQUIRED for data download)

#### Step 1: Create Kaggle Account
1. Go to https://www.kaggle.com
2. Sign up for a free account (or log in if you have one)

#### Step 2: Get API Credentials
1. Log into Kaggle
2. Click on your profile picture (top right)
3. Select "Settings"
4. Scroll down to "API" section
5. Click "Create New API Token"
6. This downloads `kaggle.json` file

#### Step 3: Install Kaggle Credentials (Windows PowerShell)
```powershell
# Create .kaggle directory in your user profile
mkdir $env:USERPROFILE\.kaggle

# Move the downloaded kaggle.json to the .kaggle directory
Move-Item -Path $env:USERPROFILE\Downloads\kaggle.json -Destination $env:USERPROFILE\.kaggle\

# Verify it's there
cat $env:USERPROFILE\.kaggle\kaggle.json
```

Your `kaggle.json` should look like:
```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key_here"
}
```

#### Step 4: Test Kaggle API
```powershell
# Activate virtual environment first
.\venv\Scripts\Activate.ps1

# Test the API
kaggle datasets list
```

If you see a list of datasets, you're good to go! ✅

---

### 2. Update Dataset Name in config.py (REQUIRED)

You need to find and specify the actual Kaggle dataset for E. coli resistance.

#### Find the Dataset:
1. Go to https://www.kaggle.com/datasets
2. Search for: "E. coli antibiotic resistance" or "E. coli MIC"
3. Find a dataset with resistance phenotypes and MIC values

#### Example datasets to look for:
- "E. coli AMR resistance data"
- "Bacterial antibiotic resistance dataset"
- "E. coli MIC values"

#### Update config.py:
```python
# Open config.py and change this line:
KAGGLE_DATASET = "actual-username/actual-dataset-name"

# Example:
KAGGLE_DATASET = "johndoe/ecoli-resistance-mic-data"
```

The format is always: `username/dataset-name` (found in the dataset URL)

---

### 3. BVBRC API (NO SETUP NEEDED)

Good news! The BVBRC API requires **no authentication** - it's completely open access.

The scripts will automatically download genomic data when you run them.

---

## Quick Setup Checklist

- [ ] Create Kaggle account
- [ ] Download `kaggle.json` API credentials
- [ ] Place `kaggle.json` in `C:\Users\cagns\.kaggle\`
- [ ] Test with `kaggle datasets list`
- [ ] Find E. coli resistance dataset on Kaggle
- [ ] Update `KAGGLE_DATASET` in `config.py`
- [ ] Install Python packages: `pip install -r requirements.txt`

---

## Alternative: Manual Data Download

If you prefer **NOT** to use the Kaggle API, you can manually download data:

### Manual Kaggle Download:
1. Go to the dataset page on Kaggle
2. Click the "Download" button (requires login)
3. Extract the CSV file
4. Place it in `data/raw/ecoli_resistance.csv`

### Skip API in Code:
When running the pipeline, you can skip the acquisition step if you manually downloaded:

```powershell
# Skip data acquisition, start from preprocessing
python main.py --steps preprocess explore train validate
```

---

## Configuration File Reference

### config.py - Settings to Update

```python
# Line ~20: Update with your Kaggle dataset
KAGGLE_DATASET = "USERNAME/DATASET-NAME"  # ← CHANGE THIS

# Optional: Adjust these based on your needs
MIN_SAMPLES_PER_ANTIBIOTIC = 100  # Minimum samples per antibiotic
TEST_SIZE = 0.2                    # 20% test set
RANDOM_STATE = 42                  # For reproducibility

# Optional: Reduce for faster testing
N_PERMUTATIONS = 1000  # Reduce to 100 for quick testing
N_MC_ITERATIONS = 100  # Reduce to 20 for quick testing
N_BOOTSTRAP = 1000     # Reduce to 100 for quick testing
```

---

## Security Notes

### ⚠️ Important: Keep Your API Keys Private!

- **DO NOT** commit `kaggle.json` to git (already in `.gitignore`)
- **DO NOT** share your API key publicly
- **DO NOT** post screenshots showing your API key

If you accidentally expose your key:
1. Go to Kaggle Settings
2. Delete the old token
3. Create a new token

---

## Troubleshooting

### Error: "Could not find kaggle.json"
```powershell
# Check if file exists
Test-Path $env:USERPROFILE\.kaggle\kaggle.json

# If False, follow Step 3 above to place the file
```

### Error: "403 Forbidden" when downloading
- You haven't accepted the dataset's terms
- Go to the dataset page on Kaggle
- Click "Download" or "Accept" to agree to terms
- Try again

### Error: "Dataset not found"
- Check the dataset name in `config.py`
- Make sure format is: `username/dataset-name`
- Verify dataset exists and is public

### Error: "Unauthorized"
- Your `kaggle.json` might be corrupted
- Re-download from Kaggle Settings
- Replace the file in `.kaggle` directory

---

## Testing Your Setup

Run this to verify everything is configured:

```powershell
# Navigate to project
cd c:\Users\cagns\Downloads\capstone_project

# Activate environment
.\venv\Scripts\Activate.ps1

# Test Kaggle API
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print('Kaggle API: OK')"

# Test BVBRC API (no auth needed)
python -c "import requests; r = requests.get('https://www.bv-brc.org/api/genome/?eq(genome_id,511145.183)&limit(1)'); print('BVBRC API: OK' if r.ok else 'BVBRC API: Error')"

# Test imports
python -c "import pandas, numpy, sklearn, xgboost; print('All packages: OK')"
```

If all three print "OK", you're ready to go! 🚀

---

## What Data You'll Download

### From Kaggle (~50-200 MB):
- E. coli isolates with resistance data
- MIC values for various antibiotics
- Resistance phenotypes (S/I/R)
- Testing standards (CLSI/EUCAST)

### From BVBRC (variable size):
- AMR gene annotations (~1-10 MB)
- Genome metadata (~1 MB)
- Optional: FASTA sequences (can be large, GBs)

**Total disk space needed**: ~500 MB - 2 GB

---

## Need Help?

If you're stuck on setup:

1. **Kaggle API Issues**: https://github.com/Kaggle/kaggle-api/issues
2. **BVBRC Issues**: https://www.bv-brc.org/docs/
3. **Python Environment**: Make sure Python 3.8+ is installed

---

## Ready to Run?

Once setup is complete:

```powershell
cd c:\Users\cagns\Downloads\capstone_project
.\venv\Scripts\Activate.ps1
python main.py --steps all --antibiotic CIPROFLOXACIN
```

The program will:
1. ✅ Download data from Kaggle (using your API key)
2. ✅ Download genomic data from BVBRC (no key needed)
3. ✅ Process and merge datasets
4. ✅ Run exploratory analysis
5. ✅ Train models
6. ✅ Validate results

All automatic! 🎉

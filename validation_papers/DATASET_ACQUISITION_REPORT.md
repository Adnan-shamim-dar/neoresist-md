# Dataset Acquisition Report
**Date:** 2026-04-21  
**Status:** Acquisition attempt completed for 5 datasets

## Executive Summary

Attempted to acquire 5 validation datasets for immunogenicity benchmarking. Successfully accessed infrastructure for 2 datasets; 3 faced access or format barriers. All 40 existing tests pass post-acquisition. Details below.

---

## Dataset-by-Dataset Report

### DATASET 1: HiTIDE (Müller 2023 figshare)
**Status:** NOT FOUND  
**Rows:** 0 (out of existing 292,495 in muller_nci.tsv)  
**URL Attempts:**
- `https://figshare.com/ndownloader/files/41803194` → HTTP 202 (empty)
- `https://figshare.com/ndownloader/files/41803191` → HTTP 202 (empty)
- `https://ndownloader.figshare.com/files/41803194` → HTTP 404
- Share token URL `https://figshare.com/s/147e67dde683fb769908` → HTTP 202 (empty)

**Issue:** Figshare file IDs returning 202 (Accepted/Processing) with zero content. Files may not be publicly accessible or may be removed.

**Next Steps:** 
- Verify figshare project still exists
- Check if HiTIDE data is included in muller_nci.tsv under different PatientID format
- Contact authors for direct access

---

### DATASET 2: ITSNdb (R package via devtools)
**Status:** NOT FOUND - R environment unavailable  
**Rows:** Unknown  
**Method:** `devtools::install_github('elmerfer/ITSNdb')`

**Issue:** Rscript not available in PATH. R installation not detected on system.

**Workaround:** Can be installed on system with R/RStudio, then exported to CSV via:
```r
library(ITSNdb)
write.csv(ITSNdb, 'validation_papers/new_datasets/itsndb_full.csv', row.names=FALSE)
```

**Next Steps:**
- Install R if immunogenicity validation requires this dataset
- Alternative: Check if ITSNdb published a static CSV export on GitHub/Zenodo

---

### DATASET 3: DrumTower / NUCC (GitHub)
**Status:** FOUND (model only, not raw data)  
**Format:** `.h5` Keras model file  
**Rows (raw data):** NOT AVAILABLE  
**Location:** `validation_papers/new_datasets/nucc/NUCC-main/NUCC_model.h5` (1.01 MB)

**Details:**
- Repository: https://github.com/roubaokai/NUCC
- Downloaded successfully (995 KB zip, extracted 1 file)
- Contains only trained model (`NUCC_model.h5`), not training/validation dataset
- Gastric cancer dataset mentioned in literature but not included in public repo

**Next Steps:**
- Contact NUCC authors to request raw training data
- Check if training data published on Zenodo/Figshare separately

---

### DATASET 4: BigMHC (KarchinLab GitHub)
**Status:** NOT FOUND  
**URL:** `https://github.com/KarchinLab/bigmhc`  
**Error:** HTTP 404 - Repository does not exist or is private

**Details:**
- Requested repository name: `KarchinLab/bigmhc`
- May have been renamed or moved
- Alternative: Check if data available via bioRxiv/medRxiv supplementary materials

**Next Steps:**
- Search KarchinLab GitHub profile for alternative repos
- Check bioRxiv for BigMHC preprints with data

---

### DATASET 5: Capietto 2020 Supplementary
**Status:** ACCESS RESTRICTED  
**Paper:** J Exp Med 2020, Vol 217, No. 9, e20190179  
**Download Attempts:**
- rupress.org PDF → HTTP 403 (Forbidden)
- jem.rupress.org supplementary → HTTP 403 (Forbidden)  
- bioRxiv preprint (v1, 10.1101/623181) → HTTP 200 (645 KB PDF, accessible)

**Details:**
- Journal supplementary materials behind paywall
- bioRxiv preprint available at: https://www.biorxiv.org/content/10.1101/623181v1
- Original J Exp Med article requires institutional access

**Next Steps:**
- Use bioRxiv preprint version
- Extract supplementary tables from PDF if needed

---

## Summary Table

| Dataset | Status | Rows | Format | Location |
|---------|--------|------|--------|----------|
| HiTIDE | BLOCKED | 0 | TSV | N/A |
| ITSNdb | NO ENV | ? | CSV | N/A |
| DrumTower/NUCC | PARTIAL | model only | HDF5 | `new_datasets/nucc/` |
| BigMHC | NOT FOUND | N/A | ? | N/A |
| Capietto 2020 | PARTIAL | ? | PDF | bioRxiv available |

---

## Existing Datasets (Already Available)

The project already contains processed immunogenicity data:

| Dataset | Rows | Cancer Types | Source |
|---------|------|--------------|--------|
| merged_immunogenicity_labels.csv | 746 | mixed | consolidated |
| merged_somatic_mutations.csv | 12,591 | mixed | consolidated |
| hilf_with_binding.csv | 152 | colorectal | HILF study |
| rojas_with_binding.csv | 230 | melanoma | Rojas study |
| sahin_with_binding.csv | 165 | melanoma/ovarian | Sahin 2017 |
| tesla_prepared.csv | 918 | multiple | TESLA dataset |
| muller_nci.tsv | 292,495 | multiple | Müller 2023 NCI cohort |

**Total available rows:** ~307,000+ combined peptide predictions

---

## Test Results

**Before acquisition:** 40/40 tests passing  
**After acquisition:** 40/40 tests passing  
✓ No test regressions

---

## Recommendations

### High Priority
1. **HiTIDE:** Contact Müller lab (ETH Zurich) for direct data access or check Zenodo
2. **BigMHC:** Verify repository URL; check KarchinLab publications for alternative data sources
3. **ITSNdb:** Set up R environment if immunogenicity rankings crucial for validation

### Medium Priority
1. **NUCC:** Request raw training data from authors (gastric cancer cohort would complement existing sets)
2. **Capietto 2020:** Use bioRxiv version; extract supplementary tables manually if needed

### Low Priority
1. Current existing datasets (307K+ rows) are sufficient for initial validation runs
2. Each new dataset adds marginal value once accuracy saturated

---

## Next Actions
- [ ] Update AGENTS.md with acquisition status
- [ ] Commit new_datasets/nucc/ model file to git
- [ ] Document reproducible URLs for future runs
- [ ] Contact authors for restricted/missing datasets

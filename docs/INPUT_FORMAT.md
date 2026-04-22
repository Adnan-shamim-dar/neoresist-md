# Upload Input Format Specification

This document defines the exact column contract for single-patient upload in NeoResist-MD.

## Supported file types

- `.maf` (tab-separated MAF-style table)
- `.tsv`, `.csv`, `.txt` (delimited mutation tables)
- `.vcf`, `.vcf.gz` (normalized through the VCF normalizer)

## Canonical schemas

The upload step auto-detects format and maps common header aliases to canonical names.

### 1) Predictor-ready table (already expanded peptide input)

Required columns:

- `gene_name`
- `protein_sequence`
- `mutation_position`
- `normal_amino_acid`
- `mutant_amino_acid`

Optional columns:

- Any additional columns are preserved but not required for acceptance.

Behavior when optional columns are missing:

- Upload still succeeds if all required columns are present.
- HLA is treated as unknown unless an HLA-like column is provided through downstream normalization paths.

### 2) MAF-style mutation table

Required columns:

- `Hugo_Symbol`
- `Chromosome`
- `Start_Position`
- `Reference_Allele`
- `Tumor_Seq_Allele2`

Optional columns:

- `Tumor_Sample_Barcode`
- `HGVSp_Short`
- `t_ref_count`
- `t_alt_count`
- `VAF`
- `HLA_Allele`

Behavior when optional columns are missing:

- If `Tumor_Sample_Barcode` is missing, sample id falls back to filename stem.
- If `HGVSp_Short` is missing, amino-acid change parsing falls back to deterministic placeholder logic.
- If `t_ref_count`/`t_alt_count` are missing, VAF-based and count-based confidence signals may be reduced.
- If only `VAF` is present, pseudo counts are generated internally for compatibility.
- If HLA is missing or invalid, binding is marked as unavailable where required and UI warns.

### 3) Minimal TSV/CSV mutation table

Required columns:

- `chrom`
- `pos`
- `ref`
- `alt`

Optional columns:

- `gene`
- `protein_change`
- `ref_count`
- `alt_count`
- `vaf`
- `sample_id`
- `hla_allele`

Behavior when optional columns are missing:

- Upload still succeeds if required genomic columns are present.
- Missing `gene` defaults to placeholder values in predictor conversion.
- Missing counts (`ref_count`/`alt_count`) reduces VAF-derived flags unless `vaf` is provided.
- Missing `sample_id` falls back to filename stem.
- Missing/unknown HLA triggers warning and pan-allele fallback behavior where applicable.

## Auto-mapping aliases

Common aliases are mapped automatically and shown in the upload validation checklist.

Examples:

- `Gene` -> `Hugo_Symbol` (MAF) or `gene` (minimal TSV path)
- `AF` -> `VAF` / `vaf`
- `Sample` -> `Tumor_Sample_Barcode` / `sample_id`
- `Chr` -> `Chromosome` / `chrom`
- `Position` -> `Start_Position` / `pos`

## Validation failure behavior

If critical columns cannot be mapped, upload fails with a clear error listing:

- Missing canonical columns
- Expected format type
- Required column set for that format

## Example rows

### Example MAF row (tab-separated)

```text
Hugo_Symbol	Chromosome	Start_Position	Reference_Allele	Tumor_Seq_Allele2	Tumor_Sample_Barcode	HGVSp_Short	t_ref_count	t_alt_count	HLA_Allele
TP53	17	7674220	C	T	PATIENT_001	p.R175H	62	38	HLA-A*02:01
```

### Example minimal TSV row (tab-separated)

```text
chrom	pos	ref	alt	gene	protein_change	vaf	sample_id	hla_allele
17	7674220	C	T	TP53	p.R175H	0.38	PATIENT_001	HLA-A*02:01
```

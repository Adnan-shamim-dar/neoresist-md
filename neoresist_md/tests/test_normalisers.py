from __future__ import annotations

from pathlib import Path

import pandas as pd

from neoresist_md.backend.normalisers.maf_normaliser import MAFNormaliser
from neoresist_md.backend.normalisers.tsv_normaliser import TSVNormaliser
from neoresist_md.backend.normalisers.vcf_normaliser import VCFNormaliser


def test_tsv_normaliser_mapping(tmp_path: Path):
    path = tmp_path / "input.tsv"
    pd.DataFrame(
        [{"gene_name": "TP53", "chromosome": "17", "position": 7579472, "ref_allele": "C", "alt_allele": "T"}]
    ).to_csv(path, sep="\t", index=False)
    out = TSVNormaliser().normalise(
        path,
        column_mapping={
            "gene": "gene_name",
            "chrom": "chromosome",
            "pos": "position",
            "ref": "ref_allele",
            "alt": "alt_allele",
        },
    )
    assert len(out) == 1
    assert set(["gene", "chrom", "pos", "ref", "alt"]).issubset(set(out.columns))


def test_maf_normaliser_vaf(tmp_path: Path):
    path = tmp_path / "sample.maf"
    pd.DataFrame(
        [
            {
                "Hugo_Symbol": "TP53",
                "Chromosome": "17",
                "Start_Position": 7579472,
                "Reference_Allele": "C",
                "Tumor_Seq_Allele2": "T",
                "t_ref_count": 30,
                "t_alt_count": 10,
            }
        ]
    ).to_csv(path, sep="\t", index=False)
    out = MAFNormaliser().normalise(path)
    assert float(out.iloc[0]["vaf"]) > 0


def test_vcf_normaliser_multiallelic(tmp_path: Path):
    path = tmp_path / "sample.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
        "1\t12345\t.\tA\tC,G\t.\tPASS\t.\tGT:AD:DP\t0/1:12,5,3:20\n",
        encoding="utf-8",
    )
    out = VCFNormaliser().normalise(path)
    assert len(out) == 2

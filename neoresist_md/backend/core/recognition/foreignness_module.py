from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..base_module import BaseModule
from ..base_module import ValidationResult

# SCIENTIFIC NOTE: self_dissimilarity uses BLOSUM62 alignment against human peptide reference.
# Replaces hash-based proxy (biologically invalid). Based on Luksza et al., Nature 2017.

LOGGER = logging.getLogger(__name__)

# REFERENCE_PEPTIDES — replace with full proteome for production.
REFERENCE_PEPTIDES = [
    "AAGIGILTV", "ALWGFFPVL", "AMKVGTLAA", "APRGPHGGA", "ATDALMTGF", "AVFDRKSDA", "AVGGKSKKL",
    "AVIRGKSNA", "AVKELLEQL", "AVKTRGDEL", "AVLDIYYYM", "AVLEYLTAE", "AVLQSGFRK", "AVMDDFAAF",
    "AVMELIRMI", "AVMLYQGQL", "AVSLQANRW", "AVTFDLEQL", "AVVQDPALK", "AYDGKDYIA", "AYEQLLSFF",
    "AYLPVNESF", "AYQKVVAGV", "AYVHMVTHF", "CINGVCWTV", "CLGGLLTMV", "CLLWSFQTS", "CTELKLSDY",
    "CVADYQQGV", "CVNGSCFTV", "DLATVYVDV", "DLGEEHFKG", "DLLVSSSTD", "DLQHGSLFL", "DLYANTVLS",
    "DMMQKIRSL", "DTDTGALLF", "DVFAGKNAD", "DVKDLVQTL", "DVWLSNQTV", "DYFLRSLPL", "EAYVTMDSG",
    "EILKGGYTL", "EIVDLIQKV", "ELAGIGILT", "ELFSYLIEK", "ELLDFVGAV", "ELLQKMQTY", "ELLSLMDAN",
    "ELVDSLNQL", "ELVNQIIQL", "ELYQKTLVV", "EPEYLQYLG", "EPHVSNDSL", "EQKLISEED", "ETVEGDKAT",
    "EVLQNAKTY", "EVMDFLQSL", "EVQKAKLAE", "EYGFQNALI", "EYLVSFGVW", "FIDSYICQV", "FIKTNMDEV",
    "FILDGLSGV", "FIMDKHFSV", "FIRGTLVGL", "FLAHIQWMV", "FLDLEPGTM", "FLEKTRQVL", "FLLDYYNAT",
    "FLLDQIPYL", "FLLTCVATV", "FLSKFGNIL", "FLTSVINRV", "FLWGPRALV", "FQDGNILLA", "FQDVQKAIL",
    "FQLNQKQTL", "FQQQSLVSV", "FRDNLTQAF", "FTLTLQQLV", "FVAVKQTVG", "FVDGVPFVV", "FVGDSVVQV",
    "FVIDDKNKV", "FVLHSYFTA", "FVLKHLNPM", "FVTESNQAI", "FYAEGSRGS", "FYLQKCSQV", "GAFQDVESV",
    "GALQNIIPI", "GAVDPLLAL", "GAYQKVVDL", "GDLGTLNQL", "GEVQQLRQV", "GILGFVFTL", "GIMVFVLNV",
    "GLCTLVAML", "GLDYYISRV", "GLEQLESLI", "GLLGTLNQV", "GLLQKASDV", "GLLVLPQLV", "GLMWLSYFV",
    "GLYDGMEHL", "GQIVADQRL", "GQMVHQAIS", "GRTGAGKSF", "GSLQYLALA", "GTLDQWQSV", "GTMNNRITV",
    "GVANALKTL", "GVDPFLVSV", "GVLKEYGVV", "GVMDVPVAI", "GVQNLLRAL", "GYLQPRTFL", "HLAELVQTV",
    "HMMNQIRTL", "HVLFGKDLI", "IAMDKNIIL", "IAVEEKNLP", "IAYQKVVQL", "IDKLTSESV", "IEEYLQAFL",
    "IETDKESKV", "IFLQKATSL", "IFMCLSYDV", "IFVQKCAQV", "IGILTLKRS", "IGMEVTPSG", "IILQKATQV",
    "IIMDVVQSV", "IITQSVQVL", "IKAAGHYAV", "IKDQIIATL", "IKEKYEGTL", "IKLMVTHVW", "ILAEQSVAA",
    "ILEKANKTL", "ILGADTSVD", "ILKEPVHGV", "ILNQKQVEL", "ILQVGQVEL", "ILRGSVAHK", "ILSLILQSI",
    "ILTQSPAIL", "ILVEPLQSV", "IMDKNIILF", "IMDQVPFSV", "IMDQVPFSV", "IMGDQLKSL", "IMLDESESV",
    "IMQKAVAAV", "INAYQKVVD", "INDPFLVKL", "INFEKLQQA", "INLQKAAEV", "IPGFGNVVL", "IPQCRLTQK",
    "IQAEPDGSN", "IQDYLQHSL", "IQKDGSLHV", "IQNLLQTFV", "IQTSESVNV", "IRKQMNDAA", "ISDEFSSNV",
    "ISFQKALDL", "ISLQKSNAA", "ISTDANLQL", "ITDQVPFSV", "ITQSLYQKV", "IVDQLGMLV", "IVLQKATGI",
    "IVMELIRMI", "IVMQSKGSL", "IVTDFSVIK", "IYDKNIIAF", "IYQKSGSLL", "KALQDVANV", "KAVYNFATM",
    "KAYQATQAL", "KCYGVSPTK", "KFQDVKQTL", "KIFGSLAFL", "KLFEKGGNA", "KLGGALQAK", "KLIANQATK",
    "KLLEIAPNC", "KLLQDVANV", "KLPDDFTGCV", "KLTPLCVTL", "KLVVGAVGV", "KMDSFLDMQL", "KQFQKETLF",
    "KQFLDTVQL", "KQFLSNAAQ", "KQYIKWPWY", "KSYGVSPTK", "KTIQNQKAM", "KTLQAIDQK", "KVMDEAHFT",
    "KVLEYVIKV", "KVQQTVQCF", "KVYQDVNLS", "LAGIGILTV", "LATVSVNKV", "LAYQKATLI", "LCVQSTHVD",
    "LDGAYVDFS", "LDKVEAEHV", "LDQFLDSNS", "LEQLESLII", "LEQSVQKII", "LFDQLRANV", "LFDYGAFSV",
    "LFGYPVYVF", "LFLDGIDKA", "LFLQFGAQG", "LFTQSPAIL", "LGATVVQKV", "LGDYYISKV", "LGQDPYVKV",
    "LIGILTLAA", "LIIANELVI", "LILGLLTKV", "LIMDQQKTL", "LLDYYIENV", "LLFGYPVYV", "LLGATCMFV",
    "LLIIVLQSV", "LLLQKATLI", "LLQDSVDFS", "LLQKATQAV", "LLQKVSNQL", "LLSRHIVTK", "LLTQSPAIL",
    "LMDAQTQSL", "LMGQFYVMV", "LMLDFSQKV", "LQDVANVKL", "LQFAYANRV", "LQGATCMFV", "LQKATQAVL",
    "LQNVVNQSL", "LQQAIVQKV", "LRDYYITKV", "LSDQSVLTL", "LSQPKIVKW", "LTFDQLNKV", "LTFKEYGSV",
    "LTLQSLQAV", "LTMQTYVSV", "LVANQVKTL", "LVDDFQKTI", "LVEEVPQLV", "LVGATCMFV", "LVLDFAPPG",
    "LVNQVKSVL", "LVQDQVFTM", "LVQKAKLAE", "LVQNVVNQSL", "LVSFGVWIR", "LVTQYLQSV", "LVYQDVNLI",
    "LYDFFVSQL", "LYQKATQSV", "MDNLLQSVA", "MEVTPSGTW", "MIFQVPFSL", "MLDQVPFSV", "MLDRSLKIV",
    "MLLAVLYCL", "MLLSVPLLL", "MLQDYSVSV", "MLTQGTTLQ", "MMWDRGLGMM", "MQDDFLSKL", "MSDVEVLEA",
    "MTEQEVSAA", "MVATVQGQV", "MVDPKQSLV", "MVLSQKATV", "MVMELIRMI", "MVTDFSVIK", "NAQDVKQTL",
    "NAYQKATLI", "NFEKLQQAV", "NIDQTVAAL", "NILGATCMF", "NLDTLMTYV", "NLEQSVQKI", "NLQKATQSV",
    "NLVPMVATV", "NMDQVPFSV", "NQKATQAVL", "NQKLIANQAT", "NSVQKATLI", "NTDVEVLEA", "NVEQLESLI",
    "NVHDSDLSL", "NVQKATQAV", "NYQKATQSV", "PIVQNLQSL", "PLDQLSAVV", "PLFQGKDLI", "PLLQKATLI",
    "PMVATVQSS", "QAVYNFATM", "QFAYANRVI", "QFQKETLFK", "QGQDPYVKV", "QIIQKATLI", "QILPDQSVV",
    "QLGATCMFV", "QLLQKATLI", "QLLQKVSNQ", "QLVFGKDLI", "QMQTYVSVI", "QQQSLVSVP", "QSVQKATLI",
    "QTFDQLNKV", "QYIKWPWYI", "RAKFKQLLQ", "RLFQGKDLI", "RLMDQQKTL", "RLSQPKIVK", "RLTQSPAIL",
    "RTDVEVLEA", "SALQDVANV", "SAYQKATLI", "SLLMWITQC", "SLYNTVATL", "SPQGRVMTI", "SVDDFQKTI",
    "SVFQKATLI", "SVLQKATLI", "SVQKATLIL", "SYFPEITHI", "TAFTIPSIK", "TLGATCMFV", "TLNAWVKVV",
    "TLQKATQAV", "TMVQNLQSL", "TSYGFQNAL", "TTDPSFLGR", "TVDDFQKTI", "TVQKATLIL", "TYQKATQSV",
    "VADQDVQTL", "VAFTIPSIK", "VANQVKTLV", "VCATVQGQV", "VDDFQKTIL", "VDQVPFSVL", "VEAEHVVFL",
    "VEQLESLII", "VFQKATLIL", "VGATCMFVV", "VLDDFQKTI", "VLFQGKDLI", "VLQKATLIL", "VLSQKATLI",
    "VLYNTVATL", "VMAPRTLIL", "VMDQVPFSV", "VMELIRMIL", "VQGQDVQTL", "VQKATLILV", "VQQTVQCFL",
    "VTDFSVIKK", "VYGFQNALI", "VYQKATQSV", "YLEPGPVTA", "YLQPRTFLL", "YLQSVLEGV", "YQKATQSVL",
]


class ForeignnessModule(BaseModule):
    NAME = "recognition"
    INPUT_COLUMNS = ["mutant_peptide", "wildtype_peptide"]
    OUTPUT_COLUMNS = [
        "self_dissimilarity",
        "mutant_wt_distance",
        "recognition_score",
        "recognition_tool",
        "recognition_confidence",
    ]

    def validate_inputs(self, df: pd.DataFrame) -> ValidationResult:
        missing = [c for c in ["mutant_peptide"] if c not in df.columns]
        return ValidationResult(valid=len(missing) == 0, missing_cols=missing)

    @staticmethod
    def _safe_peptide(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value or "").strip().upper()

    @staticmethod
    def _blosum62():
        from Bio.Align import substitution_matrices

        return substitution_matrices.load("BLOSUM62")

    @staticmethod
    def _build_aligner(matrix):
        from Bio.Align import PairwiseAligner

        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.substitution_matrix = matrix
        aligner.open_gap_score = -11.0
        aligner.extend_gap_score = -1.0
        return aligner

    @staticmethod
    def _alignment_score(aligner, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return float(aligner.score(left, right))

    @staticmethod
    def _normalized_dissimilarity(score: float, max_possible_score: float) -> float:
        if max_possible_score <= 0:
            return 0.0
        ratio = score / max_possible_score
        # Keep negative alignment signal informative instead of collapsing to full dissimilarity.
        similarity = max(0.0, min(1.0, (ratio + 1.0) / 2.0))
        return round(max(0.0, min(1.0, 1.0 - similarity)), 6)

    @staticmethod
    def _reference_file() -> Path:
        return Path(__file__).resolve().parents[3] / "data" / "reference" / "human_9mers_reference.tsv"

    @classmethod
    def _load_reference_peptides(cls) -> list[str]:
        path = cls._reference_file()
        if path.is_file():
            try:
                ref = pd.read_csv(path, sep="\t")
                for column in ("peptide", "human_9mer", "sequence"):
                    if column in ref.columns:
                        peptides = [cls._safe_peptide(v) for v in ref[column].tolist()]
                        peptides = [pep for pep in peptides if len(pep) == 9]
                        if peptides:
                            return list(dict.fromkeys(peptides))
            except Exception as exc:
                LOGGER.warning("Failed to load human 9mer reference from %s: %s", path, exc)
        return list(dict.fromkeys(REFERENCE_PEPTIDES))

    @classmethod
    def _best_reference_score(cls, aligner, mutant: str, reference_peptides: list[str]) -> float:
        if not mutant:
            return 0.0
        best = None
        for reference in reference_peptides:
            score = cls._alignment_score(aligner, mutant, reference)
            if best is None or score > best:
                best = score
        return float(best or 0.0)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.OUTPUT_COLUMNS:
            if col not in out.columns:
                out[col] = None

        matrix = self._blosum62()
        aligner = self._build_aligner(matrix)
        reference_peptides = self._load_reference_peptides()
        tool = "blosum62_alignment"

        if "genome_build" not in out.columns:
            out["genome_build"] = None
        if out["genome_build"].isna().all():
            manifest_build = None
            case_id = None
            for key in ("case_id", "run_id"):
                if key in out.columns:
                    values = out[key].dropna().astype(str)
                    if not values.empty:
                        case_id = values.iloc[0]
                        break
            if case_id:
                try:
                    from neoresist.case_store import read_case_manifest

                    manifest_build = str(read_case_manifest(case_id).get("genome_build") or "").strip() or None
                except Exception as exc:
                    LOGGER.warning("Could not resolve genome_build from manifest for %s: %s", case_id, exc)
            if not manifest_build:
                LOGGER.warning("genome_build not specified; defaulting recognition output to GRCh38.")
            out["genome_build"] = out["genome_build"].fillna(manifest_build or "GRCh38")
        else:
            out["genome_build"] = out["genome_build"].fillna("GRCh38")

        mt_vals = []
        self_vals = []
        rec_vals = []
        conf_vals = []
        for _, row in out.iterrows():
            m = self._safe_peptide(row.get("mutant_peptide"))
            w = self._safe_peptide(row.get("wildtype_peptide"))
            if not m:
                mt_vals.append(None)
                self_vals.append(None)
                rec_vals.append(None)
                conf_vals.append("UNAVAILABLE")
                continue
            max_possible_score = self._alignment_score(aligner, m, m)
            if w:
                mutant_wt_score = self._alignment_score(aligner, m, w)
                mt_dist = self._normalized_dissimilarity(mutant_wt_score, max_possible_score)
            else:
                mt_dist = None
            best_reference_score = self._best_reference_score(aligner, m, reference_peptides)
            self_dist = self._normalized_dissimilarity(best_reference_score, max_possible_score)
            if mt_dist is None:
                rec = round(max(0.0, min(1.0, self_dist)), 6)
            else:
                rec = round(max(0.0, min(1.0, 0.4 * mt_dist + 0.6 * self_dist)), 6)
            length_ok = 8 <= len(m) <= 11
            if not length_ok:
                confidence = "LOW"
            elif w:
                confidence = "HIGH"
            else:
                confidence = "MEDIUM"
            mt_vals.append(mt_dist)
            self_vals.append(self_dist)
            rec_vals.append(rec)
            conf_vals.append(confidence)

        out["mutant_wt_distance"] = mt_vals
        out["self_dissimilarity"] = self_vals
        out["recognition_score"] = rec_vals
        out["recognition_tool"] = tool
        out["recognition_confidence"] = conf_vals
        return out

"""Bio-Hermes-001 data preprocessing pipeline for NeuroFusion-AD.

Loads Bio-Hermes-001 SDTM files, extracts plasma biomarkers, demographics,
acoustic/motor features, and amyloid classification label.
Applies ADNI-fitted StandardScaler (does NOT refit).

Document traceability:
    DRD-001 § 3.2 — Bio-Hermes-001 data processing requirements
    SRS-001 § 4.2 — Input feature specifications

NOTE: Bio-Hermes-001 is cross-sectional. Only amyloid classification label
is available. No MMSE slope or survival labels.
PHI: USUBJID hashed before logging.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

log = structlog.get_logger(__name__)

RAW_DIR           = Path("data/raw/biohermes/BIOHERMES001")
PROCESSED_DIR     = Path("data/processed/biohermes")
ADNI_PROCESSED_DIR = Path("data/processed/adni")

# File paths (verified by data-explorer-agent 2026-03-11)
LILLY_PTAU217_FILE = (
    RAW_DIR / "BloodBasedBiomarkers/Lilly/pTau217_Data/LB_LILLY_CLINICAL_DIAGNOST.csv"
)
ROCHE_FILE         = RAW_DIR / "BloodBasedBiomarkers/Roche Diagnostics/LB_ROCHE.csv"
QUANTERIX_FILE     = RAW_DIR / "BloodBasedBiomarkers/Quanterix/LB_QUANTERIX_CORPORATION.csv"
DM_FILE            = RAW_DIR / "GAP-Clinical/DM.csv"
SC_FILE            = RAW_DIR / "GAP-Clinical/SC.csv"
NV_FILE            = RAW_DIR / "GAP-Clinical/NV.csv"
FT_FILE            = RAW_DIR / "GAP-Clinical/FT.csv"
AURAL_FILE         = RAW_DIR / "DigitalTests/Aural Analytics/FT_AURAL_ANALYTICS.csv"
LINUS_FILE         = RAW_DIR / "DigitalTests/Linus/FT_LINUS_HEALTH.csv"


def _hash_uid(uid: str) -> str:
    """Hash patient USUBJID for safe logging (PHI compliance)."""
    return hashlib.sha256(str(uid).encode()).hexdigest()[:12]


def _pivot_sdtm(
    df: pd.DataFrame,
    testcd_col: str,
    value_col: str,
    id_col: str = "USUBJID",
    visitnum: int = 1,
) -> pd.DataFrame:
    """Pivot an SDTM long-format lab/functional test file to wide format.

    Filters to VISITNUM == visitnum (baseline), pivots testcd -> column.

    Args:
        df: Long-format SDTM dataframe.
        testcd_col: Column containing test codes (e.g., 'LBTESTCD').
        value_col: Column containing numeric results (e.g., 'LBSTRESN').
        id_col: Patient ID column.
        visitnum: Baseline visit number (default 1).

    Returns:
        Wide-format DataFrame with one row per patient.
    """
    if "VISITNUM" in df.columns:
        df = df[df["VISITNUM"] == visitnum].copy()
    # Drop duplicates per subject+test
    df = df.drop_duplicates(subset=[id_col, testcd_col], keep="first")
    wide = df.pivot(index=id_col, columns=testcd_col, values=value_col)
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def load_ptau217() -> pd.DataFrame:
    """Load pTau-217 from Lilly immunoassay file.

    Filter: LBTESTCD == 'TAU217P'
    Value: LBSTRESN

    Returns:
        DataFrame with USUBJID and PTAU217 columns.
    """
    df = pd.read_csv(LILLY_PTAU217_FILE)
    df_ptau = df[df["LBTESTCD"] == "TAU217P"].copy()
    df_ptau = df_ptau.drop_duplicates("USUBJID", keep="first")
    df_ptau = df_ptau.rename(columns={"LBSTRESN": "PTAU217"})[["USUBJID", "PTAU217"]]
    log.info("Loaded pTau217", n=len(df_ptau))
    return df_ptau


def load_roche_panel() -> pd.DataFrame:
    """Load Roche plasma panel: NfL, GFAP, Abeta42, Abeta40, pTau181.

    SDTM format: filter LBTESTCD in (NFLP, GFAP, AMYLB40, AMYLB42, TAU181P)

    Returns:
        Wide DataFrame with USUBJID and one column per analyte.
    """
    df = pd.read_csv(ROCHE_FILE)
    codes_of_interest = {"NFLP", "GFAP", "AMYLB40", "AMYLB42", "TAU181P"}
    df = df[df["LBTESTCD"].isin(codes_of_interest)].copy()
    wide = _pivot_sdtm(df, "LBTESTCD", "LBSTRESN")
    # Rename columns to NeuroFusion schema
    rename = {
        "NFLP":    "NFL_PLASMA",
        "GFAP":    "GFAP_PLASMA",
        "AMYLB42": "ABETA42_PLASMA",
        "AMYLB40": "ABETA40_PLASMA",
        "TAU181P": "PTAU181_PLASMA",
    }
    wide = wide.rename(columns={k: v for k, v in rename.items() if k in wide.columns})
    # Compute ratio if both columns present
    if "ABETA42_PLASMA" in wide.columns and "ABETA40_PLASMA" in wide.columns:
        wide["ABETA4240_RATIO"] = wide["ABETA42_PLASMA"] / wide["ABETA40_PLASMA"].replace(0, np.nan)
    log.info("Loaded Roche panel", n=len(wide))
    return wide


def load_demographics() -> pd.DataFrame:
    """Load Bio-Hermes-001 demographics.

    Source: GAP-Clinical/DM.csv
    Columns: USUBJID, AGE, SEX, RACE, ETHNIC

    Returns:
        DataFrame with USUBJID, AGE, SEX_CODE (0=Female, 1=Male), RACE.
    """
    df = pd.read_csv(DM_FILE)
    df["SEX_CODE"] = df["SEX"].map({"M": 1, "F": 0}).fillna(-1).astype(int)
    log.info("Loaded Bio-Hermes-001 demographics", n=len(df))
    return df[["USUBJID", "AGE", "SEX_CODE", "RACE"]].drop_duplicates("USUBJID")


def load_education() -> pd.DataFrame:
    """Load years of education from SC.csv.

    Filter: SCTESTCD == 'EDUYRNUM'; value in SCSTRESC.

    Returns:
        DataFrame with USUBJID and EDUCATION_YEARS.
    """
    df = pd.read_csv(SC_FILE)
    df_edu = df[df["SCTESTCD"] == "EDUYRNUM"].copy()
    df_edu = df_edu.drop_duplicates("USUBJID", keep="first")
    df_edu["EDUCATION_YEARS"] = pd.to_numeric(df_edu["SCSTRESC"], errors="coerce")
    return df_edu[["USUBJID", "EDUCATION_YEARS"]]


def load_amyloid_label() -> pd.DataFrame:
    """Load amyloid classification ground truth.

    Source: GAP-Clinical/NV.csv
    Filter: NVTESTCD == 'AMYCLAS'
    Label: NVSTRESC ('POSITIVE'/'NEGATIVE') -> 1/0

    Returns:
        DataFrame with USUBJID and AMYLOID_POSITIVE (0/1 float).
    """
    df = pd.read_csv(NV_FILE)
    df_amy = df[df["NVTESTCD"] == "AMYCLAS"].copy()
    df_amy = df_amy.drop_duplicates("USUBJID", keep="first")
    df_amy["AMYLOID_POSITIVE"] = (df_amy["NVSTRESC"].str.upper() == "POSITIVE").astype(float)
    log.info("Loaded amyloid labels",
             n=len(df_amy),
             n_positive=int(df_amy["AMYLOID_POSITIVE"].sum()))
    return df_amy[["USUBJID", "AMYLOID_POSITIVE"]]


def load_mmse_baseline() -> pd.DataFrame:
    """Load baseline MMSE total score from GAP-Clinical/FT.csv.

    Filter: FTTESTCD == 'MMS112' (MMSE Total Score), VISITNUM == 1

    Returns:
        DataFrame with USUBJID and MMSE_BASELINE.
    """
    df = pd.read_csv(FT_FILE)
    df_mmse = df[(df["FTTESTCD"] == "MMS112") & (df["VISITNUM"] == 1)].copy()
    df_mmse = df_mmse.drop_duplicates("USUBJID", keep="first")
    df_mmse = df_mmse.rename(columns={"FTSTRESN": "MMSE_BASELINE"})[["USUBJID", "MMSE_BASELINE"]]
    return df_mmse


def load_acoustic_features() -> pd.DataFrame:
    """Load acoustic digital biomarker features from Aural Analytics.

    Source: DigitalTests/Aural Analytics/FT_AURAL_ANALYTICS.csv
    Quality filter: FTACPTFL == 'Y'
    Feature codes (from data-explorer-agent verified inventory):
        CA65CBDF -> speaking_rate
        MAD174CB -> pause_rate (jitter proxy)
        MAC168D2 -> monotonicity (shimmer proxy)
        CA55A559 -> intraword_pause
        CA67FDBA -> verbal_fluency
        CA6E2CA5 -> image_speaking_rate
        CA37153B -> delayed_story_recall
        CA52304B -> object_recall
        CA5550C4 -> image_descr_score
        MA8D60D3 -> naming_total_dur

    Returns:
        Wide DataFrame with USUBJID and one column per acoustic feature.
    """
    df = pd.read_csv(AURAL_FILE)
    # Quality filter — only apply if the column has actual 'Y'/'N' values
    if "FTACPTFL" in df.columns and df["FTACPTFL"].notna().any():
        df = df[df["FTACPTFL"] == "Y"].copy()
    else:
        df = df.copy()
    feature_codes = {
        "CA65CBDF": "acoustic_speaking_rate",
        "MAD174CB": "acoustic_pause_rate",
        "MAC168D2": "acoustic_monotonicity",
        "CA55A559": "acoustic_intraword_pause",
        "CA67FDBA": "acoustic_verbal_fluency",
        "CA6E2CA5": "acoustic_image_speaking_rate",
        "CA37153B": "acoustic_delayed_recall",
        "CA52304B": "acoustic_object_recall",
        "CA5550C4": "acoustic_image_descr_score",
        "MA8D60D3": "acoustic_naming_duration",
    }
    df = df[df["FTTESTCD"].isin(feature_codes.keys())].copy()
    wide = _pivot_sdtm(df, "FTTESTCD", "FTSTRESN")
    wide = wide.rename(columns=feature_codes)
    n_acoustic = len([c for c in wide.columns if c.startswith("acoustic_")])
    log.info("Loaded acoustic features", n=len(wide), n_features=n_acoustic)
    return wide


def load_motor_features() -> pd.DataFrame:
    """Load motor/cognitive digital biomarker features from Linus Health.

    Source: DigitalTests/Linus/FT_LINUS_HEALTH.csv
    Quality filter: FTACPTFL == 'Y'
    Feature codes (from data-explorer-agent verified inventory):
        SPCWDTM, SPCWNTM, SPCCWDTM, SPCCWNTM -> spiral drawing times
        TRLSTIME, TRLSACC -> Trails B
        SDMTACC, SDMTATTP -> SDMT
        DNMNRT, DNMEDRT, DNMNTHRU, DNPCTCOR -> digit naming
        DCRSCR, DCRDCTSC, DCRDLRCL -> DCR

    Returns:
        Wide DataFrame with USUBJID and one column per motor feature.
    """
    df = pd.read_csv(LINUS_FILE)
    # Quality filter — only apply if the column has actual 'Y'/'N' values
    if "FTACPTFL" in df.columns and df["FTACPTFL"].notna().any():
        df = df[df["FTACPTFL"] == "Y"].copy()
    else:
        df = df.copy()
    feature_codes = {
        "SPCWDTM":  "motor_spiral_cw_dom",
        "SPCWNTM":  "motor_spiral_cw_nondom",
        "SPCCWDTM": "motor_spiral_ccw_dom",
        "SPCCWNTM": "motor_spiral_ccw_nondom",
        "TRLSTIME": "motor_trails_b_time",
        "TRLSACC":  "motor_trails_b_acc",
        "SDMTACC":  "motor_sdmt_acc",
        "SDMTATTP": "motor_sdmt_attempted",
        "DNMNRT":   "motor_digit_naming_rt",
        "DNMEDRT":  "motor_digit_naming_median_rt",
        "DNMNTHRU": "motor_digit_naming_throughput",
        "DNPCTCOR": "motor_digit_naming_pct_correct",
        "DCRSCR":   "motor_dcr_score",
        "DCRDCTSC": "motor_dcr_clock_score",
        "DCRDLRCL": "motor_dcr_delayed_recall",
    }
    df = df[df["FTTESTCD"].isin(feature_codes.keys())].copy()
    wide = _pivot_sdtm(df, "FTTESTCD", "FTSTRESN")
    wide = wide.rename(columns=feature_codes)
    n_motor = len([c for c in wide.columns if c.startswith("motor_")])
    log.info("Loaded motor features", n=len(wide), n_features=n_motor)
    return wide


def build_biohermes_dataset() -> pd.DataFrame:
    """Build the Bio-Hermes-001 master dataset.

    Merges: demographics + education + pTau217 + Roche panel +
            amyloid label + MMSE + acoustic + motor features.
    Filters to participants with amyloid classification available.

    Returns:
        Master DataFrame, one row per participant with amyloid label.
    """
    log.info("Building Bio-Hermes-001 master dataset")

    df_dm       = load_demographics()
    df_edu      = load_education()
    df_ptau     = load_ptau217()
    df_roche    = load_roche_panel()
    df_amy      = load_amyloid_label()
    df_mmse     = load_mmse_baseline()
    df_acoustic = load_acoustic_features()
    df_motor    = load_motor_features()

    # Start from amyloid-confirmed participants
    df = df_amy.merge(df_dm, on="USUBJID", how="left")
    df = df.merge(df_edu, on="USUBJID", how="left")
    df = df.merge(df_ptau, on="USUBJID", how="left")
    df = df.merge(df_roche, on="USUBJID", how="left")
    df = df.merge(df_mmse, on="USUBJID", how="left")
    df = df.merge(df_acoustic, on="USUBJID", how="left")
    df = df.merge(df_motor, on="USUBJID", how="left")

    # No MMSE slope or survival labels for Bio-Hermes-001 (cross-sectional)
    df["MMSE_SLOPE"]      = np.nan
    df["TIME_TO_EVENT"]   = np.nan
    df["EVENT_INDICATOR"] = np.nan

    n_pos = int(df["AMYLOID_POSITIVE"].sum())
    rate  = df["AMYLOID_POSITIVE"].mean()
    log.info("Bio-Hermes-001 master dataset built",
             n_participants=len(df),
             n_amyloid_pos=n_pos,
             amyloid_rate=f"{rate:.1%}")
    return df


def apply_adni_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ADNI-fitted StandardScaler to Bio-Hermes-001 features.

    Loads scaler from data/processed/adni/scaler.pkl.
    DOES NOT refit — uses train-set statistics from ADNI.

    Args:
        df: Bio-Hermes-001 master dataset.

    Returns:
        DataFrame with scaled feature columns.
    """
    scaler_path = ADNI_PROCESSED_DIR / "scaler.pkl"
    if not scaler_path.exists():
        log.warning("ADNI scaler not found — run adni_preprocessing.py first",
                    path=str(scaler_path))
        return df

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Get the full feature list the scaler was fitted on
    if hasattr(scaler, "feature_names_in_"):
        all_scaler_cols = list(scaler.feature_names_in_)
    else:
        all_scaler_cols = []

    # Only scale columns present in both the scaler fit set and this dataframe
    PREFERRED_COLS = [
        "PTAU217", "ABETA4240_RATIO", "NFL_PLASMA", "GFAP_PLASMA",
        "AGE", "MMSE_BASELINE", "EDUCATION_YEARS",
    ]
    scale_cols = [c for c in PREFERRED_COLS if c in df.columns and c in all_scaler_cols]

    if not scale_cols:
        log.warning("No overlapping columns between Bio-Hermes-001 and ADNI scaler — skipping scaling")
        return df

    df = df.copy()
    # Build a full-width array matching the scaler's expected input, filling
    # non-BH columns with zeros (they won't be used; only BH cols are written back)
    import numpy as np
    n = len(df)
    X_full = np.zeros((n, len(all_scaler_cols)), dtype=np.float64)
    col_to_idx = {col: i for i, col in enumerate(all_scaler_cols)}
    for col in scale_cols:
        X_full[:, col_to_idx[col]] = df[col].fillna(0).values

    X_scaled = scaler.transform(X_full)

    for col in scale_cols:
        df[col] = X_scaled[:, col_to_idx[col]]

    log.info("Applied ADNI scaler to Bio-Hermes-001", n_cols_scaled=len(scale_cols))
    return df


def run_biohermes_pipeline() -> None:
    """Main entry point: build, scale, split, and save Bio-Hermes-001 processed data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_biohermes_dataset()
    df = apply_adni_scaler(df)

    # 80/20 train/val split (no test — ADNI test set is held out)
    df_train, df_val = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

    df_train.to_csv(PROCESSED_DIR / "biohermes001_train.csv", index=False)
    df_val.to_csv(PROCESSED_DIR / "biohermes001_val.csv", index=False)

    log.info("Bio-Hermes-001 pipeline complete",
             n_train=len(df_train), n_val=len(df_val))


if __name__ == "__main__":
    run_biohermes_pipeline()

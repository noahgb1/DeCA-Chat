# functions_tabular.py
# ---- Mixed tabular analytics over Excel-like docs (prompt-driven, timeboxed) ----
# Uses parquet manifests produced at ingest time. Prefers DuckDB for big files; falls back to pandas.

from typing import Dict, Any, Optional, List, Tuple
import os
import time
import json
import re
import hashlib  # needed for _checksum

# Patterns that indicate the user is asking for analysis (broad, natural language friendly)
_ANALYTIC_PATTERNS = [
    r"\banaly[sz]e?\b", r"\banalysis\b", r"\binsight(s)?\b", r"\bsummar(y|ize|ise)\b",
    r"\bprofile\b", r"\bstats?\b", r"\bstatistic(s)?\b", r"\bmetric(s)?\b",
    r"\bcompute\b", r"\baggregate\b", r"\bgroup\b",
    r"\bdistribution\b", r"\bcorrelat(e|ion)\b", r"\bcompare\b",
    r"\bmean\b", r"\bmedian\b", r"\baverage\b", r"\bmin(imum)?\b", r"\bmax(imum)?\b",
    r"\bcount(s)?\b", r"\bsum(s|mary)?\b", r"\bbreak ?down\b", r"\boverview\b",
    r"\bdescrib(e|e\s+the)\b", r"\bexplor(e|ation|atory)\b",
    r"\bvariance\b", r"\bvar\b", r"\bstandard deviation\b", r"\bstd\b", r"\bstdev\b"
]

# Narrower hints that specifically justify computing time trends
_TIME_TREND_HINTS = [
    r"\btrend(s)?\b", r"\bover time\b", r"\btime[- ]?series\b", r"\bmonthly\b",
    r"\bseason(al|ality)?\b", r"\bby month\b", r"\bper month\b"
]

def should_run_tabular(question: Optional[str]) -> bool:
    if not question:
        return False
    q = (question or "").lower()
    for pat in _ANALYTIC_PATTERNS:
        if re.search(pat, q):
            return True
    # Also allow very general requests like "tell me about the document"
    if "tell me about" in q or "general analysis" in q or "provide insight" in q:
        return True
    return False

def wants_trend(question: Optional[str]) -> bool:
    if not question:
        return False
    q = (question or "").lower()
    for pat in _TIME_TREND_HINTS:
        if re.search(pat, q):
            return True
    return False

# Containers (prefer config, fallback to env, then defaults)
GROUP_CONTAINER = None
USER_CONTAINER = None
try:
    from config import (
        storage_account_group_documents_container_name as GROUP_CONTAINER,
        storage_account_user_documents_container_name as USER_CONTAINER,
    )
except Exception:
    GROUP_CONTAINER = os.environ.get("GROUP_DOCS_CONTAINER") or os.environ.get("BLOB_CONTAINER_GROUP") or "group-documents"
    USER_CONTAINER  = os.environ.get("USER_DOCS_CONTAINER")  or os.environ.get("BLOB_CONTAINER_USER")  or "user-documents"

def _lazy_imports():
    pd = None; np = None; duckdb = None
    try:
        import pandas as _pd  # type: ignore
        pd = _pd
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore
        np = _np
    except Exception:
        pass
    try:
        import duckdb as _duckdb  # type: ignore
        duckdb = _duckdb
    except Exception:
        pass
    return pd, np, duckdb

# Increased budgets so small (e.g., 14x50) sheets never time out spuriously
BASE_TIME_BUDGET_MS = 1200
BOOSTED_TIME_BUDGET_MS = 2600

ANALYTIC_HINTS = (
    "count","counts","total","sum","average","avg","median","mean","min","max",
    "group","by","top","trend","distribution","percent","percentage","rate",
    "histogram","boxplot","chart","graph","plot","describe","profile","analyze",
    "analysis","insight","eda","explore","exploration","breakdown","overview",
    "summary","summarize","variance","var","standard deviation","std","stdev","time series","monthly"
)

def _scaled_time_budget(prompt: str) -> int:
    if not prompt:
        return BASE_TIME_BUDGET_MS
    p = (prompt or "").lower()
    if any(h in p for h in ANALYTIC_HINTS) or "analy" in p or "insight" in p:
        return BOOSTED_TIME_BUDGET_MS
    return BASE_TIME_BUDGET_MS

# ---------- Blob helpers ----------
def _blob_clients():
    # Prefer connection string; if needed, this can be extended to use configured CLIENTS like other modules
    try:
        from azure.storage.blob import BlobServiceClient
        conn = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                or os.environ.get("AZUREWebJobsStorage") or "")
        if not conn:
            return None
        return BlobServiceClient.from_connection_string(conn)
    except Exception:
        return None

def _get_containers() -> Tuple[Optional[str], Optional[str]]:
    return GROUP_CONTAINER, USER_CONTAINER

def _guess_blob_prefix(user_id: Optional[str], active_group_id: Optional[str]) -> Optional[str]:
    if active_group_id:
        return f"{active_group_id}/"
    if user_id:
        return f"{user_id}/"
    return None

def _try_list_blobs(container_name: Optional[str], prefix: Optional[str]) -> List[str]:
    names: List[str] = []
    if not container_name:
        return names
    bsc = _blob_clients()
    if not bsc:
        return names
    try:
        cont = bsc.get_container_client(container_name)
        for blob in cont.list_blobs(name_starts_with=prefix or ""):
            names.append(blob.name)
    except Exception:
        return []
    return names

def _download_blob_to_temp(container_name: Optional[str], blob_name: str) -> Optional[str]:
    """
    Securely download a blob to a uniquely named temp file.
    Caller MUST remove the returned file when finished.
    """
    if not container_name:
        return None
    bsc = _blob_clients()
    if not bsc:
        return None
    try:
        cont = bsc.get_container_client(container_name)
        blob = cont.get_blob_client(blob_name)
        # preserve extension if present
        _, ext = os.path.splitext(blob_name)
        import tempfile
        # create a unique file; NamedTemporaryFile with delete=False so caller can control lifecycle
        tmp = tempfile.NamedTemporaryFile(prefix="tabular_", suffix=ext or "", delete=False)
        tmp_path = tmp.name
        try:
            data = blob.download_blob()
            tmp.write(data.readall())
            tmp.flush()
        finally:
            tmp.close()
        return tmp_path
    except Exception:
        try:
            # best-effort cleanup if partially created
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        # let caller observe None on failure
        return None

def _find_manifest_blob(user_id: Optional[str], active_group_id: Optional[str], document_id: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Prefer exact manifest by document_id at deterministic path(s:
      - If active_group_id: "<active_group_id>/{document_id}__manifest.json"
      - Then user: "<user_id>/{document_id}__manifest.json"
    Fall back to scanning existing heuristics if not found.
    """
    group_c, user_c = _get_containers()
    bsc = _blob_clients()

    # Try exact candidate paths first when document_id provided
    if document_id and bsc:
        if active_group_id and group_c:
            try:
                candidate = f"{active_group_id}/{document_id}__manifest.json"
                blob_client = bsc.get_container_client(group_c).get_blob_client(candidate)
                if blob_client.exists():
                    return (group_c, candidate)
            except Exception:
                pass
        if user_id and user_c:
            try:
                candidate = f"{user_id}/{document_id}__manifest.json"
                blob_client = bsc.get_container_client(user_c).get_blob_client(candidate)
                if blob_client.exists():
                    return (user_c, candidate)
            except Exception:
                pass

    # Fallback: previous heuristic scan
    prefix = _guess_blob_prefix(user_id, active_group_id)
    if active_group_id and group_c and prefix:
        for name in _try_list_blobs(group_c, prefix):
            if name.lower().endswith("__manifest.json"):
                return (group_c, name)
    if user_id and user_c:
        uprefix = _guess_blob_prefix(user_id, None)
        for name in _try_list_blobs(user_c, uprefix or ""):
            if name.lower().endswith("__manifest.json"):
                return (user_c, name)
    for container_name in filter(None, [group_c, user_c]):
        for name in _try_list_blobs(container_name, None):
            if name.lower().endswith("__manifest.json"):
                return (container_name, name)
    return None

# ---------- Markdown helpers ----------
def _markdown_table_from_records(records, cols, max_rows: int = 12) -> str:
    if not records:
        return ""
    lines = []
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.extend([header, sep])
    for r in records[:max_rows]:
        row = "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |"
        lines.append(row)
    return "\n".join(lines)

# ---------- Pandas small-ops ----------
def _profile_dataframe(pd, df, max_cols: int = 8):
    prof = []; n = len(df)
    for i, col in enumerate(df.columns):
        if i >= max_cols:
            break
        s = df[col]; non_null = int(s.notna().sum())
        non_null_pct = float(non_null) / float(n) * 100.0 if n else 0.0
        dtype = str(s.dtype)
        is_num = pd.api.types.is_numeric_dtype(s)
        avg_len = None; distinct_ratio = None
        try:
            if not is_num:
                ss = s.dropna().astype(str)
                if len(ss) > 0:
                    avg_len = float(ss.str.len().mean())
                    distinct_ratio = float(ss.nunique()) / float(len(ss))
        except Exception:
            pass
        prof.append({
            "column": str(col), "dtype": dtype,
            "non_null_pct": round(non_null_pct, 2),
            "is_numeric": bool(is_num),
            "avg_str_len": None if avg_len is None else round(avg_len, 2),
            "distinct_ratio": None if distinct_ratio is None else round(distinct_ratio, 3)
        })
    return prof

def _pick_columns(pd, df, max_numeric=2, max_categorical=1):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if (not pd.api.types.is_numeric_dtype(df[c])) and df[c].nunique() <= 1000]
    return num_cols[:max_numeric], cat_cols[:max_categorical]

def _safe_trend(pd, s):
    sc = s.dropna()
    if sc.empty:
        return None
    try:
        dt = pd.to_datetime(sc, errors="coerce", infer_datetime_format=True)
        if dt.notna().sum() < max(10, int(0.3 * len(sc))):
            return None
        g = dt.dt.to_period("M").value_counts().sort_index()
        return [{"period": str(ix), "count": int(val)} for ix, val in g.items()]
    except Exception:
        return None

def _basic_aggregates_pandas(pd, df, compute_trend: bool):
    out = {"numeric": [], "categorical_top": [], "trend": []}
    num_cols, cat_cols = _pick_columns(pd, df)
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        try:
            out["numeric"].append({
                "column": c,
                "min": float(s.min()),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "max": float(s.max()),
                "non_null": int(s.size)
            })
        except Exception:
            continue
    for c in cat_cols:
        s = df[c].dropna().astype(str)
        if len(s) == 0:
            continue
        try:
            vc = s.value_counts().head(5)
            for k, v in vc.items():
                out["categorical_top"].append({"column": c, "value": k, "count": int(v)})
        except Exception:
            continue
    if compute_trend:
        for col in df.columns:
            tr = _safe_trend(pd, df[col])
            if tr:
                out["trend"] = tr[:24]
                break
    return out

# ---------- DuckDB big-ops ----------
def _duckdb_schema(duckdb, parquet_path: str):
    try:
        con = duckdb.connect()
        rel = con.sql(f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0")
        cols = [(c, str(t)) for c, t in zip(rel.columns, rel.dtypes)]
        con.close()
        return cols
    except Exception:
        return []

def _duckdb_basic_aggregates(duckdb, parquet_path: str, compute_trend: bool, max_numeric=2, max_categorical=1):
    out = {"numeric": [], "categorical_top": [], "trend": []}
    cols = _duckdb_schema(duckdb, parquet_path)
    if not cols:
        return out
    numeric_types = ("TINYINT","SMALLINT","INTEGER","BIGINT","HUGEINT","UTINYINT","USMALLINT","UINTEGER","UBIGINT","FLOAT","DOUBLE","DECIMAL")
    date_types = ("DATE","TIMESTAMP","TIMESTAMPTZ","TIME")
    num_cols = [c for c,t in cols if any(t.upper().startswith(nt) for nt in numeric_types)][:max_numeric]
    cat_cols = [c for c,t in cols if t.upper().startswith("VARCHAR")][:max_categorical]
    date_cols = [c for c,t in cols if any(t.upper().startswith(dt) for dt in date_types)]
    try:
        con = duckdb.connect()
        for c in num_cols:
            q = f"SELECT min({c}) AS min, avg({c}) AS mean, median({c}) AS median, max({c}) AS max, count({c}) AS non_null FROM read_parquet('{parquet_path}')"
            row = con.sql(q).fetchone()
            if row:
                out["numeric"].append({
                    "column": c,
                    "min": float(row[0]) if row[0] is not None else None,
                    "mean": float(row[1]) if row[1] is not None else None,
                    "median": float(row[2]) if row[2] is not None else None,
                    "max": float(row[3]) if row[3] is not None else None,
                    "non_null": int(row[4]) if row[4] is not None else 0
                })
        for c in cat_cols:
            q = f"SELECT {c} AS value, COUNT(*) AS count FROM read_parquet('{parquet_path}') GROUP BY {c} ORDER BY count DESC NULLS LAST LIMIT 5"
            rows = con.sql(q).fetchall()
            for v, cnt in rows:
                out["categorical_top"].append({"column": c, "value": "" if v is None else str(v), "count": int(cnt)})
        if compute_trend and date_cols:
            c = date_cols[0]
            q = f"""
                SELECT strftime({c}, '%Y-%m-01') AS period, COUNT(*) AS count
                FROM read_parquet('{parquet_path}')
                WHERE {c} IS NOT NULL
                GROUP BY period
                ORDER BY period
                LIMIT 24
            """
            rows = con.sql(q).fetchall()
            out["trend"] = [{"period": str(p), "count": int(cnt)} for p, cnt in rows]
        con.close()
    except Exception:
        pass
    return out

# ---------- NEW: deterministic aggregate entry point ----------
# (expanded with value listing, unique listing, distinct counts; fixed lazy import unpack; added stddev/variance and monthly trends)
_OPS = {
    "mean": ["mean", "average", "avg"],
    "sum": ["sum", "total"],
    "count_rows": ["count rows", "how many rows", "row count", "rows total", "count of rows"],
    "min": ["min", "minimum", "smallest", "lowest"],
    "max": ["max", "maximum", "largest", "highest"],
    "median": ["median"],
    "stddev": ["std", "stdev", "standard deviation", "stddev"],
    "variance": ["variance", "var"],
    "list_values": ["list values", "values", "show values"],
    "unique_values": ["unique values", "distinct values", "unique", "distinct"],
    "count_distinct": ["count distinct", "distinct count", "unique count"],
    # NEW: strict, full, in-order column dump with pagination
    "exact_column": ["exact column", "dump column", "full column", "column contents", "list entire column"]
}

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _detect_op_and_column(prompt: str, columns: List[str]) -> Tuple[Optional[str], Optional[str]]:
    p = prompt.lower()
    # operation
    op_hit = None
    for op, variants in _OPS.items():
        for v in variants:
            if v in p:
                op_hit = op
                break
        if op_hit:
            break
    # column (fuzzy)
    target_col = None
    if columns:
        norm_cols = { _normalize(c): c for c in columns }
        # pick longest token from prompt that matches prefix in any column
        tokens = sorted(set(re.findall(r"[a-zA-Z0-9_]+", prompt)), key=len, reverse=True)
        for t in tokens:
            nt = _normalize(t)
            if not nt or len(nt) < 3:
                continue
            # exact
            if nt in norm_cols:
                target_col = norm_cols[nt]
                break
            # contains
            for ncol, orig in norm_cols.items():
                if nt in ncol or ncol in nt:
                    target_col = orig
                    break
            if target_col:
                break
    return op_hit, target_col

def _coerce_numeric_series(pd, s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    # remove common thousands separators, currency/percent, and accounting parentheses
    x = (s.astype(str)
           .str.replace(",", "", regex=False)
           .str.replace("$", "", regex=False)
           .str.replace("%", "", regex=False)
           .str.replace("(", "-", regex=False)
           .str.replace(")", "", regex=False)
           .str.strip())
    return pd.to_numeric(x, errors="coerce")

def _combine_numeric(series_list):
    import pandas as pd
    vals = []
    for s in series_list:
        vals.append(pd.Series(s, dtype="float64").dropna())
    if not vals:
        return None
    return pd.concat(vals, ignore_index=True)

def _sheet_iter_from_manifest(manifest: dict) -> List[dict]:
    return [s for s in manifest.get("sheets", []) if isinstance(s, dict) and "parquet_blob_path" in s]

def _checksum(parts: List[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]

# NEW: manifest validator used by callers before proceeding
def _validate_manifest(m: dict) -> bool:
    try:
        sheets = m.get("sheets", [])
        if not isinstance(sheets, list) or not sheets:
            return False
        for s in sheets:
            if not isinstance(s, dict):
                return False
            if not s.get("parquet_blob_path"):
                return False
            # optional/nice-to-have fields (do not fail if missing)
            _ = s.get("sheet_name", "")
            _ = s.get("rows", None)
            _ = s.get("cols", None)
        return True
    except Exception:
        return False

def _int(arg, default):
    try:
        return int(arg)
    except Exception:
        return default

def _parse_pagination(prompt: str):
    lim = re.search(r"\blimit\s*=\s*(\d+)", prompt, flags=re.I)
    off = re.search(r"\boffset\s*=\s*(\d+)", prompt, flags=re.I)
    return (_int(lim.group(1), 200) if lim else 200,
            _int(off.group(1), 0) if off else 0)

def analyze_or_aggregate(prompt: str, req_json: dict, chunks: Optional[List[dict]] = None) -> Dict[str, Any]:
    """
    Deterministic aggregate path used by /chat/stream.
    Supports:
      - Basic ops: mean/average, sum, min, max, median, count rows, stddev, variance
      - Grouping: "group by <col>" or "by <col>"
      - Sheet hint: sheet=<name> (exact or fuzzy)
      - Listing ops: list values, unique values, count distinct
      - EXACT column dump op: returns full column values in stable row order with pagination
      - Ordering flags: "ascending"/"descending"/"sorted", or "order by <col> asc|desc"
      - Time-series: if the question suggests trends, compute monthly counts over a detected date column (or date=<col> hint)
    Returns: {"text": str, "audit": {...}, "structured": {...}, "citations": []}
    """
    import statistics as stats
    start = time.time()
    pd, _, duckdb = _lazy_imports()  # FIX: unpack correctly (pd, np, duckdb)
    budget = _scaled_time_budget(prompt or "")
    user_id = req_json.get("user_id") or req_json.get("user") or None
    active_group_id = req_json.get("active_group_id") or req_json.get("group_id") or None

    temp_paths: List[str] = []
    try:
        # Parse sheet=<name>, date=<name>, and group by hints
        sheet_hint = None
        m_sheet = re.search(r"sheet\s*=\s*([\"']?)([^\"'\n\r]+)\1", prompt, flags=re.I)
        if m_sheet:
            sheet_hint = m_sheet.group(2).strip()

        date_hint = None
        m_date = re.search(r"\bdate\s*=\s*([\"']?)([^\"'\n\r]+)\1", prompt, flags=re.I)
        if m_date:
            date_hint = m_date.group(2).strip()

        group_col_hint = None
        m_grp = re.search(r"\bgroup\s+by\s+([A-Za-z0-9 _\-./]+)", prompt, flags=re.I)
        if not m_grp:
            m_grp = re.search(r"\bby\s+([A-Za-z0-9 _\-./]+)", prompt, flags=re.I)
        if m_grp:
            group_col_hint = m_grp.group(1).strip()

        # Ordering hints
        order_hint = None
        m_ord = re.search(r"\b(order(?:ed)?|sort(?:ed)?)\s+by\s+([A-Za-z0-9 _\-./]+)\s*(asc|desc)?", prompt, flags=re.I)
        if m_ord:
            order_hint = (m_ord.group(2).strip(), (m_ord.group(3) or "").lower() or "asc")
        # fallback: "ascending/descending/sorted" referencing the measure column
        order_dir = None
        if re.search(r"\bdesc(end(?:ing)?)?\b", prompt, flags=re.I):
            order_dir = "desc"
        elif re.search(r"\basc(end(?:ing)?)?\b", prompt, flags=re.I) or re.search(r"\bsort(?:ed)?\b", prompt, flags=re.I):
            order_dir = "asc"

        # Fetch manifest (prefer explicit selected_document_id when present)
        sel_doc = req_json.get("selected_document_id") or req_json.get("document_id") or None
        found = _find_manifest_blob(user_id, active_group_id, sel_doc)
        if not found:
            return {}
        container_for_data, manifest_blob = found
        local_manifest = _download_blob_to_temp(container_for_data, manifest_blob)
        if local_manifest:
            temp_paths.append(local_manifest)
        if not local_manifest or not os.path.exists(local_manifest):
            return {}
        try:
            with open(local_manifest, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            return {}
        # Validate manifest structure before proceeding
        if not _validate_manifest(manifest):
            return {
                "text": (
                    "Tabular manifest found but it does not include valid sheet/parquet entries. "
                    "Ensure ingestion completed successfully and the manifest contains a 'sheets' list with 'parquet_blob_path' for each sheet."
                ),
                "audit": {}, "structured": {}, "citations": []
            }

        sheets = _sheet_iter_from_manifest(manifest)
        if not sheets:
            return {}

        # Filter sheets by sheet_hint (fuzzy)
        if sheet_hint:
            target = _normalize(sheet_hint)
            filtered = [s for s in sheets if _normalize(s.get("sheet_name","")) == target or target in _normalize(s.get("sheet_name",""))]
            if filtered:
                sheets = filtered

        # Column discovery from first matching sheet
        columns = []
        for s in sheets:
            if (time.time() - start) * 1000.0 > budget:
                break
            local_parquet = _download_blob_to_temp(container_for_data, s["parquet_blob_path"])
            if local_parquet:
                temp_paths.append(local_parquet)
            if not local_parquet or not os.path.exists(local_parquet):
                continue
            try:
                if duckdb is not None:
                    import duckdb as _ddb
                    con = _ddb.connect()
                    rel = con.sql(f"SELECT * FROM read_parquet('{local_parquet}') LIMIT 0")
                    columns = [str(c) for c in rel.columns]
                    con.close()
                    if columns: break
            except Exception:
                pass
            try:
                if pd is not None:
                    df = pd.read_parquet(local_parquet)
                    columns = [str(c) for c in df.columns]
                    break
            except Exception:
                continue

        # Detect operation and measure column
        op, measure_col = _detect_op_and_column(prompt or "", columns)

        # ---------- NEW: Trends (monthly) branch ----------
        if wants_trend(prompt) or (op is None and date_hint):
            if pd is None:
                return {"text": "Time-series requires pandas.", "audit": {}, "structured": {}, "citations": []}
            # Find a candidate date column
            candidate_date = None
            if date_hint:
                # resolve fuzzily against discovered columns
                nt = _normalize(date_hint)
                for c in columns:
                    if _normalize(c) == nt or nt in _normalize(c) or _normalize(c) in nt:
                        candidate_date = c
                        break

            # Scan sheets to confirm/auto-detect if needed
            monthly_counts: Dict[str, int] = {}
            checked = False
            for s in sheets:
                if (time.time() - start) * 1000.0 > budget:
                    break
                lp = _download_blob_to_temp(container_for_data, s["parquet_blob_path"])
                if lp:
                    temp_paths.append(lp)
                if not lp or not os.path.exists(lp):
                    continue
                try:
                    df = pd.read_parquet(lp)
                except Exception:
                    continue

                # pick/verify candidate_date per sheet
                date_col = candidate_date
                if date_col is None:
                    # auto-detect on this sheet
                    best_col = None; best_ratio = 0.0
                    for c in df.columns:
                        s_col = df[c].dropna()
                        if s_col.empty:
                            continue
                        sample = s_col.head(500)
                        dt = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                        valid = int(dt.notna().sum())
                        ratio = float(valid) / float(len(sample)) if len(sample) else 0.0
                        if valid >= 10 and ratio > best_ratio and ratio >= 0.3:
                            best_ratio = ratio; best_col = c
                    date_col = best_col

                if date_col and date_col in df.columns:
                    dt_all = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
                    periods = dt_all.dropna().dt.to_period("M").astype(str)
                    vc = periods.value_counts()
                    for period, cnt in vc.items():
                        monthly_counts[period] = monthly_counts.get(period, 0) + int(cnt)
                    checked = True

            if checked and monthly_counts:
                # sort by period ascending
                series = [{"period": k, "count": monthly_counts[k]} for k in sorted(monthly_counts.keys())][:24]
                head_lines = [f"Monthly counts{(f' using {candidate_date}' if candidate_date else '')}{(f' on sheet {sheet_hint}' if sheet_hint else '')}:"]
                for r in series[:15]:
                    head_lines.append(f"- {r['period']}: {r['count']}")
                if len(series) > 15:
                    head_lines.append(f"- … {len(series)-15} more months")
                audit = {
                    "rows_used": None,
                    "columns_used": [candidate_date] if candidate_date else [],
                    "sheets_used": [s.get("sheet_name","?") for s in sheets],
                    "parquet_count": len(sheets),
                    "operation": "monthly_trend",
                    "operands_count": None,
                    "filters": {"sheet": sheet_hint} if sheet_hint else None,
                    "group_by": None,
                    "groups_count": 0,
                    "checksum": _checksum([s.get("parquet_blob_path","") for s in sheets] + ([candidate_date] if candidate_date else []))
                }
                return {
                    "text": "\n".join(head_lines),
                    "audit": audit,
                    "structured": {"trend_monthly": series, "date_column": candidate_date},
                    "citations": []
                }
            # If we couldn't detect any, fall through to normal guidance
            # and let user specify exact date column
            if not candidate_date:
                return {
                    "text": "I couldn't detect a date column to compute a monthly trend. Specify one with date=ColumnName.",
                    "audit": {}, "structured": {}, "citations": []
                }

        # ---------- NEW: Strict EXACT column dump with stable order + pagination ----------
        if op == "exact_column":
            if not measure_col:
                # auto-pick if user wrote "exact column" without a name
                if columns:
                    measure_col = columns[0]
                else:
                    return {"text": "No column found.", "audit": {}, "structured": {}, "citations": []}

            limit, offset = _parse_pagination(prompt or "")
            total = 0
            values: List[str] = []

            # explicit sort: "order by <col> asc|desc"
            ord_col, ord_dir_local = None, None
            m_ord2 = re.search(r"\border\s+by\s+([A-Za-z0-9 _./-]+)\s*(asc|desc)?", prompt or "", flags=re.I)
            if m_ord2:
                ord_col = m_ord2.group(1).strip()
                ord_dir_local = (m_ord2.group(2) or "asc").lower()

            for s in sheets:
                if (time.time() - start) * 1000.0 > budget:
                    break
                local_parquet = _download_blob_to_temp(container_for_data, s["parquet_blob_path"])
                if local_parquet:
                    temp_paths.append(local_parquet)
                if not local_parquet or not os.path.exists(local_parquet):
                    continue

                if duckdb is not None:
                    import duckdb as _ddb
                    con = _ddb.connect()
                    try:
                        # columns for fuzzy resolution of ord_col
                        cols = [str(c) for c in con.sql(f"SELECT * FROM read_parquet('{local_parquet}') LIMIT 0").columns]
                        order_clause = ""
                        resolved_order_by = None
                        if ord_col:
                            resolved_order_by = ord_col
                            if resolved_order_by not in cols:
                                no = _normalize(resolved_order_by)
                                for c in cols:
                                    if _normalize(c) == no or no in _normalize(c) or _normalize(c) in no:
                                        resolved_order_by = c; break
                            if resolved_order_by in cols:
                                order_clause = f' ORDER BY "{resolved_order_by}" {"DESC" if (ord_dir_local=="desc") else "ASC"}'
                        else:
                            # prefer stable original order if row_index exists
                            if "row_index" in cols:
                                order_clause = ' ORDER BY row_index'

                        q = f'''
                            SELECT "{measure_col}" AS v
                            FROM read_parquet('{local_parquet}')
                            {order_clause}
                            LIMIT {limit} OFFSET {offset}
                        '''
                        page_vals = [r[0] for r in con.sql(q).fetchall()]
                        tq = f"SELECT COUNT(*) FROM read_parquet('{local_parquet}')"
                        total += int(con.sql(tq).fetchone()[0] or 0)
                        values.extend([("" if v is None else str(v)) for v in page_vals])
                    finally:
                        con.close()
                else:
                    # pandas fallback
                    if pd is None:
                        continue
                    try:
                        df = pd.read_parquet(local_parquet)
                        total += int(df.shape[0])
                        # stable order preference
                        if ord_col and ord_col in df.columns:
                            asc = (ord_dir_local != "desc")
                            df = df.sort_values(ord_col, ascending=asc, kind="mergesort")
                        elif "row_index" in df.columns:
                            df = df.sort_values("row_index", kind="mergesort")
                        sub = df.iloc[offset: offset+limit]
                        if measure_col not in sub.columns:
                            # fuzzy resolve
                            nt = _normalize(measure_col)
                            for c in df.columns:
                                if _normalize(str(c)) == nt or nt in _normalize(str(c)) or _normalize(str(c)) in nt:
                                    measure_col = str(c); break
                            sub = df.iloc[offset: offset+limit]
                        vals = sub[measure_col].astype(str).tolist()
                        values.extend(vals)
                    except Exception:
                        continue

            # Build audit for exact op
            parquet_ids = [s.get("parquet_blob_path","") for s in sheets]
            audit = {
                "rows_used": int(total),
                "columns_used": [measure_col],
                "sheets_used": [s.get("sheet_name","?") for s in sheets],
                "parquet_count": len(parquet_ids),
                "operation": op,
                "operands_count": int(len(values)),
                "filters": {"sheet": sheet_hint} if sheet_hint else None,
                "group_by": None,
                "groups_count": 0,
                "checksum": _checksum(parquet_ids + [measure_col])
            }

            result = {
                "values": values,
                "total": total,
                "page_size": limit,
                "offset": offset,
                "column": measure_col,
                "order_applied": {
                    "by": (ord_col or ("row_index" if ord_col is None else None)) or "insertion",
                    "dir": (ord_dir_local or "asc")
                }
            }
            text = f"Exact values for **{measure_col}** (first {len(values)} of {total})."
            # Optional integrity note if obvious truncation by limit/offset
            if len(values) < total and offset == 0:
                text += " (Paginated; increase limit= to see more.)"
            return {"text": text, "audit": audit, "structured": result, "citations": []}
        # ---------- END exact column branch ----------

        # If the user asked for a column-based op but we couldn't match, fail closed
        if op not in ("count_rows", "list_values", "unique_values", "count_distinct", None) and not measure_col:
            return {
                "text": ("I need the exact column name to compute this (e.g., 'average Distance'). "
                         "You can also specify a sheet like: sheet=Sheet1."),
                "audit": {"rows_used": 0, "columns_used": [], "operation": None, "checksum": None},
                "structured": {},
                "citations": []
            }

        # Resolve group-by column if provided
        group_col = None
        if group_col_hint and columns:
            norm_cols = { _normalize(c): c for c in columns }
            ng = _normalize(group_col_hint)
            if ng in norm_cols:
                group_col = norm_cols[ng]
            else:
                for ncol, orig in norm_cols.items():
                    if ng in ncol or ncol in ng:
                        group_col = orig
                        break

        # Aggregate across sheets
        total_rows = 0
        operands_used = 0
        sheet_names = []
        parquet_ids = []

        # For ungrouped: collect measure values across all sheets
        series_pool = []

        # For grouped: map[group_value] -> list of numeric values
        grouped_values = {}

        # For listing ops: collect values (possibly deduplicate later)
        listing_values: List[Any] = []

        for s in sheets:
            if (time.time() - start) * 1000.0 > budget:
                break
            sheet_name = s.get("sheet_name", "?")
            local_parquet = _download_blob_to_temp(container_for_data, s["parquet_blob_path"])
            if local_parquet:
                temp_paths.append(local_parquet)
            if not local_parquet or not os.path.exists(local_parquet):
                continue

            parquet_ids.append(s["parquet_blob_path"])
            sheet_names.append(sheet_name)

            try:
                if pd is None:
                    continue
                df = pd.read_parquet(local_parquet)
            except Exception:
                continue

            total_rows += int(df.shape[0])

            # Early exit accumulation for count_rows
            if op == "count_rows":
                continue

            # For list/unique/count_distinct we only need the raw column values
            if op in ("list_values", "unique_values", "count_distinct"):
                # Fuzzy-adjust column per sheet if needed
                mcol_list = measure_col
                if (not mcol_list) and columns:
                    # If user didn't specify a column but asked to "list values", prefer the first non-numeric column
                    non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
                    if non_num:
                        mcol_list = str(non_num[0])
                        measure_col = measure_col or mcol_list  # reflect auto-select in audit
                if mcol_list and mcol_list not in df.columns:
                    nt = _normalize(mcol_list)
                    for c in df.columns:
                        if _normalize(str(c)) == nt or nt in _normalize(str(c)) or _normalize(str(c)) in nt:
                            mcol_list = str(c)
                            break
                if mcol_list and mcol_list in df.columns:
                    colser = df[mcol_list]
                    # If ordering is asked and refers to another column, prepare sortable key
                    if order_hint:
                        ord_col, ord_dir2 = order_hint
                        # try resolve order column fuzzily
                        ocol = ord_col
                        if ocol not in df.columns:
                            no = _normalize(ocol)
                            for c in df.columns:
                                if _normalize(str(c)) == no or no in _normalize(c) or _normalize(c) in no:
                                    ocol = str(c)
                                    break
                        if ocol in df.columns:
                            tmp = df[[mcol_list, ocol]].copy()
                            # numeric-friendly sort
                            tmp["_ord"] = _coerce_numeric_series(pd, tmp[ocol]) if pd.api.types.is_numeric_dtype(tmp[ocol]) else tmp[ocol].astype(str)
                            tmp = tmp.sort_values("_ord", ascending=(ord_dir2 != "desc"), kind="mergesort")
                            listing_values.extend(tmp[mcol_list].astype(str).tolist())
                        else:
                            listing_values.extend(colser.astype(str).tolist())
                    elif order_dir and measure_col:
                        # sort by the measure column itself if user said "ascending/descending"
                        tmp = df[[mcol_list]].copy()
                        tmp["_ord"] = _coerce_numeric_series(pd, tmp[mcol_list]) if pd.api.types.is_numeric_dtype(tmp[mcol_list]) else tmp[mcol_list].astype(str)
                        tmp = tmp.sort_values("_ord", ascending=(order_dir != "desc"), kind="mergesort")
                        listing_values.extend(tmp[mcol_list].astype(str).tolist())
                    else:
                        listing_values.extend(colser.astype(str).tolist())
                continue  # listing ops don't fall through

            # Fuzzy-adjust measure and group columns per sheet if needed
            mcol = measure_col
            if mcol and mcol not in df.columns:
                nt = _normalize(mcol)
                for c in df.columns:
                    if _normalize(str(c)) == nt or nt in _normalize(str(c)) or _normalize(str(c)) in nt:
                        mcol = str(c)
                        break

            gcol = group_col
            if group_col and group_col not in df.columns:
                ng = _normalize(gcol)
                for c in df.columns:
                    if _normalize(str(c)) == ng or ng in _normalize(str(c)) or _normalize(str(c)) in ng:
                        gcol = str(c)
                        break

            if mcol and mcol in df.columns:
                sdata = _coerce_numeric_series(pd, df[mcol]).dropna()
                operands_used += int(sdata.size)
                if gcol and gcol in df.columns:
                    keys = df[gcol].astype(str).fillna("")
                    for key, val in zip(keys, sdata):
                        grouped_values.setdefault(key, []).append(float(val))
                else:
                    series_pool.append(sdata.values.tolist())

        # Build audit
        audit = {
            "rows_used": int(total_rows),
            "columns_used": [] if not measure_col else [measure_col] + ([group_col] if group_col else []),
            "sheets_used": sheet_names,
            "parquet_count": len(parquet_ids),
            "operation": op,
            "operands_count": int(operands_used),
            "filters": {"sheet": sheet_hint} if sheet_hint else None,
            "group_by": group_col,
            "groups_count": len(grouped_values) if grouped_values else 0,
            "checksum": _checksum(parquet_ids + ([measure_col] if measure_col else []) + ([group_col] if group_col else []))
        }

        # Handle count_rows
        if op == "count_rows":
            text = f"Row count across selected sheet(s){' (filtered by sheet)' if sheet_hint else ''}: **{total_rows}**."
            return {"text": text, "audit": audit, "structured": {"count_rows": total_rows}, "citations": []}

        # Handle listing/unique/distinct-count
        if op in ("list_values", "unique_values", "count_distinct"):
            colname = measure_col or "(auto-selected)"
            if op == "count_distinct":
                distinct_n = len(set(listing_values))
                return {
                    "text": f"Distinct count for **{colname}**{f' on sheet {sheet_hint}' if sheet_hint else ''}: **{distinct_n}**.",
                    "audit": audit, "structured": {"count_distinct": distinct_n, "column": colname}, "citations": []
                }
            if op == "unique_values":
                uniq = sorted(set(listing_values))
                head = uniq[:50]
                more = len(uniq) - len(head)
                text_lines = [f"Unique values in **{colname}**{f' on sheet {sheet_hint}' if sheet_hint else ''} (showing up to 50):"]
                for v in head:
                    text_lines.append(f"- {v}")
                if more > 0:
                    text_lines.append(f"- … {more} more")
                return {"text": "\n".join(text_lines), "audit": audit, "structured": {"unique_values": head, "column": colname}, "citations": []}
            # list_values (raw, possibly ordered)
            head = listing_values[:200]
            more = len(listing_values) - len(head)
            text_lines = [f"Values from **{colname}**{f' on sheet {sheet_hint}' if sheet_hint else ''} (first {len(head)}):"]
            for v in head:
                text_lines.append(f"- {v}")
            if more > 0:
                text_lines.append(f"- … {more} more")
            return {"text": "\n".join(text_lines), "audit": audit, "structured": {"values_sample": head, "column": colname, "total": len(listing_values)}, "citations": []}

        # If no measure column provided
        if not measure_col:
            if should_run_tabular(prompt):
                guide = "Tell me which column to use (e.g., 'average Distance')."
                if columns:
                    guide += f" Columns include: {', '.join(columns[:8])}" + ("…" if len(columns) > 8 else "")
                guide += " You can also add sheet=SheetName and 'group by <column>'."
                return {"text": guide, "audit": audit, "structured": {}, "citations": []}
            return {}

        # Grouped aggregate
        if grouped_values:
            results = []
            for key, vals in grouped_values.items():
                if not vals:
                    continue
                if op in (None, "mean"):
                    agg = float(sum(vals) / len(vals))
                    results.append({"group": key, "n": len(vals), "mean": agg})
                elif op == "sum":
                    agg = float(sum(vals))
                    results.append({"group": key, "n": len(vals), "sum": agg})
                elif op == "min":
                    agg = float(min(vals))
                    results.append({"group": key, "n": len(vals), "min": agg})
                elif op == "max":
                    agg = float(max(vals))
                    results.append({"group": key, "n": len(vals), "max": agg})
                elif op == "median":
                    agg = float(stats.median(vals))
                    results.append({"group": key, "n": len(vals), "median": agg})
                elif op == "stddev":
                    agg = float(stats.stdev(vals)) if len(vals) >= 2 else 0.0
                    results.append({"group": key, "n": len(vals), "stddev": agg})
                elif op == "variance":
                    agg = float(stats.variance(vals)) if len(vals) >= 2 else 0.0
                    results.append({"group": key, "n": len(vals), "variance": agg})
                else:
                    agg = float(sum(vals) / len(vals))
                    results.append({"group": key, "n": len(vals), "mean": agg})

            # Sort by the chosen metric, descending
            metric_key = "mean" if op in (None, "mean") else op
            results.sort(key=lambda r: r.get(metric_key, 0.0), reverse=True)

            op_name = op if op else "mean"
            head = f"{op_name.capitalize()} of **{measure_col}** by **{group_col}**"
            if sheet_hint:
                head += f" on sheet '{sheet_hint}'"
            lines = [head + ":"]
            for r in results[:15]:
                metric = next((k for k in ("mean","sum","min","max","median","stddev","variance") if k in r), "mean")
                val_fmt = f"{r[metric]:,.2f}" if isinstance(r[metric], (int, float)) else str(r[metric])
                lines.append(f"- {r['group']}: {val_fmt} (n={r['n']})")
            if len(results) > 15:
                lines.append(f"- … {len(results)-15} more groups")

            return {
                "text": "\n".join(lines),
                "audit": audit,
                "structured": {"group_aggregate": results, "measure": measure_col, "group_by": group_col, "operation": op or "mean"},
                "citations": []
            }

        # Ungrouped aggregate
        all_vals = []
        for lst in series_pool:
            all_vals.extend(lst)
        if not all_vals:
            return {
                "text": f"The column '{measure_col}' exists but contains no numeric values I can aggregate.",
                "audit": audit, "structured": {}, "citations": []
            }

        if op in (None, "mean"):
            val = float(sum(all_vals) / len(all_vals))
            return {
                "text": f"Average of **{measure_col}**{f' on sheet {sheet_hint}' if sheet_hint else ''}: **{val:,.2f}** (n={len(all_vals)}).",
                "audit": audit, "structured": {"mean": val, "n": len(all_vals)}, "citations": []
            }
        elif op == "sum":
            val = float(sum(all_vals))
            return {
                "text": f"Sum of **{measure_col}**{f' on sheet {sheet_hint}' if sheet_hint else ''}: **{val:,.2f}** (n={len(all_vals)}).",
                "audit": audit, "structured": {"sum": val, "n": len(all_vals)}, "citations": []
            }
        elif op == "min":
            val = float(min(all_vals))
            return {"text": f"Minimum of **{measure_col}**: **{val:,.2f}**.", "audit": audit, "structured": {"min": val}, "citations": []}
        elif op == "max":
            val = float(max(all_vals))
            return {"text": f"Maximum of **{measure_col}**: **{val:,.2f}**.", "audit": audit, "structured": {"max": val}, "citations": []}
        elif op == "median":
            val = float(stats.median(all_vals))
            return {"text": f"Median of **{measure_col}**: **{val:,.2f}**.", "audit": audit, "structured": {"median": val}, "citations": []}
        elif op == "stddev":
            val = float(stats.stdev(all_vals)) if len(all_vals) >= 2 else 0.0
            return {"text": f"Standard deviation of **{measure_col}**: **{val:,.2f}**.", "audit": audit, "structured": {"stddev": val, "n": len(all_vals)}, "citations": []}
        elif op == "variance":
            val = float(stats.variance(all_vals)) if len(all_vals) >= 2 else 0.0
            return {"text": f"Variance of **{measure_col}**: **{val:,.2f}**.", "audit": audit, "structured": {"variance": val, "n": len(all_vals)}, "citations": []}
        else:
            val = float(sum(all_vals) / len(all_vals))
            return {
                "text": f"Average of **{measure_col}**: **{val:,.2f}** (n={len(all_vals)}).",
                "audit": audit, "structured": {"mean": val, "n": len(all_vals)}, "citations": []
            }
    finally:
        # cleanup downloaded temp files (ensure no orphan temp files remain)
        if temp_paths:
            try:
                from functions_logging import log_exception
            except Exception:
                log_exception = None
            for p in temp_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    try:
                        if log_exception:
                            log_exception(f"Failed to remove temp file {p}: {e}")
                    except Exception:
                        pass

# ---------- Main entry ----------
def tabular_plan_and_execute(
    user_id: Optional[str] = None,
    question: Optional[str] = None,
    selected_document_id: Optional[str] = None,
    document_scope: Optional[str] = None,
    active_group_id: Optional[str] = None,
    time_budget_ms: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Lightweight, robust implementation: locate manifest (prefers selected_document_id),
    validate it, and return a concise sheet-level summary. Always cleans up temp files.
    This avoids syntax/import/runtime errors while preserving the manifest-based flow.
    """
    temp_paths: List[str] = []
    try:
        sel_doc = selected_document_id or kwargs.get("selected_document_id")
        found = _find_manifest_blob(user_id, active_group_id, sel_doc)
        if not found:
            return {}
        container_for_data, manifest_blob = found

        local_manifest = _download_blob_to_temp(container_for_data, manifest_blob)
        if local_manifest:
            temp_paths.append(local_manifest)
        if not local_manifest or not os.path.exists(local_manifest):
            return {}

        try:
            with open(local_manifest, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            return {}

        # validate manifest structure
        if not _validate_manifest(manifest):
            return {
                "system_message": "Tabular manifest is present but does not contain valid sheet/parquet entries.",
                "citations": [], "metrics": {}, "tables": []
            }

        sheets = manifest.get("sheets", []) or []
        # Build a small summary for the caller/UI
        tables: List[Dict[str, Any]] = []
        for s in sheets[:10]:
            tables.append({
                "sheet_name": s.get("sheet_name", ""),
                "rows": s.get("rows"),
                "cols": s.get("cols"),
                "parquet_blob_path": s.get("parquet_blob_path", "")
            })

        return {
            "system_message": f"Found {len(sheets)} sheet(s) in manifest; returning top {len(tables)} summaries.",
            "tables": tables,
            "metrics": {"sheets_found": len(sheets)},
            "citations": []
        }
    finally:
        # best-effort cleanup of downloaded temp files
        if temp_paths:
            try:
                from functions_logging import log_exception
            except Exception:
                log_exception = None
            for p in temp_paths:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception as e:
                    try:
                        if log_exception:
                            log_exception(f"Failed to remove temp file {p}: {e}")
                    except Exception:
                        pass
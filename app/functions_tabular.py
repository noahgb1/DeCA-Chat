
# functions_tabular.py
# ---- Mixed tabular analytics over Excel-like docs (prompt-driven, timeboxed) ----
# Uses parquet manifests produced at ingest time. Prefers DuckDB for big files; falls back to pandas.

from typing import Dict, Any, Optional, List, Tuple
import os
import time
import json
import tempfile
import re
import math
import hashlib

# Patterns that indicate the user is asking for analysis (broad, natural language friendly)
_ANALYTIC_PATTERNS = [
    r"\banaly[sz]e?\b", r"\banalysis\b", r"\binsight(s)?\b", r"\bsummar(y|ize|ise)\b",
    r"\bprofile\b", r"\bstats?\b", r"\bstatistic(s)?\b", r"\bmetric(s)?\b",
    r"\bcompute\b", r"\baggregate\b", r"\bgroup\b",
    r"\bdistribution\b", r"\bcorrelat(e|ion)\b", r"\bcompare\b",
    r"\bmean\b", r"\bmedian\b", r"\baverage\b", r"\bmin(imum)?\b", r"\bmax(imum)?\b",
    r"\bcount(s)?\b", r"\bsum(s|mary)?\b", r"\bbreak ?down\b", r"\boverview\b",
    r"\bdescrib(e|e\s+the)\b", r"\bexplor(e|ation|atory)\b"
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

BASE_TIME_BUDGET_MS = 600
BOOSTED_TIME_BUDGET_MS = 1200

ANALYTIC_HINTS = (
    "count","counts","total","sum","average","avg","median","mean","min","max",
    "group","by","top","trend","distribution","percent","percentage","rate",
    "histogram","boxplot","chart","graph","plot","describe","profile","analyze",
    "analysis","insight","eda","explore","exploration","breakdown","overview",
    "summary","summarize"
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
    if not container_name:
        return None
    bsc = _blob_clients()
    if not bsc:
        return None
    try:
        cont = bsc.get_container_client(container_name)
        blob = cont.get_blob_client(blob_name)
        temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(blob_name))
        with open(temp_path, "wb") as f:
            data = blob.download_blob()
            f.write(data.readall())
        return temp_path
    except Exception:
        return None

def _find_manifest_blob(user_id: Optional[str], active_group_id: Optional[str]) -> Optional[Tuple[str, str]]:
    group_c, user_c = _get_containers()
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
# ####noah changed this####
_OPS = {
    "mean": ["mean", "average", "avg"],
    "sum": ["sum", "total"],
    "count_rows": ["count rows", "how many rows", "row count", "rows total", "count of rows"],
    "min": ["min", "minimum", "smallest", "lowest"],
    "max": ["max", "maximum", "largest", "highest"],
    "median": ["median"]
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
    # remove common thousands separators, spaces
    x = s.astype(str).str.replace(",", "", regex=False).str.strip()
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

def analyze_or_aggregate(prompt: str, req_json: dict, chunks: Optional[List[dict]] = None) -> Dict[str, Any]:
    """
    Deterministic aggregate path used by /chat/stream.
    Returns: {"text": str, "audit": {...}, "structured": {...}, "citations": []}
    """
    start = time.time()
    pd, np, duckdb = _lazy_imports()
    budget = _scaled_time_budget(prompt or "")
    user_id = req_json.get("user_id") or req_json.get("user") or None
    active_group_id = req_json.get("active_group_id") or req_json.get("group_id") or None

    # Fetch manifest
    found = _find_manifest_blob(user_id, active_group_id)
    if not found:
        return {}
    container_for_data, manifest_blob = found
    local_manifest = _download_blob_to_temp(container_for_data, manifest_blob)
    if not local_manifest or not os.path.exists(local_manifest):
        return {}
    try:
        with open(local_manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return {}

    sheets = _sheet_iter_from_manifest(manifest)
    if not sheets:
        return {}

    # Column discovery (first sheet to get names; we will aggregate across all matching sheets)
    columns = []
    sample_paths = []
    for s in sheets:
        if (time.time() - start) * 1000.0 > budget:
            break
        local_parquet = _download_blob_to_temp(container_for_data, s["parquet_blob_path"])
        if not local_parquet or not os.path.exists(local_parquet):
            continue
        sample_paths.append(local_parquet)
        try:
            if duckdb is not None:
                import duckdb as _ddb
                con = _ddb.connect()
                rel = con.sql(f"SELECT * FROM read_parquet('{local_parquet}') LIMIT 0")
                columns = [str(c) for c in rel.columns]
                con.close()
                if columns:
                    break
        except Exception:
            pass
        try:
            if pd is not None:
                df = pd.read_parquet(local_parquet)
                columns = [str(c) for c in df.columns]
                break
        except Exception:
            continue

    op, col = _detect_op_and_column(prompt or "", columns)

    # If the user asked for a column-based op but we couldn't match, fail closed
    if op not in ("count_rows", None) and not col:
        return {
            "text": ("I need the exact column name to compute this (e.g., 'average Distance'). "
                     "Please restate with the column, or specify the sheet and column."),
            "audit": {"rows_used": 0, "columns_used": [], "operation": None, "checksum": None},
            "structured": {},
            "citations": []
        }

    # Aggregate across all sheets
    total_rows = 0
    operands_used = 0
    series_pool = []
    sheet_names = []
    parquet_ids = []

    for s in sheets:
        if (time.time() - start) * 1000.0 > budget:
            break
        sheet_name = s.get("sheet_name", "?")
        local_parquet = _download_blob_to_temp(container_for_data, s["parquet_blob_path"])
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

        if op == "count_rows":
            continue

        # Match column within this sheet too (case-insensitive/fuzzy)
        if col and col not in df.columns:
            # attempt fuzzy within this df
            norm_target = _normalize(col)
            for c in df.columns:
                if _normalize(str(c)) == norm_target or norm_target in _normalize(str(c)) or _normalize(str(c)) in norm_target:
                    col = str(c)
                    break

        if col and col in df.columns:
            sdata = _coerce_numeric_series(pd, df[col])
            series_pool.append(sdata.dropna().values.tolist())
            operands_used += int(sdata.notna().sum())

    # Now compute
    audit = {
        "rows_used": int(total_rows),
        "columns_used": [] if not col else [col],
        "sheets_used": sheet_names,
        "parquet_count": len(parquet_ids),
        "operation": op,
        "operands_count": int(operands_used),
        "filters": None,
        "checksum": _checksum(parquet_ids + ([col] if col else []))
    }

    if op == "count_rows":
        text = f"Row count across selected sheet(s): **{total_rows}**."
        return {"text": text, "audit": audit, "structured": {"count_rows": total_rows}, "citations": []}

    if not col:
        # If no op detected but user clearly wants numbers, guide
        if should_run_tabular(prompt):
            return {
                "text": "Tell me which column to use (e.g., 'average Distance') or say 'count rows'.",
                "audit": audit, "structured": {}, "citations": []
            }
        return {}

    # Flatten values
    all_vals = []
    for lst in series_pool:
        all_vals.extend(lst)
    if not all_vals:
        return {
            "text": f"The column '{col}' exists but contains no numeric values I can aggregate.",
            "audit": audit, "structured": {}, "citations": []
        }

    # Compute stats deterministically
    import statistics as stats
    result = {}
    if op in (None, "mean"):  # default to mean if user said "average/mean"
        mean_val = float(sum(all_vals) / len(all_vals))
        result["mean"] = mean_val
        text = f"Average of **{col}** across sheet(s): **{mean_val:,.2f}** (n={len(all_vals)})."
    elif op == "sum":
        ssum = float(sum(all_vals))
        result["sum"] = ssum
        text = f"Sum of **{col}** across sheet(s): **{ssum:,.2f}** (n={len(all_vals)})."
    elif op == "min":
        v = float(min(all_vals))
        result["min"] = v
        text = f"Minimum of **{col}**: **{v:,.2f}**."
    elif op == "max":
        v = float(max(all_vals))
        result["max"] = v
        text = f"Maximum of **{col}**: **{v:,.2f}**."
    elif op == "median":
        v = float(stats.median(all_vals))
        result["median"] = v
        text = f"Median of **{col}**: **{v:,.2f}**."
    else:
        # Fallback to mean if operation was unclear but numeric intent was detected
        mean_val = float(sum(all_vals) / len(all_vals))
        result["mean"] = mean_val
        text = f"Average of **{col}** across sheet(s): **{mean_val:,.2f}** (n={len(all_vals)})."

    return {"text": text, "audit": audit, "structured": result, "citations": []}
# ######

# ---------- Main entry ----------
def tabular_plan_and_execute(
    user_id: str,
    question: str,
    selected_document_id: Optional[str],
    document_scope: Optional[str],
    active_group_id: Optional[str],
    time_budget_ms: Optional[int] = None
) -> Dict[str, Any]:
    if not should_run_tabular(question):
        return {}
    start = time.time()
    budget = time_budget_ms if time_budget_ms is not None else _scaled_time_budget(question or "")
    compute_trend = wants_trend(question)

    pd, np, duckdb = _lazy_imports()
    if pd is None:
        if (time.time() - start) * 1000.0 > budget:
            return {}
        return {
            "system_message": (
                "When the user requests data analysis, provide a concise, structured answer. "
                "Report concrete counts/averages only when derivable from provided context; "
                "add a compact table (≤5 rows) if helpful; clearly separate facts from interpretations; "
                "cite specific sources for quoted text; keep explanations concise."
            ),
            "citations": [], "metrics": {}, "tables": []
        }

    found = _find_manifest_blob(user_id, active_group_id)
    manifest = None; container_for_data = None
    if found:
        container_for_data, manifest_blob = found
        local_manifest = _download_blob_to_temp(container_for_data, manifest_blob)
        if local_manifest and os.path.exists(local_manifest):
            try:
                with open(local_manifest, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = None

    if manifest is None:
        if (time.time() - start) * 1000.0 > budget:
            return {}
        return {
            "system_message": (
                "When the user requests data analysis, provide a concise, structured answer. "
                "Report concrete counts/averages only when derivable from provided context; "
                "add a compact table (≤5 rows) if helpful; clearly separate facts from interpretations; "
                "cite specific sources for quoted text; keep explanations concise."
            ),
            "citations": [], "metrics": {}, "tables": []
        }

    sheets = [s for s in manifest.get("sheets", []) if isinstance(s, dict) and "parquet_blob_path" in s]
    if not sheets:
        if (time.time() - start) * 1000.0 > budget:
            return {}
        return {"system_message": "Provide a concise analysis only if the data structure is clear. If numeric values are present, report key counts/averages and add a small table.",
                "citations": [], "metrics": {}, "tables": []}

    tables_md: List[str] = []; metrics: Dict[str, Any] = {}
    total_rows = 0; sheet_summaries = []

    for sheet in sheets[:2]:
        if (time.time() - start) * 1000.0 > budget:
            break
        blobname = sheet["parquet_blob_path"]
        local_parquet = _download_blob_to_temp(container_for_data, blobname)
        if not local_parquet or not os.path.exists(local_parquet):
            continue

        used_duckdb = False
        numeric_md = ""
        cat_md = ""
        trend_md = ""

        nrows = None; ncols = None

        if duckdb is not None:
            used_duckdb = True
            # get schema quickly
            try:
                import duckdb as _ddb
                con = _ddb.connect()
                cnt = con.sql(f"SELECT COUNT(*) FROM read_parquet('{local_parquet}')").fetchone()[0]
                nrows = int(cnt)
                # approximate ncols via LIMIT 0
                rel = con.sql(f"SELECT * FROM read_parquet('{local_parquet}') LIMIT 0")
                ncols = len(rel.columns)
                con.close()
            except Exception:
                pass

            aggs = _duckdb_basic_aggregates(duckdb, local_parquet, compute_trend, max_numeric=2, max_categorical=1)
            if aggs.get("numeric"):
                numeric_md = _markdown_table_from_records(aggs["numeric"], ["column","min","mean","median","max","non_null"], max_rows=8)
            if aggs.get("categorical_top"):
                cat_md = _markdown_table_from_records(aggs["categorical_top"], ["column","value","count"], max_rows=10)
            if compute_trend and aggs.get("trend"):
                trend_md = _markdown_table_from_records(aggs["trend"], ["period","count"], max_rows=24)

        if not used_duckdb:
            try:
                df = pd.read_parquet(local_parquet)
            except Exception:
                continue

            nrows, ncols = df.shape
            total_rows += int(nrows)

            prof = _profile_dataframe(pd, df, max_cols=8)
            prof_md = _markdown_table_from_records(prof, ["column","dtype","non_null_pct","is_numeric","avg_str_len","distinct_ratio"], max_rows=8)
            if prof_md:
                tables_md.append(f"**Column profile — {sheet.get('sheet_name','?')}**\n{prof_md}")

            aggs = _basic_aggregates_pandas(pd, df, compute_trend=compute_trend)
            if aggs["numeric"]:
                numeric_md = _markdown_table_from_records(aggs["numeric"], ["column","min","mean","median","max","non_null"], max_rows=8)
            if aggs["categorical_top"]:
                cat_md = _markdown_table_from_records(aggs["categorical_top"], ["column","value","count"], max_rows=10)
            if compute_trend and aggs["trend"]:
                trend_md = _markdown_table_from_records(aggs["trend"], ["period","count"], max_rows=24)
        else:
            total_rows += int(nrows or 0)

        sheet_summaries.append({"sheet": sheet.get("sheet_name","?"), "rows": nrows, "cols": ncols})
        if numeric_md:
            tables_md.append(f"**Numeric summary — {sheet.get('sheet_name','?')}**\n{numeric_md}")
        if cat_md:
            tables_md.append(f"**Top categories — {sheet.get('sheet_name','?')}**\n{cat_md}")
        if trend_md:
            tables_md.append(f"**Trend (count by month) — {sheet.get('sheet_name','?')}**\n{trend_md}")

    if not sheet_summaries:
        if (time.time() - start) * 1000.0 > budget:
            return {}
        return {"system_message": "Provide a concise analysis only if the data structure is clear. If numeric values are present, report key counts/averages and add a small table.",
                "citations": [], "metrics": {}, "tables": []}

    sheets_md = _markdown_table_from_records(sheet_summaries, ["sheet","rows","cols"], max_rows=12)
    if sheets_md:
        tables_md.insert(0, f"**Sheets overview**\n{sheets_md}")

    metrics["sheet_count"] = len(manifest.get("sheets", []))
    metrics["rows_sampled"] = total_rows

    body_lines = [
        "Concise tabular analysis context (prompt-driven):",
        f"- Sheets detected: {metrics['sheet_count']}",
        f"- Rows sampled for profiling: {metrics['rows_sampled']}",
        ""
    ] + tables_md

    system_message = "\n".join(body_lines)

    return {"system_message": system_message, "citations": [], "metrics": metrics, "tables": tables_md}

"""
functions_tabular_ingest.py
Utilities to ingest chat-uploaded CSV/XLS/XLSX files into per-sheet Parquet + manifest,
stored in Azure Blob Storage, consistent with existing workspace ingestion conventions.

Exports:
- ingest_tabular_upload(temp_file_path, original_filename, user_id, group_id=None, update_callback=...)
- read_manifest_for_document(document_id, user_id=None, group_id=None)
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
import os
import json
import time
import uuid
import tempfile

# Optional dependencies
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# Prefer config globals (CLIENTS/containers) provided by the app; fallback to env/defaults
try:
    from config import (
        CLIENTS,
        storage_account_group_documents_container_name as GROUP_CONTAINER,
        storage_account_user_documents_container_name as USER_CONTAINER,
    )
except Exception:
    CLIENTS = {}
    GROUP_CONTAINER = os.environ.get("GROUP_DOCS_CONTAINER") or os.environ.get("BLOB_CONTAINER_GROUP") or "group-documents"
    USER_CONTAINER  = os.environ.get("USER_DOCS_CONTAINER")  or os.environ.get("BLOB_CONTAINER_USER")  or "user-documents"


def _get_blob_client(container: str, blob_path: str):
    """
    Prefer the initialized CLIENTS blob service (used when enable_enhanced_citations = True).
    Fallback to connection string if present.
    """
    # Preferred path: configured client
    try:
        blob_service_client = CLIENTS.get("storage_account_office_docs_client")
        if blob_service_client:
            return blob_service_client.get_container_client(container).get_blob_client(blob_path)
    except Exception:
        pass

    # Fallback using connection string
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore
        conn = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                or os.environ.get("AZUREWebJobsStorage") or "")
        if not conn:
            return None
        bs = BlobServiceClient.from_connection_string(conn)
        return bs.get_container_client(container).get_blob_client(blob_path)
    except Exception:
        return None


def read_manifest_for_document(document_id: str,
                               user_id: Optional[str] = None,
                               group_id: Optional[str] = None) -> Optional[dict]:
    """
    Load the JSON manifest for a previously ingested tabular document.
    Returns manifest dict (with "sheets": [{sheet_name, parquet_blob_path, columns, row_count, rows, cols, ...}]) or None.

    NOTE: This helper reads manifests written at '<prefix>{document_id}__manifest.json'
    to match the existing functions_documents read/write convention.
    """
    container = GROUP_CONTAINER if group_id else USER_CONTAINER
    if not container:
        return None

    prefix = f"{group_id}/" if group_id else f"{user_id}/"
    # Align with existing convention used across the app
    manifest_blob_path = f"{prefix}{document_id}__manifest.json"

    try:
        bc = _get_blob_client(container, manifest_blob_path)
        if not bc:
            return None
        data = bc.download_blob().readall().decode("utf-8", errors="ignore")
        return json.loads(data)
    except Exception:
        return None


def _to_parquet_and_upload(df, container: str, parquet_blob_path: str) -> bool:
    """
    Write a DataFrame to a temp parquet file and upload to blob storage.
    Returns True if upload succeeded, else False.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_pq:
            df.to_parquet(tmp_pq.name, index=False)
            tmp_pq.flush()
            bc = _get_blob_client(container, parquet_blob_path)
            if bc:
                with open(tmp_pq.name, "rb") as fh:
                    bc.upload_blob(fh, overwrite=True)
                return True
            return False
    except Exception:
        return False
    finally:
        try:
            if 'tmp_pq' in locals() and tmp_pq and os.path.exists(tmp_pq.name):
                os.remove(tmp_pq.name)
        except Exception:
            pass


def ingest_tabular_upload(temp_file_path: str,
                          original_filename: str,
                          user_id: str,
                          group_id: Optional[str] = None,
                          update_callback: Callable[..., None] = lambda **kwargs: None
                          ) -> Tuple[Optional[str], Optional[dict]]:
    """
    Ingest a CSV/XLS/XLSX uploaded via chat:
      - Writes per-sheet Parquet files to blob storage
      - Writes a deterministic manifest at '<prefix>{document_id}__manifest.json'
        (with columns, row counts, optional rows/cols, and parquet paths)

    Returns: (document_id, manifest_dict) or (None, None) on failure.
    """
    if pd is None:
        update_callback(status="pandas not available on server")
        return None, None

    document_id = str(uuid.uuid4())
    is_group = group_id is not None
    container = GROUP_CONTAINER if is_group else USER_CONTAINER
    prefix = f"{group_id}/" if is_group else f"{user_id}/"

    # Store per-sheet parquets under a stable path; include document_id folder for clarity.
    base_blob_dir = f"{prefix}{document_id}/"

    file_ext = os.path.splitext(original_filename)[1].lower()
    sheets_meta: List[Dict[str, Any]] = []

    update_callback(status="Reading tabular file...")

    try:
        if file_ext == ".csv":
            # CSV as single sheet
            df = pd.read_csv(temp_file_path, keep_default_na=False, dtype=str)
            parquet_blob_path = f"{base_blob_dir}sheet0.parquet"
            if not _to_parquet_and_upload(df, container, parquet_blob_path):
                update_callback(status="Failed to upload parquet for CSV")
                return None, None
            sheets_meta.append({
                "sheet_name": "Sheet1",
                "row_count": int(len(df)),
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "columns": [str(c) for c in df.columns],
                "parquet_blob_path": parquet_blob_path,
            })

        elif file_ext in (".xls", ".xlsx"):
            # Excel: multi-sheet
            engine = "openpyxl" if file_ext == ".xlsx" else None
            try:
                xls = pd.ExcelFile(temp_file_path, engine=engine)
            except Exception:
                # Fallback without engine hint
                xls = pd.ExcelFile(temp_file_path)

            for i, sn in enumerate(xls.sheet_names):
                try:
                    df = xls.parse(sn, dtype=str)
                except Exception:
                    df = xls.parse(sn)

                parquet_blob_path = f"{base_blob_dir}sheet{i}.parquet"
                if not _to_parquet_and_upload(df, container, parquet_blob_path):
                    update_callback(status=f"Failed to upload parquet for sheet '{sn}'")
                    return None, None

                sheets_meta.append({
                    "sheet_name": sn,
                    "row_count": int(len(df)),
                    "rows": int(len(df)),
                    "cols": int(len(df.columns)),
                    "columns": [str(c) for c in df.columns],
                    "parquet_blob_path": parquet_blob_path,
                })
        else:
            update_callback(status=f"Unsupported tabular extension: {file_ext}")
            return None, None
    except Exception:
        update_callback(status="Failed to read/convert file.")
        return None, None

    manifest = {
        "document_id": document_id,
        "file_name": original_filename,
        "sheets": sheets_meta,
        "created_at": int(time.time()),
        "version": 1
    }

    # Write manifest to match the convention used elsewhere in the app:
    # '<prefix>{document_id}__manifest.json'
    manifest_blob_path = f"{prefix}{document_id}__manifest.json"
    try:
        bc = _get_blob_client(container, manifest_blob_path)
        if bc:
            bc.upload_blob(json.dumps(manifest, ensure_ascii=False).encode("utf-8"), overwrite=True)
        else:
            update_callback(status="Storage client unavailable for manifest upload")
            return None, None
    except Exception:
        update_callback(status="Failed to write manifest.json")
        return None, None

    update_callback(status="Ingestion complete.", number_of_pages=len(sheets_meta))
    return document_id, manifest
# route_backend_analytics.py
from flask import request, jsonify
from typing import Optional
import io
import os
import pandas as pd

# Optional SQL engine (only if you want SQL queries)
try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None

from config import (
    app,
    CLIENTS,
    storage_account_user_documents_container_name,
    storage_account_group_documents_container_name,
)
from functions_documents import (
    detect_doc_type,
    read_manifest_for_document,  # you'll add this helper below
)


def _get_container_and_prefix(user_id: Optional[str], group_id: Optional[str]):
    """
    Resolve the correct blob container + prefix consistent with how manifests/parquets are written.
    """
    if group_id:
        return storage_account_group_documents_container_name, f"{group_id}/"
    if user_id:
        return storage_account_user_documents_container_name, f"{user_id}/"
    return None, ""


def _get_blob_client(container: str, blob_path: str):
    """
    Prefer the initialized CLIENTS blob service (used when enable_enhanced_citations = True).
    Fallback to AZURE_STORAGE_CONNECTION_STRING / AZUREWebJobsStorage if present.
    """
    try:
        blob_service_client = CLIENTS.get("storage_account_office_docs_client")
        if blob_service_client:
            return blob_service_client.get_container_client(container).get_blob_client(blob_path)
    except Exception:
        pass

    try:
        from azure.storage.blob import BlobServiceClient
        conn = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                or os.environ.get("AZUREWebJobsStorage") or "")
        if not conn:
            return None
        bs = BlobServiceClient.from_connection_string(conn)
        return bs.get_container_client(container).get_blob_client(blob_path)
    except Exception:
        return None


def register_route_backend_analytics(app):
    @app.route("/analytics/<document_id>/sheets", methods=["GET"])
    def list_tabular_sheets(document_id):
        """
        Returns the manifest for a tabular document:
        { "document_id": "...", "sheets": [{sheet_name, parquet_blob_path, rows, cols}], ... }
        Query params: user_id (personal) OR group_id (group).
        """
        user_id = request.args.get("user_id")
        group_id = request.args.get("group_id")

        if not document_id or (not user_id and not group_id):
            return jsonify({"error": "document_id and (user_id or group_id) are required"}), 400

        # best-effort scope/ownership guard
        try:
            scope = detect_doc_type(document_id, user_id=user_id)
            if scope is None:
                return jsonify({"error": "Document not found"}), 404
            scope_name, scope_id = scope
            if scope_name == "personal" and user_id and scope_id != user_id:
                return jsonify({"error": "Access denied"}), 403
            if scope_name == "group" and group_id and scope_id != group_id:
                return jsonify({"error": "Access denied"}), 403
        except Exception:
            pass

        manifest = read_manifest_for_document(document_id, user_id, group_id)
        if not manifest:
            return jsonify({"error": "Manifest not found (document may not be tabular or ingestion not complete)"}), 404

        return jsonify(manifest), 200

    @app.route("/analytics/query", methods=["POST"])
    def analytics_query_sheet():
        """
        Body:
        {
          "document_id": "...",
          "user_id": "...",            # or group_id
          "group_id": "...",
          "sheet_name": "Sheet1",      # required
          "columns": ["colA","colB"],  # optional (pandas)
          "filters": { "Country": "US" },  # optional (equals-only)
          "limit": 100,                # optional
          "sql": "SELECT * FROM data WHERE Country='US' LIMIT 100"  # optional; needs duckdb
        }
        """
        body = request.get_json(force=True, silent=True) or {}
        document_id = body.get("document_id")
        user_id = body.get("user_id")
        group_id = body.get("group_id")
        sheet_name = body.get("sheet_name")
        columns = body.get("columns")
        filters = body.get("filters") or {}
        limit = body.get("limit")
        sql = body.get("sql")

        if not document_id or (not user_id and not group_id) or not sheet_name:
            return jsonify({"error": "document_id, sheet_name, and (user_id or group_id) are required"}), 400

        # scope/ownership
        try:
            scope = detect_doc_type(document_id, user_id=user_id)
            if scope is None:
                return jsonify({"error": "Document not found"}), 404
            scope_name, scope_id = scope
            if scope_name == "personal" and user_id and scope_id != user_id:
                return jsonify({"error": "Access denied"}), 403
            if scope_name == "group" and group_id and scope_id != group_id:
                return jsonify({"error": "Access denied"}), 403
        except Exception:
            pass

        manifest = read_manifest_for_document(document_id, user_id, group_id)
        if not manifest or not manifest.get("sheets"):
            return jsonify({"error": "Manifest not found or has no sheets"}), 404

        sheet = next((s for s in manifest["sheets"] if s.get("sheet_name") == sheet_name), None)
        if not sheet:
            return jsonify({"error": f"Sheet '{sheet_name}' not found in manifest"}), 404

        parquet_path = sheet.get("parquet_blob_path")
        if not parquet_path:
            return jsonify({"error": "No parquet_blob_path for this sheet"}), 404

        try:
            container, prefix = _get_container_and_prefix(user_id, group_id)
            if not container:
                return jsonify({"error": "Storage container not configured"}), 500

            # ensure full relative path under the container
            blob_rel_path = parquet_path if parquet_path.startswith(prefix) else f"{prefix}{parquet_path}"
            blob_client = _get_blob_client(container, blob_rel_path)
            if not blob_client:
                return jsonify({"error": "No blob client available (enable Enhanced Citations or set AZURE_STORAGE_CONNECTION_STRING)"}), 500
            if not blob_client.exists():
                return jsonify({"error": f"Parquet not found at {blob_rel_path}"}), 404

            parquet_bytes = blob_client.download_blob().readall()
            buf = io.BytesIO(parquet_bytes)
            df = pd.read_parquet(buf)

            # SQL path (DuckDB)
            if sql:
                if duckdb is None:
                    return jsonify({"error": "SQL queries require DuckDB; install duckdb or omit 'sql' and use filters/columns/limit."}), 400
                con = duckdb.connect()
                try:
                    con.register("data", df)
                    out = con.execute(sql).fetch_df()
                finally:
                    con.close()
            else:
                # Simple pandas filter/column selection
                out = df
                for k, v in filters.items():
                    if k in out.columns:
                        out = out[out[k] == v]
                if columns:
                    cols_present = [c for c in (columns or []) if c in out.columns]
                    if cols_present:
                        out = out[cols_present]
                if isinstance(limit, int) and limit > 0:
                    out = out.head(limit)

            preview = out.head(200).to_dict(orient="records")
            return jsonify({
                "document_id": document_id,
                "sheet_name": sheet_name,
                "rows": int(out.shape[0]),
                "cols": int(out.shape[1]),
                "preview": preview
            }), 200

        except Exception as e:
            # soft log; avoid hard dependency on logging
            try:
                from functions_logging import log_exception
                log_exception(f"/analytics/query failed for {document_id}/{sheet_name}: {e}")
            except Exception:
                pass
            return jsonify({"error": f"Query failed: {str(e)}"}), 500

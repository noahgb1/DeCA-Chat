# functions_documents.py

from config import *
from functions_content import *
from functions_settings import *
from functions_search import *
from functions_logging import *
from functions_authentication import *
import re
import hashlib
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import os
import tempfile
import math
import traceback
import pandas as pd
import requests
import fitz
import time
from typing import Callable

# Flask helper (routes expect jsonify). Provide safe fallback for non-Flask contexts.
try:
    from flask import jsonify
except Exception:
    def jsonify(obj):
        return obj

# HTML parsing: prefer bs4 but allow graceful fallback (file-level logic checks for None).
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# Text splitters: try to import explicit classes (functions_content may already provide them via star import).
try:
    from functions_content import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
except Exception:
    try:
        from some_text_utils import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None
        MarkdownHeaderTextSplitter = None

# Azure Search batch helper: import if available, else set to None (delete_document_chunks checks for this).
try:
    from azure.search.documents import IndexDocumentsBatch
except Exception:
    IndexDocumentsBatch = None

# Ensure Cosmos exception is available (already attempted earlier)
try:
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
except Exception:
    CosmosResourceNotFoundError = Exception

def _write_tabular_manifest(container_name: str, prefix: Optional[str], document_id: str, sheets: List[Dict[str, Any]], source_blob: Optional[str] = None) -> bool:
    """
    Low-level manifest writer. Prefer using the configured CLIENTS blob client when available;
    fall back to AZURE_STORAGE_CONNECTION_STRING / BlobServiceClient if not.
    Writes JSON to blob path: <prefix><document_id>__manifest.json
    Returns True on success, False on failure. Best-effort (does not raise).
    """
    blob_path = f"{(prefix or '')}{document_id}__manifest.json"
    manifest = {
        "document_id": document_id,
        "source_blob_path": source_blob,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sheets": sheets
    }
    try:
        # Prefer the initialized CLIENTS blob service if present (safer in this app)
        try:
            blob_service_client = CLIENTS.get("storage_account_office_docs_client")
        except Exception:
            blob_service_client = None

        if blob_service_client:
            try:
                container_client = blob_service_client.get_container_client(container_name)
                blob_client = container_client.get_blob_client(blob_path)
                blob_client.upload_blob(json.dumps(manifest, ensure_ascii=False).encode("utf-8"), overwrite=True)
                return True
            except Exception:
                # If the CLIENTS-based path fails, fall through to connection-string fallback
                pass

        # Fallback: use connection string + azure.storage.blob
        try:
            from azure.storage.blob import BlobServiceClient
            conn = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                    or os.environ.get("AZUREWebJobsStorage") or "")
            if not conn:
                return False
            bs = BlobServiceClient.from_connection_string(conn)
            bs.get_container_client(container_name).get_blob_client(blob_path).upload_blob(
                json.dumps(manifest, ensure_ascii=False).encode("utf-8"), overwrite=True
            )
            return True
        except Exception as e:
            try:
                log_exception(f"_write_tabular_manifest fallback failed for document_id={document_id}: {e}")
            except Exception:
                pass
            return False

    except Exception as e:
        try:
            log_exception(f"_write_tabular_manifest failed for document_id={document_id}: {e}")
        except Exception:
            pass
        return False

def write_manifest_for_document(document_id: str, user_id: Optional[str], group_id: Optional[str], sheets: List[Dict[str, Any]], source_blob: Optional[str] = None) -> bool:
    """
    Convenience wrapper: choose container & prefix then call _write_tabular_manifest.
    Best-effort; does not raise.
    """
    try:
        # prefer group container when group_id present
        try:
            group_container = globals().get("storage_account_group_documents_container_name", None)
            user_container = globals().get("storage_account_user_documents_container_name", None)
        except Exception:
            group_container = None
            user_container = None

        if group_id and group_container:
            container = group_container
            prefix = f"{group_id}/"
        elif user_id and user_container:
            container = user_container
            prefix = f"{user_id}/"
        else:
            # fallback to in-file GROUP_CONTAINER / USER_CONTAINER if present
            container = globals().get("GROUP_CONTAINER") or globals().get("USER_CONTAINER")
            prefix = (f"{group_id}/" if group_id else (f"{user_id}/" if user_id else ""))

        if not container:
            return False

        return _write_tabular_manifest(container, prefix, document_id, sheets, source_blob=source_blob)
    except Exception as e:
        try:
            log_exception(f"write_manifest_for_document failed for document {document_id}: {e}")
        except Exception:
            pass
        return False
def read_manifest_for_document(document_id: str, user_id: Optional[str], group_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Reads the tabular manifest JSON for a document from blob storage.
    Looks in the same container/prefix scheme used by write_manifest_for_document().
    Returns a dict or None if not found.
    """
    try:
        # resolve container & prefix exactly like write_manifest_for_document()
        try:
            group_container = globals().get("storage_account_group_documents_container_name", None)
            user_container = globals().get("storage_account_user_documents_container_name", None)
        except Exception:
            group_container = None
            user_container = None

        if group_id and group_container:
            container = group_container
            prefix = f"{group_id}/"
        elif user_id and user_container:
            container = user_container
            prefix = f"{user_id}/"
        else:
            container = globals().get("GROUP_CONTAINER") or globals().get("USER_CONTAINER")
            prefix = (f"{group_id}/" if group_id else (f"{user_id}/" if user_id else ""))

        if not container:
            return None

        blob_path = f"{prefix}{document_id}__manifest.json"

        # prefer configured CLIENTS blob service
        try:
            blob_service_client = CLIENTS.get("storage_account_office_docs_client")
        except Exception:
            blob_service_client = None

        if blob_service_client:
            try:
                container_client = blob_service_client.get_container_client(container)
                blob_client = container_client.get_blob_client(blob_path)
                if not blob_client.exists():
                    return None
                data = blob_client.download_blob().readall()
                return json.loads(data.decode("utf-8"))
            except Exception:
                pass

        # fallback to connection string
        try:
            from azure.storage.blob import BlobServiceClient
            conn = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
                    or os.environ.get("AZUREWebJobsStorage") or "")
            if not conn:
                return None
            bs = BlobServiceClient.from_connection_string(conn)
            blob_client = bs.get_container_client(container).get_blob_client(blob_path)
            if not blob_client.exists():
                return None
            data = blob_client.download_blob().readall()
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            try:
                log_exception(f"read_manifest_for_document fallback failed for document_id={document_id}: {e}")
            except Exception:
                pass
            return None

    except Exception as e:
        try:
            log_exception(f"read_manifest_for_document failed for document_id={document_id}: {e}")
        except Exception:
            pass
        return None

def allowed_file(filename, allowed_extensions=None):
    if not allowed_extensions:
        allowed_extensions = ALLOWED_EXTENSIONS
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
def create_document(file_name, user_id, document_id, num_file_chunks, status, group_id=None):
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    is_group = group_id is not None

    # Choose the correct cosmos_container and query parameters
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT * 
            FROM c
            WHERE c.file_name = @file_name 
                AND c.group_id = @group_id
        """
        parameters = [
            {"name": "@file_name", "value": file_name},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT * 
            FROM c
            WHERE c.file_name = @file_name 
                AND c.user_id = @user_id
        """
        parameters = [
            {"name": "@file_name", "value": file_name},
            {"name": "@user_id", "value": user_id}
        ]

    try:
        existing_document = list(
            cosmos_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            )
        )
        version = existing_document[0]['version'] + 1 if existing_document else 1
        
        if is_group:
            document_metadata = {
                "id": document_id,
                "file_name": file_name,
                "num_chunks": 0,
                "number_of_pages": 0,
                "current_file_chunk": 0,
                "num_file_chunks": num_file_chunks,
                "upload_date": current_time,
                "last_updated": current_time,
                "version": version,
                "status": status,
                "percentage_complete": 0,
                "document_classification": "Pending",
                "type": "document_metadata",
                "group_id": group_id
            }
        else:
            document_metadata = {
                "id": document_id,
                "file_name": file_name,
                "num_chunks": 0,
                "number_of_pages": 0,
                "current_file_chunk": 0,
                "num_file_chunks": num_file_chunks,
                "upload_date": current_time,
                "last_updated": current_time,
                "version": version,
                "status": status,
                "percentage_complete": 0,
                "document_classification": "Pending",
                "type": "document_metadata",
                "user_id": user_id
            }

        cosmos_container.upsert_item(document_metadata)

        add_file_task_to_file_processing_log(
            document_id,
            user_id,
            f"Document {file_name} created."
        )

    except Exception as e:
        print(f"Error creating document: {e}")
        raise


def get_document_metadata(document_id, user_id, group_id=None):
    is_group = group_id is not None
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT * 
            FROM c
            WHERE c.id = @document_id 
                AND c.group_id = @group_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT * 
            FROM c
            WHERE c.id = @document_id 
                AND c.user_id = @user_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@user_id", "value": user_id}
        ]

    add_file_task_to_file_processing_log(
        document_id=document_id, 
        user_id=group_id if is_group else user_id,
        content=f"Query is {query}, parameters are {parameters}."
    )
    try:
        document_items = list(
            cosmos_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            )
        )
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Document metadata retrieved: {document_items}."
        )
        return document_items[0] if document_items else None

    except Exception as e:
        print(f"Error retrieving document metadata: {repr(e)}\nTraceback:\n{traceback.format_exc()}")
        return None

def save_video_chunk(
    page_text_content,
    ocr_chunk_text,
    start_time,
    file_name,
    user_id,
    document_id,
    group_id
):
    """
    Saves one 30-second video chunk to the search index, with separate fields for transcript and OCR.
    The chunk_id is built from document_id and the integer second offset to ensure a valid key.
    """
    try:
        current_time = datetime.now(timezone.utc).isoformat()
        is_group = group_id is not None

        # Convert start_time "HH:MM:SS.mmm" to integer seconds
        h, m, s = start_time.split(':')
        seconds = int(h) * 3600 + int(m) * 60 + int(float(s))

        # 1) generate embedding on the transcript text
        try:
            embedding = generate_embedding(page_text_content)
            print(f"[VideoChunk] EMBEDDING OK for {document_id}@{start_time}", flush=True)
        except Exception as e:
            print(f"[VideoChunk] EMBEDDING ERROR for {document_id}@{start_time}: {e}", flush=True)
            return

        # 2) build chunk document
        try:
            meta = get_document_metadata(document_id, user_id, group_id)
            version = meta.get("version", 1) if meta else 1

            # Use integer seconds to build a safe document key
            chunk_id = f"{document_id}_{seconds}"

            chunk = {
                "id":                   chunk_id,
                "document_id":          document_id,
                "chunk_text":           page_text_content,
                "video_ocr_chunk_text": ocr_chunk_text,
                "embedding":            embedding,
                "file_name":            file_name,
                "start_time":           start_time,
                "chunk_sequence":       seconds,
                "upload_date":          current_time,
                "version":              version,
            }

            if is_group:
                chunk["group_id"] = group_id
                client = CLIENTS["search_client_group"]
            else:
                chunk["user_id"] = user_id
                client = CLIENTS["search_client_user"]

            print(f"[VideoChunk] CHUNK BUILT {chunk_id}", flush=True)

        except Exception as e:
            print(f"[VideoChunk] CHUNK BUILD ERROR for {document_id}@{start_time}: {e}", flush=True)
            return

        # 3) upload to search index
        try:
            client.upload_documents(documents=[chunk])
            print(f"[VideoChunk] UPLOAD OK for {chunk_id}", flush=True)
        except Exception as e:
            print(f"[VideoChunk] UPLOAD ERROR for {chunk_id}: {e}", flush=True)

    except Exception as e:
        print(f"[VideoChunk] UNEXPECTED ERROR for {document_id}@{start_time}: {e}", flush=True)



def process_video_document(
    document_id,
    user_id,
    temp_file_path,
    original_filename,
    update_callback,
    group_id
):
    """
    Processes a video by dividing transcript into 30-second chunks,
    extracting OCR separately, and saving each as a chunk with safe IDs.
    """

    def to_seconds(ts: str) -> float:
        parts = ts.split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3:
            h, m, s = parts
        else:
            h = 0.0
            m, s = parts
        return h * 3600 + m * 60 + s

    settings = get_settings()
    if not settings.get("enable_video_file_support", False):
        print("[VIDEO] indexing disabled in settings", flush=True)
        update_callback(status="VIDEO: indexing disabled")
        return 0
    
    if settings.get("enable_enhanced_citations", False):
        update_callback(status="Uploading video for enhanced citations...")
        try:
            # this helper is already in your file below
            blob_path = upload_to_blob(
                temp_file_path,
                user_id,
                document_id,
                original_filename,
                update_callback,
                group_id
            )
            update_callback(status=f"Enhanced citations: video at {blob_path}")
        except Exception as e:
            print(f"[VIDEO] BLOB UPLOAD ERROR: {e}", flush=True)
            update_callback(status=f"VIDEO: blob upload failed → {e}")

    vi_ep, vi_loc, vi_acc = (
        settings["video_indexer_endpoint"],
        settings["video_indexer_location"],
        settings["video_indexer_account_id"]
    )

    # 1) Auth
    try:
        token = get_video_indexer_account_token(settings)
    except Exception as e:
        print(f"[VIDEO] AUTH ERROR: {e}", flush=True)
        update_callback(status=f"VIDEO: auth failed → {e}")
        return 0

    # 2) Upload video to Indexer
    try:
        url = f"{vi_ep}/{vi_loc}/Accounts/{vi_acc}/Videos"
        params = {"accessToken": token, "name": original_filename}
        with open(temp_file_path, "rb") as f:
            resp = requests.post(url, params=params, files={"file": f})
        resp.raise_for_status()
        vid = resp.json().get("id")
        if not vid:
            raise ValueError("no video ID returned")
        print(f"[VIDEO] UPLOAD OK, videoId={vid}", flush=True)
        update_callback(status=f"VIDEO: uploaded id={vid}")
    except Exception as e:
        print(f"[VIDEO] UPLOAD ERROR: {e}", flush=True)
        update_callback(status=f"VIDEO: upload failed → {e}")
        return 0

    # 3) Poll until ready
    index_url = (
        f"{vi_ep}/{vi_loc}/Accounts/{vi_acc}/Videos/{vid}/Index"
        f"?accessToken={token}&includeInsights=Transcript&includeStreamingUrls=false"
    )
    while True:
        r = requests.get(index_url)
        if r.status_code in (401, 404):
            time.sleep(30); continue
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 30))); continue
        if r.status_code == 504:
            time.sleep(30); continue
        r.raise_for_status()
        data = r.json()


        info = data.get("videos", [{}])[0]
        prog = info.get("processingProgress", "0%").rstrip("%")
        state = info.get("state", "").lower()
        update_callback(status=f"VIDEO: {prog}%")
        if state == "failed":
            update_callback(status="VIDEO: indexing failed")
            return 0
        if prog == "100":
            break
        time.sleep(30)

    # 4) Extract transcript & OCR
    insights = info.get("insights", {})
    transcript = insights.get("transcript", [])
    ocr_blocks = insights.get("ocr", [])

    speech_context = [
        {"text": seg["text"].strip(), "start": inst["start"]}
        for seg in transcript if seg.get("text", "").strip()
        for inst in seg.get("instances", [])
    ]
    ocr_context = [
        {"text": block["text"].strip(), "start": inst["start"]}
        for block in ocr_blocks if block.get("text", "").strip()
        for inst in block.get("instances", [])
    ]

    speech_context.sort(key=lambda x: to_seconds(x["start"]))
    ocr_context.sort(key=lambda x: to_seconds(x["start"]))

    total = 0
    idx_s = 0
    n_s = len(speech_context)
    idx_o = 0
    n_o = len(ocr_context)

    while idx_s < n_s:
        window_start = to_seconds(speech_context[idx_s]["start"])
        window_end = window_start + 30.0

        speech_lines = []
        while idx_s < n_s and to_seconds(speech_context[idx_s]["start"]) <= window_end:
            speech_lines.append(speech_context[idx_s]["text"])
            idx_s += 1

        ocr_lines = []
        while idx_o < n_o and to_seconds(ocr_context[idx_o]["start"]) <= window_end:
            ocr_lines.append(ocr_context[idx_o]["text"])
            idx_o += 1

        start_ts = speech_context[total]["start"]
        chunk_text = " ".join(speech_lines).strip()
        ocr_text = " ".join(ocr_lines).strip()

        update_callback(current_file_chunk=total+1, status=f"VIDEO: saving chunk @ {start_ts}")
        save_video_chunk(
            page_text_content=chunk_text,
            ocr_chunk_text=ocr_text,
            start_time=start_ts,
            file_name=original_filename,
            user_id=user_id,
            document_id=document_id,
            group_id=group_id
        )
        total += 1

    update_callback(status=f"VIDEO: done, {total} chunks")
    return total


def calculate_processing_percentage(doc_metadata):
    """
    Calculates a simpler, step-based processing percentage based on status
    and page saving progress.

    Args:
        doc_metadata (dict): The current document metadata dictionary.

    Returns:
        int: The calculated percentage (0-100).
    """
    status = doc_metadata.get('status', '')
    if isinstance(status, str):
        status = status.lower()
    elif isinstance(status, bytes):
        status = status.decode('utf-8').lower()
    elif isinstance(status, dict):
        status = json.dumps(status).lower()
        

    current_pct = doc_metadata.get('percentage_complete', 0)
    estimated_pages = doc_metadata.get('number_of_pages', 0)
    total_chunks_saved = doc_metadata.get('current_file_chunk', 0)

    # --- Final States ---
    if "processing complete" in status or current_pct == 100:
        # Ensure it stays 100 if it ever reached it
        return 100
    if "error" in status or "failed" in status:
        # Keep the last known percentage on error/failure
        return current_pct

    # --- Calculate percentage based on phase/status ---
    calculated_pct = 0

    # Phase 1: Initial steps up to sending to DI
    if "queued" in status:
        calculated_pct = 0

    elif "sending" in status:
        # Explicitly sending data for analysis
        calculated_pct = 5

    # Phase 3: Saving Pages (The main progress happens here: 10% -> 90%)
    elif "saving page" in status or "saving chunk" in status: # Status indicating the loop saving pages is active
        if estimated_pages > 0:
            # Calculate progress ratio (0.0 to 1.0)
            # Ensure saved count doesn't exceed estimate for the ratio
            safe_chunks_saved = min(total_chunks_saved, estimated_pages)
            progress_ratio = safe_chunks_saved / estimated_pages

            # Map the ratio to the percentage range [10, 90]
            # The range covers 80 percentage points (90 - 10)
            calculated_pct = 5 + (progress_ratio * 80)
        else:
            # If page count is unknown, we can't show granular progress.
            # Stay at the beginning of this phase.
            calculated_pct = 5

    # Phase 4: Final Metadata Extraction (Optional, after page saving)
    elif "extracting final metadata" in status:
        # This phase should start after page saving is effectively done (>=90%)
        # Assign a fixed value during this step.
        calculated_pct = 95

    # Default/Fallback: If status doesn't match known phases,
    # use the current percentage. This handles intermediate statuses like
    # "Chunk X/Y saved" which might occur between "saving page" updates.
    else:
        calculated_pct = current_pct


    # --- Final Adjustments ---

    # Cap at 99% - only "Processing Complete" status should trigger 100%
    final_pct = min(int(round(calculated_pct)), 99)

    # Prevent percentage from going down, unless it's due to an error state (handled above)
    # Compare the newly calculated capped percentage with the value read at the function start
    # This ensures progress is monotonic upwards until completion or error.
    return max(final_pct, current_pct)

def update_document(**kwargs):
    document_id = kwargs.get('document_id')
    user_id = kwargs.get('user_id')
    group_id = kwargs.get('group_id')
    num_chunks_increment = kwargs.pop('num_chunks_increment', 0)

    if (not document_id or not user_id) or (not document_id and not group_id):
        # Cannot proceed without these identifiers
        print("Error: document_id and user_id or document_id and group_id are required for update_document")
        # Depending on context, you might raise an error or return failure
        raise ValueError("document_id and user_id or document_id and group_id are required")

    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    is_group = group_id is not None

    # if is_group:
    #     log_msg = f"Group document update requested for {document_id} by group {group_id}."
    # else:
    #     log_msg = f"User document update requested for {document_id} by user {user_id}."

    # add_file_task_to_file_processing_log(
    #     document_id=document_id, 
    #     user_id=group_id if group_id else user_id, 
    #     content=log_msg
    # )

    # Choose the correct cosmos_container and query parameters
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT * 
            FROM c
            WHERE c.id = @document_id 
                AND c.group_id = @group_id
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT * 
            FROM c
            WHERE c.id = @document_id 
                AND c.user_id = @user_id
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@user_id", "value": user_id}
        ]
    
    add_file_task_to_file_processing_log(
        document_id=document_id, 
        user_id=group_id if is_group else user_id, 
        content=f"Query is {query}, parameters are {parameters}."
    )

    try:
        existing_documents = list(
            cosmos_container.query_items(
                query=query, 
                parameters=parameters, 
                enable_cross_partition_query=True
            )
        )

        status = kwargs.get('status', '')

        if status:
            add_file_task_to_file_processing_log(
                document_id=document_id,
                user_id=group_id if is_group else user_id,
                content=f"Status: {status}"
            )

        if not existing_documents:
            # Log specific error before raising
            log_msg = f"Document {document_id} not found for user {user_id} during update."
            print(log_msg)
            add_file_task_to_file_processing_log(
                document_id=document_id, 
                user_id=group_id if is_group else user_id, 
                content=log_msg
            )
            raise CosmosResourceNotFoundError(
                message=f"Document {document_id} not found",
                status=404
            )


        existing_document = existing_documents[0]
        original_percentage = existing_document.get('percentage_complete', 0) # Store for comparison

        # 2. Apply updates from kwargs
        update_occurred = False
        updated_fields_requiring_chunk_sync = set() # Track fields needing propagation

        if num_chunks_increment > 0:
            current_num_chunks = existing_document.get('num_chunks', 0)
            existing_document['num_chunks'] = current_num_chunks + num_chunks_increment
            update_occurred = True # Incrementing counts as an update
            add_file_task_to_file_processing_log(
                document_id=document_id, 
                user_id=group_id if is_group else user_id,  
                content=f"Incrementing num_chunks by {num_chunks_increment} to {existing_document['num_chunks']}"
            )

        for key, value in kwargs.items():
            if value is not None and existing_document.get(key) != value:
                # Avoid overwriting num_chunks if it was just incremented
                if key == 'num_chunks' and num_chunks_increment > 0:
                    continue # Skip direct assignment if increment was used
                existing_document[key] = value
                update_occurred = True
                if key in ['title', 'authors', 'file_name', 'document_classification']:
                    updated_fields_requiring_chunk_sync.add(key)

        # 3. If any update happened, handle timestamps and percentage
        if update_occurred:
            existing_document['last_updated'] = current_time

            # Calculate new percentage based on the *updated* existing_document state
            # This now includes the potentially incremented num_chunks
            new_percentage = calculate_processing_percentage(existing_document)
            
            # Handle final state overrides for percentage

            status_lower = existing_document.get('status', '')
            if isinstance(status_lower, str):
                status_lower = status_lower.lower()
            elif isinstance(status_lower, bytes):
                status_lower = status_lower.decode('utf-8').lower()
            elif isinstance(status_lower, dict):
                status_lower = json.dumps(status_lower).lower()

            if "processing complete" in status_lower:
                new_percentage = 100
            elif "error" in status_lower or "failed" in status_lower:
                 pass # Percentage already calculated by helper based on 'failed' status

            # Ensure percentage doesn't decrease (unless reset on failure or hitting 100)
            # Compare against original_percentage fetched *before* any updates in this call
            if new_percentage < original_percentage and new_percentage != 0 and "failed" not in status_lower and "error" not in status_lower:
                 existing_document['percentage_complete'] = original_percentage
            else:
                 existing_document['percentage_complete'] = new_percentage

        # 4. Propagate relevant changes to search index chunks
        # This happens regardless of 'update_occurred' flag because the *intent* from kwargs might trigger it,
        # even if the main doc update didn't happen (e.g., only percentage changed).
        # However, it's better to only do this if the relevant fields *actually* changed.
        if update_occurred and updated_fields_requiring_chunk_sync:
            try:
                chunks_to_update = get_all_chunks(document_id, user_id)
                for chunk in chunks_to_update:
                    chunk_updates = {}
                    if 'title' in updated_fields_requiring_chunk_sync:
                        chunk_updates['title'] = existing_document.get('title')
                    if 'authors' in updated_fields_requiring_chunk_sync:
                         # Ensure authors is a list for the chunk metadata if needed
                        chunk_updates['author'] = existing_document.get('authors')
                    if 'file_name' in updated_fields_requiring_chunk_sync:
                        chunk_updates['file_name'] = existing_document.get('file_name')
                    if 'document_classification' in updated_fields_requiring_chunk_sync:
                        chunk_updates['document_classification'] = existing_document.get('document_classification')

                    if chunk_updates: # Only call update if there's something to change
                         update_chunk_metadata(chunk_id=chunk['id'], user_id=user_id, document_id=document_id, group_id=group_id, **chunk_updates)
                add_file_task_to_file_processing_log(
                    document_id=document_id, 
                    user_id=group_id if is_group else user_id,
                    content=f"Propagated updates for fields {updated_fields_requiring_chunk_sync} to search chunks."
                )
            except Exception as chunk_sync_error:
                # Log error but don't necessarily fail the whole document update
                error_msg = f"Warning: Failed to sync metadata updates to search chunks for doc {document_id}: {chunk_sync_error}"
                print(error_msg)
                add_file_task_to_file_processing_log(
                    document_id=document_id, 
                    user_id=group_id if is_group else user_id, 
                    content=error_msg
                )


        # 5. Upsert the document if changes were made
        if update_occurred:
            cosmos_container.upsert_item(existing_document)

    except CosmosResourceNotFoundError as e:
        # Error already logged where it was first detected
        print(f"Document {document_id} not found or access denied: {e}")
        raise # Re-raise for the caller to handle
    except Exception as e:
        error_msg = f"Error during update_document for {document_id}: {repr(e)}\nTraceback:\n{traceback.format_exc()}"
        print(error_msg)
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=error_msg
        )
        # Optionally update status to failure here if the exception is critical
        # try:
        #    existing_document['status'] = f"Update failed: {str(e)[:100]}" # Truncate error
        #    existing_document['percentage_complete'] = calculate_processing_percentage(existing_document) # Recalculate % based on failure
        #    documents_container.upsert_item(existing_document)
        # except Exception as inner_e:
        #    print(f"Failed to update status to error state for {document_id}: {inner_e}")
        raise # Re-raise the original exception

def save_chunks(page_text_content, page_number, file_name, user_id, document_id, group_id=None):
    """
    Save a single chunk (one page) at a time:
      - Generate embedding
      - Build chunk metadata
      - Upload to Search index
    """
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    is_group = group_id is not None

    # Choose the correct cosmos_container and query parameters
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    try:
        # Update document status
        #num_chunks = 1  # because we only have one chunk (page) here
        #status = f"Processing 1 chunk (page {page_number})"
        #update_document(document_id=document_id, user_id=user_id, status=status)
        
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id, 
            content=f"Saving chunk, cosmos_container:{cosmos_container}, page_text_content:{page_text_content}, page_number:{page_number}, file_name:{file_name}, user_id:{user_id}, document_id:{document_id}, group_id:{group_id}"
        )

        if is_group:
            metadata = get_document_metadata(
                document_id=document_id, 
                user_id=user_id, 
                group_id=group_id
            )
        else:
            metadata = get_document_metadata(
                document_id=document_id, 
                user_id=user_id
            )

        if not metadata:
            raise ValueError(f"No metadata found for document {document_id} (group: {is_group})")

        version = metadata.get("version") if metadata.get("version") else 1 
        if version is None:
            raise ValueError(f"Metadata for document {document_id} missing 'version' field")
        
    except Exception as e:
        print(f"Error updating document status or retrieving metadata for document {document_id}: {repr(e)}\nTraceback:\n{traceback.format_exc()}")
        raise

    ####noah changed this####
    # Heuristic: extract sheet name from the first 200 chars when present like "[Sheet: <name>]".
    sheet_name_match = re.search(r"\[\s*Sheet\s*:\s*([^\]]+)\]", page_text_content[:200] or "")
    sheet_name = sheet_name_match.group(1).strip() if sheet_name_match else ""
    ######
    # Generate embedding
    try:
        #status = f"Generating embedding for page {page_number}"
        #update_document(document_id=document_id, user_id=user_id, status=status)
        embedding = generate_embedding(page_text_content)
    except Exception as e:
        print(f"Error generating embedding for page {page_number} of document {document_id}: {e}")
        raise

    # Build chunk document
    try:
        ####noah changed this####
        # Collision-proof chunk id across sheets/pages
        chunk_id = f"{document_id}_" + hashlib.sha1(f"{sheet_name}|{page_number}|{file_name}".encode("utf-8")).hexdigest()[:16]
        ######
        chunk_keywords = []
        chunk_summary = ""
        author = []
        title = ""

        if is_group:
            chunk_document = {
                "id": chunk_id,
                "document_id": document_id,
                ####noah changed this####
                "chunk_id": f"{sheet_name}_{page_number}" if sheet_name else str(page_number),
                ######
                "chunk_text": page_text_content,
                "embedding": embedding,
                "file_name": file_name,
                ####noah changed this####
                "sheet_name": sheet_name,
                ######
                "chunk_keywords": chunk_keywords,
                "chunk_summary": chunk_summary,
                "page_number": page_number,
                "author": author,
                "title": title,
                "document_classification": "Pending",
                "chunk_sequence": page_number,  # or you can keep an incremental idx
                "upload_date": current_time,
                "version": version,
                "group_id": group_id
            }
        else:
            chunk_document = {
                "id": chunk_id,
                "document_id": document_id,
                ####noah changed this####
                "chunk_id": f"{sheet_name}_{page_number}" if sheet_name else str(page_number),
                ######
                "chunk_text": page_text_content,
                "embedding": embedding,
                "file_name": file_name,
                ####noah changed this####
                "sheet_name": sheet_name,
                ######
                "chunk_keywords": chunk_keywords,
                "chunk_summary": chunk_summary,
                "page_number": page_number,
                "author": author,
                "title": title,
                "document_classification": "Pending",
                "chunk_sequence": page_number,  # or you can keep an incremental idx
                "upload_date": current_time,
                "version": version,
                "user_id": user_id
            }
    except Exception as e:
        print(f"Error creating chunk document for page {page_number} of document {document_id}: {e}")
        raise

    # Upload chunk document to Search
    try:
        #status = f"Uploading page {page_number} of document {document_id} to index."
        #update_document(document_id=document_id, user_id=user_id, status=status)

        search_client = CLIENTS["search_client_group"] if is_group else CLIENTS["search_client_user"]
        # Upload as a single-document list
        search_client.upload_documents(documents=[chunk_document])

    except Exception as e:
        print(f"Error uploading chunk document for document {document_id}: {e}")
        raise

def get_all_chunks(document_id, user_id, group_id=None):
    is_group = group_id is not None

    search_client = CLIENTS["search_client_group"] if is_group else CLIENTS["search_client_user"]
    filter_expr = (
        f"document_id eq '{document_id}' and group_id eq '{group_id}'"
        if is_group else
        f"document_id eq '{document_id}' and user_id eq '{user_id}'"
    )

    select_fields = [
        "id", 
        "chunk_text", 
        "chunk_id", 
        "file_name",
        "group_id" if is_group else "user_id",
        "version", 
        "chunk_sequence", 
        "upload_date"
    ]

    try:
        results = search_client.search(
            search_text="*",
            filter=filter_expr,
            select=",".join(select_fields)
        )
        return results

    except Exception as e:
        print(f"Error retrieving chunks for document {document_id}: {e}")
        raise

def update_chunk_metadata(chunk_id, user_id, group_id, document_id, **kwargs):
    is_group = group_id is not None

    try:
        search_client = CLIENTS["search_client_group"] if is_group else CLIENTS["search_client_user"]
        chunk_item = search_client.get_document(key=chunk_id)

        if not chunk_item:
            raise Exception("Chunk not found")

        if chunk_item.get('user_id') != user_id or (is_group and chunk_item.get('group_id') != group_id):
            raise Exception("Unauthorized access to chunk")

        if chunk_item.get('document_id') != document_id:
            raise Exception("Chunk does not belong to document")

        # Update only supported fields
        updatable_fields = [
            'chunk_keywords',
            'chunk_summary',
            'author',
            'title',
            'document_classification'
        ]
        for field in updatable_fields:
            if field in kwargs:
                chunk_item[field] = kwargs[field]

        search_client.upload_documents(documents=[chunk_item])

    except Exception as e:
        print(f"Error updating chunk metadata for chunk {chunk_id}: {e}")
        raise

def get_pdf_page_count(pdf_path: str) -> int:
    """
    Returns the total number of pages in the given PDF using PyMuPDF.
    """
    try:
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception as e:
        print(f"Error reading PDF page count: {e}")
        return 0

def chunk_pdf(input_pdf_path: str, max_pages: int = 500) -> list:
    """
    Splits a PDF into multiple PDFs, each with up to `max_pages` pages,
    using PyMuPDF. Returns a list of file paths for the newly created chunks.
    """
    chunks = []
    try:
        with fitz.open(input_pdf_path) as doc:
            total_pages = doc.page_count
            current_page = 0
            chunk_index = 1
            
            base_name, ext = os.path.splitext(input_pdf_path)
            
            # Loop through the PDF in increments of `max_pages`
            while current_page < total_pages:
                end_page = min(current_page + max_pages, total_pages)
                
                # Create a new, empty document for this chunk
                chunk_doc = fitz.open()
                
                # Insert the range of pages in one go
                chunk_doc.insert_pdf(doc, from_page=current_page, to_page=end_page - 1)
                
                chunk_pdf_path = f"{base_name}_chunk_{chunk_index}{ext}"
                chunk_doc.save(chunk_pdf_path)
                chunk_doc.close()
                
                chunks.append(chunk_pdf_path)
                
                current_page = end_page
                chunk_index += 1

    except Exception as e:
        print(f"Error chunking PDF: {e}")

    return chunks

def get_documents(user_id, group_id=None):
    is_group = group_id is not None

    # Choose the correct cosmos_container and query parameters
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT * 
            FROM c
            WHERE c.group_id = @group_id
        """
        parameters = [
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT * 
            FROM c
            WHERE c.user_id = @user_id
        """
        parameters = [
            {"name": "@user_id", "value": user_id}
        ]
    
    try:       
        documents = list(
            cosmos_container.query_items(
                query=query,
                parameters=parameters, 
                enable_cross_partition_query=True
            )
        )

        latest_documents = {}

        for doc in documents:
            file_name = doc['file_name']
            if file_name not in latest_documents or doc['version'] > latest_documents[file_name]['version']:
                latest_documents[file_name] = doc
                
        return jsonify({"documents": list(latest_documents.values())}), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving documents: {str(e)}'}), 500

def get_document(user_id, document_id, group_id=None):
    is_group = group_id is not None

    # Choose the correct cosmos_container and query parameters
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT TOP 1 * 
            FROM c
            WHERE c.id = @document_id 
                AND c.group_id = @group_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT TOP 1 * 
            FROM c
            WHERE c.id = @document_id 
                AND c.user_id = @user_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@user_id", "value": user_id}
        ]

    try:
        document_results = list(
            cosmos_container.query_items(
                query=query, 
                parameters=parameters, 
                enable_cross_partition_query=True
            )
        )

        if not document_results:
            return jsonify({'error': 'Document not found or access denied'}), 404

        return jsonify(document_results[0]), 200

    except Exception as e:
        return jsonify({'error': f'Error retrieving document: {str(e)}'}), 500

def get_latest_version(document_id, user_id, group_id=None):
    is_group = group_id is not None
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT c.version 
            FROM c
            WHERE c.id = @document_id 
                AND c.group_id = @group_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT c.version
            FROM c
            WHERE c.id = @document_id 
                AND c.user_id = @user_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@user_id", "value": user_id}
        ]

    try:
        results = list(
            cosmos_container.query_items(
                query=query, 
                parameters=parameters, 
                enable_cross_partition_query=True
            )
        )

        if results:
            return results[0]['version']
        else:
            return None

    except Exception as e:
        return None
    
def get_document_version(user_id, document_id, version, group_id=None):
    is_group = group_id is not None
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT * 
            FROM c
            WHERE c.id = @document_id
                AND c.version = @version
                AND c.group_id = @group_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@version", "value": version},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT *
            FROM c
            WHERE c.id = @document_id 
                AND c.version = @version
                AND c.user_id = @user_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@version", "value": version},
            {"name": "@user_id", "value": user_id}
        ]

    try:
        document_results = list(
            cosmos_container.query_items(
                query=query, 
                parameters=parameters, 
                enable_cross_partition_query=True
            )
        )

        if not document_results:
            return jsonify({'error': 'Document version not found'}), 404

        return jsonify(document_results[0]), 200

    except Exception as e:
        return jsonify({'error': f'Error retrieving document version: {str(e)}'}), 500

def delete_from_blob_storage(document_id, user_id, file_name, group_id=None):
    """Delete a document from Azure Blob Storage."""
    is_group = group_id is not None
    storage_account_container_name = (
        storage_account_group_documents_container_name
        if is_group else
        storage_account_user_documents_container_name
    )
    
    # Check if enhanced citations are enabled and blob client is available
    settings = get_settings()
    enable_enhanced_citations = settings.get("enable_enhanced_citations", False)
    
    if not enable_enhanced_citations:
        return  # No need to proceed if enhanced citations are disabled
    
    try:
        # Construct the blob path using the same format as in upload_to_blob
        blob_path = f"{group_id}/{file_name}" if is_group else f"{user_id}/{file_name}"
        
        # Get the blob client
        blob_service_client = CLIENTS.get("storage_account_office_docs_client")
        if not blob_service_client:
            print(f"Warning: Enhanced citations enabled but blob service client not configured.")
            return
            
        # Get container client
        container_client = blob_service_client.get_container_client(storage_account_container_name)
        if not container_client:
            print(f"Warning: Could not get container client for {storage_account_container_name}")
            return
            
        # Get blob client
        blob_client = container_client.get_blob_client(blob_path)
        
        # Delete the blob if it exists
        if blob_client.exists():
            blob_client.delete_blob()
            print(f"Successfully deleted blob at {blob_path}")
        else:
            print(f"No blob found at {blob_path} to delete")
            
    except Exception as e:
        print(f"Error deleting document from blob storage: {str(e)}")
        # Don't raise the exception, as we want the Cosmos DB deletion to proceed
        # even if blob deletion fails

def delete_document(user_id, document_id, group_id=None):
    """Delete a document from the user's documents in Cosmos DB and blob storage if enhanced citations are enabled."""
    is_group = group_id is not None
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    try:
        document_item = cosmos_container.read_item(
            item=document_id,
            partition_key=document_id
        )

        if (document_item.get('user_id') != user_id) or (is_group and document_item.get('group_id') != group_id):
            raise Exception("Unauthorized access to document")
            
        # Get the file name from the document to use for blob deletion
        file_name = document_item.get('file_name')
        
        # First try to delete from blob storage
        try:
            if file_name:
                delete_from_blob_storage(document_id, user_id, file_name, group_id)
        except Exception as blob_error:
            # Log the error but continue with Cosmos DB deletion
            print(f"Error deleting from blob storage (continuing with document deletion): {str(blob_error)}")
        
        # Then delete from Cosmos DB
        cosmos_container.delete_item(
            item=document_id,
            partition_key=document_id
        )

    except CosmosResourceNotFoundError:
        raise Exception("Document not found")
    except Exception as e:
        raise

def delete_document_chunks(document_id, group_id=None):
    """Delete document chunks from Azure Cognitive Search index."""

    is_group = group_id is not None

    try:
        search_client = CLIENTS["search_client_group"] if is_group else CLIENTS["search_client_user"]
        results = search_client.search(
            search_text="*",
            filter=f"document_id eq '{document_id}'",
            select=["id"]
        )

        ids_to_delete = [doc['id'] for doc in results]

        if not ids_to_delete:
            return

        documents_to_delete = [{"id": doc_id} for doc_id in ids_to_delete]
        batch = IndexDocumentsBatch()
        batch.add_delete_actions(documents_to_delete)
        result = search_client.index_documents(batch)
    except Exception as e:
        raise

def delete_document_version_chunks(document_id, version, group_id=None):
    """Delete document chunks from Azure Cognitive Search index for a specific version."""
    is_group = group_id is not None
    search_client = CLIENTS["search_client_group"] if is_group else CLIENTS["search_client_user"]

    search_client.delete_documents(
        actions=[
            {"@search.action": "delete", "id": chunk['id']} for chunk in 
            search_client.search(
                search_text="*",
                filter=f"document_id eq '{document_id}' and version eq {version}",
                select="id"
            )
        ]
    )

def get_document_versions(user_id, document_id, group_id=None):
    """ Get all versions of a document for a user."""
    is_group = group_id is not None
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    if is_group:
        query = """
            SELECT c.id, c.file_name, c.version, c.upload_date
            FROM c
            WHERE c.id = @document_id
                AND c.group_id = @group_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT c.id, c.file_name, c.version, c.upload_date
            FROM c
            WHERE c.id = @document_id 
                AND c.user_id = @user_id
            ORDER BY c.version DESC
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@user_id", "value": user_id}
        ]

    try:
        versions_results = list(
            cosmos_container.query_items(
                query=query, 
                parameters=parameters, 
                enable_cross_partition_query=True
            )
        )

        if not versions_results:
            return []
        return versions_results

    except Exception as e:
        return []
    
def detect_doc_type(document_id, user_id=None):
    """
    Check Cosmos to see if this doc belongs to the user's docs (has user_id)
    or the group's docs (has group_id).
    Returns one of: "user", "group", or None if not found.
    Optionally checks if user_id matches (for user docs).
    """

    try:
        doc_item = cosmos_user_documents_container.read_item(
            document_id, 
            partition_key=document_id
        )
        if user_id and doc_item.get('user_id') != user_id:
            pass
        else:
            return "personal", doc_item['user_id']
    except:
        pass

    try:
        group_doc_item = cosmos_group_documents_container.read_item(
            document_id, 
            partition_key=document_id
        )
        return "group", group_doc_item['group_id']
    except:
        pass

    return None

def process_metadata_extraction_background(document_id, user_id, group_id=None):
    """
    Background function that calls extract_document_metadata(...)
    and updates Cosmos DB accordingly.
    """
    is_group = group_id is not None

    try:
        # Log status: starting
        args = {
            "document_id": document_id,
            "user_id": user_id,
            "percentage_complete": 5,
            "status": "Metadata extraction started..."
        }

        if is_group:
            args["group_id"] = group_id

        update_document(**args)

        # Call your existing extraction function
        args = {
            "document_id": document_id,
            "user_id": user_id
        }

        if is_group:
            args["group_id"] = group_id

        metadata = extract_document_metadata(**args)


        if not metadata:
            # If it fails or returns nothing, log an error status and quit
            args = {
                "document_id": document_id,
                "user_id": user_id,
                "status": "Metadata extraction returned empty or failed"
            }

            if is_group:
                args["group_id"] = group_id

            update_document(**args)

            return

        # Persist the returned metadata fields back into Cosmos
        args_metadata = {
            "document_id": document_id,
            "user_id": user_id,
            "title": metadata.get('title'),
            "authors": metadata.get('authors'),
            "abstract": metadata.get('abstract'),
            "keywords": metadata.get('keywords'),
            "publication_date": metadata.get('publication_date'),
            "organization": metadata.get('organization')
        }

        if is_group:
            args_metadata["group_id"] = group_id

        update_document(**args_metadata)

        args_status = {
            "document_id": document_id,
            "user_id": user_id,
            "status": "Metadata extraction complete",
            "percentage_complete": 100
        }

        if is_group:
            args_status["group_id"] = group_id

        update_document(**args_status)

    except Exception as e:
        # Log any exceptions
        args = {
            "document_id": document_id,
            "user_id": user_id,
            "status": f"Metadata extraction failed: {str(e)}"
        }

        if is_group:
            args["group_id"] = group_id

        update_document(**args)

        
def extract_document_metadata(document_id, user_id, group_id=None):
    """
    Extract metadata from a document stored in Cosmos DB.
    This function is called in the background after the document is uploaded.
    It retrieves the document from Cosmos DB, extracts metadata, and performs
    content safety checks.
    """

    settings = get_settings()
    enable_gpt_apim = settings.get('enable_gpt_apim', False)
    enable_user_workspace = settings.get('enable_user_workspace', False)
    enable_group_workspaces = settings.get('enable_group_workspaces', False)

    is_group = group_id is not None
    cosmos_container = cosmos_group_documents_container if is_group else cosmos_user_documents_container
    id_key = "group_id" if is_group else "user_id"
    id_value = group_id if is_group else user_id

    add_file_task_to_file_processing_log(
        document_id=document_id, 
        user_id=group_id if is_group else user_id,
        content=f"Querying metadata for document {document_id} and user {user_id}"
    )
    
    # Example structure for reference
    meta_data_example = {
        "title": "Title here",
        "authors": ["Author 1", "Author 2"],
        "organization": "Organization or Unknown",
        "publication_date": "MM/YYYY or N/A",
        "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
        "abstract": "two sentence abstract"
    }
    
    # Pre-initialize metadata dictionary
    meta_data = {
        "title": "",
        "authors": [],
        "organization": "",
        "publication_date": "",
        "keywords": [],
        "abstract": ""
    }

    if is_group:
        query = """
            SELECT *
            FROM c
            WHERE c.id = @document_id
                AND c.group_id = @group_id
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@group_id", "value": group_id}
        ]
    else:
        query = """
            SELECT *
            FROM c
            WHERE c.id = @document_id 
                AND c.user_id = @user_id
        """
        parameters = [
            {"name": "@document_id", "value": document_id},
            {"name": "@user_id", "value": user_id}
        ]

    # --- Step 1: Retrieve document from Cosmos ---
    try:
        document_items = list(
            cosmos_container.query_items(
                query=query, 
                parameters=parameters, 
                enable_cross_partition_query=True
            )
        )

        args = {
            "document_id": document_id,
            "user_id": user_id,
            "status": f"Retrieved document items for document {document_id}"
        }

        if is_group:
            args["group_id"] = group_id

        update_document(**args)


        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Retrieved document items for document {document_id}: {document_items}"
        )
    except Exception as e:
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Error querying document items for document {document_id}: {e}"
        )
        print(f"Error querying document items for document {document_id}: {e}")

    if not document_items:
        return None

    document_metadata = document_items[0]
    
    # --- Step 2: Populate meta_data from DB ---
    # Convert the DB fields to the correct structure
    if "title" in document_metadata:
        meta_data["title"] = document_metadata["title"]
    if "authors" in document_metadata:
        meta_data["authors"] = ensure_list(document_metadata["authors"])
    if "organization" in document_metadata:
        meta_data["organization"] = document_metadata["organization"]
    if "publication_date" in document_metadata:
        meta_data["publication_date"] = document_metadata["publication_date"]
    if "keywords" in document_metadata:
        meta_data["keywords"] = ensure_list(document_metadata["keywords"])
    if "abstract" in document_metadata:
        meta_data["abstract"] = document_metadata["abstract"]

    add_file_task_to_file_processing_log(
        document_id=document_id, 
        user_id=group_id if is_group else user_id,
        content=f"Extracted metadata for document {document_id}, metadata: {meta_data}"
    )

    args = {
        "document_id": document_id,
        "user_id": user_id,
        "status": f"Extracted metadata for document {document_id}"
    }

    if is_group:
        args["group_id"] = group_id

    update_document(**args)


    # --- Step 3: Content Safety Check (if enabled) ---
    if settings.get('enable_content_safety') and "content_safety_client" in CLIENTS:
        content_safety_client = CLIENTS["content_safety_client"]
        blocked = False
        block_reasons = []
        triggered_categories = []
        blocklist_matches = []

        try:
            request_obj = AnalyzeTextOptions(text=json.dumps(meta_data))
            cs_response = content_safety_client.analyze_text(request_obj)

            max_severity = 0
            for cat_result in cs_response.categories_analysis:
                triggered_categories.append({
                    "category": cat_result.category,
                    "severity": cat_result.severity
                })
                if cat_result.severity > max_severity:
                    max_severity = cat_result.severity

            if cs_response.blocklists_match:
                for match in cs_response.blocklists_match:
                    blocklist_matches.append({
                        "blocklistName": match.blocklist_name,
                        "blocklistItemId": match.blocklist_item_id,
                        "blocklistItemText": match.blocklist_item_text
                    })

            if max_severity >= 4:
                blocked = True
                block_reasons.append("Max severity >= 4")
            if blocklist_matches:
                blocked = True
                block_reasons.append("Blocklist match")
            
            if blocked:
                add_file_task_to_file_processing_log(
                    document_id=document_id, 
                    user_id=group_id if is_group else user_id,
                    content=f"Blocked document metadata: {document_metadata}, reasons: {block_reasons}"
                )
                print(f"Blocked document metadata: {document_metadata}\nReasons: {block_reasons}")
                return None

        except Exception as e:
            add_file_task_to_file_processing_log(
                document_id=document_id, 
                user_id=group_id if is_group else user_id,
                content=f"Error checking content safety for document metadata: {e}"
            )
            print(f"Error checking content safety for document metadata: {e}")

    # --- Step 4: Hybrid Search ---
    try:
        if enable_user_workspace or enable_group_workspaces:
            add_file_task_to_file_processing_log(
                document_id=document_id, 
                user_id=group_id if is_group else user_id,
                content=f"Processing Hybrid search for document {document_id} using json dump of metadata {json.dumps(meta_data)}"
            )

            args = {
                "document_id": document_id,
                "user_id": user_id,
                "status": f"Collecting document data to generate metadata from document: {document_id}"
            }

            if is_group:
                args["group_id"] = group_id

            update_document(**args)


            document_scope, scope_id = detect_doc_type(
                document_id, 
                user_id
            )

            if document_scope == "personal":
                search_results = hybrid_search(
                    json.dumps(meta_data), 
                    user_id, 
                    document_id=document_id, 
                    top_n=12, 
                    doc_scope=document_scope
                )
            elif document_scope == "group":
                search_results = hybrid_search(
                    json.dumps(meta_data), 
                    user_id, 
                    document_id=document_id,
                    top_n=12, 
                    doc_scope=document_scope, 
                    active_group_id=scope_id
                )

        else:
            search_results = "No Hybrid results"
    except Exception as e:
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Error processing Hybrid search for document {document_id}: {e}"
        )
        print(f"Error processing Hybrid search for document {document_id}: {e}")
        search_results = "No Hybrid results"

    gpt_model = settings.get('metadata_extraction_model')

    # --- Step 5: Prepare GPT Client ---
    if enable_gpt_apim:
        # APIM-based GPT client
        gpt_client = AzureOpenAI(
            api_version=settings.get('azure_apim_gpt_api_version'),
            azure_endpoint=settings.get('azure_apim_gpt_endpoint'),
            api_key=settings.get('azure_apim_gpt_subscription_key')
        )
    else:
        # Standard Azure OpenAI approach
        if settings.get('azure_openai_gpt_authentication_type') == 'managed_identity':
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(), 
                cognitive_services_scope
            )
            gpt_client = AzureOpenAI(
                api_version=settings.get('azure_openai_gpt_api_version'),
                azure_endpoint=settings.get('azure_openai_gpt_endpoint'),
                azure_ad_token_provider=token_provider
            )
        else:
            gpt_client = AzureOpenAI(
                api_version=settings.get('azure_openai_gpt_api_version'),
                azure_endpoint=settings.get('azure_openai_gpt_endpoint'),
                api_key=settings.get('azure_openai_gpt_key')
            )

    # --- Step 6: GPT Prompt and JSON Parsing ---
    try:
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Sending search results to AI to generate metadata {document_id}"
        )
        messages = [
            {
                "role": "system", 
                "content": "You are an AI assistant that extracts metadata. Return valid JSON."
            },
            {
                "role": "user", 
                "content": (
                    f"Search results from AI search index:\n{search_results}\n\n"
                    f"Current known metadata:\n{json.dumps(meta_data, indent=2)}\n\n"
                    f"Desired metadata structure:\n{json.dumps(meta_data_example, indent=2)}\n\n"
                    f"Please attempt to fill in any missing, or empty values."
                    f"If generating keywords, please create 5-10 keywords."
                    f"Return only JSON."
                )
            }
        ]

        response = gpt_client.chat.completions.create(
            model=gpt_model, 
            messages=messages
        )
        
    except Exception as e:
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Error processing GPT request for document {document_id}: {e}"
        )
        print(f"Error processing GPT request for document {document_id}: {e}")
        return meta_data  # Return what we have so far
    
    if not response:
        return meta_data  # or None, depending on your logic

    response_content = response.choices[0].message.content
    add_file_task_to_file_processing_log(
        document_id=document_id, 
        user_id=group_id if is_group else user_id,
        content=f"GPT response for document {document_id}: {response_content}"
    )

    # --- Step 7: Clean and parse the GPT JSON output ---
    try:
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Decoding JSON from GPT response for document {document_id}"
        )

        cleaned_str = clean_json_codeFence(response_content)

        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Cleaned JSON from GPT response for document {document_id}: {cleaned_str}"
        )

        gpt_output = json.loads(cleaned_str)

        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Decoded JSON from GPT response for document {document_id}: {gpt_output}"
        )

        # Ensure authors and keywords are always lists
        gpt_output["authors"] = ensure_list(gpt_output.get("authors", []))
        gpt_output["keywords"] = ensure_list(gpt_output.get("keywords", []))

    except (json.JSONDecodeError, TypeError) as e:
        add_file_task_to_file_processing_log(
            document_id=document_id, 
            user_id=group_id if is_group else user_id,
            content=f"Error decoding JSON from GPT response for document {document_id}: {e}"
        )
        print(f"Error decoding JSON from response: {e}")
        return meta_data  # or None

    # --- Step 8: Merge GPT Output with Existing Metadata ---
    #
    # If the DB’s version is effectively empty/worthless, then overwrite 
    # with the GPT’s version if GPT has something non-empty.
    # Otherwise keep the DB’s version.
    #

    # Title
    if is_effectively_empty(meta_data["title"]):
        meta_data["title"] = gpt_output.get("title", meta_data["title"])

    # Authors
    if is_effectively_empty(meta_data["authors"]):
        # If GPT has no authors either, fallback to ["Unknown"]
        meta_data["authors"] = gpt_output["authors"] or ["Unknown"]

    # Organization
    if is_effectively_empty(meta_data["organization"]):
        meta_data["organization"] = gpt_output.get("organization", meta_data["organization"])

    # Publication Date
    if is_effectively_empty(meta_data["publication_date"]):
        meta_data["publication_date"] = gpt_output.get("publication_date", meta_data["publication_date"])

    # Keywords
    if is_effectively_empty(meta_data["keywords"]):
        meta_data["keywords"] = gpt_output["keywords"]

    # Abstract
    if is_effectively_empty(meta_data["abstract"]):
        meta_data["abstract"] = gpt_output.get("abstract", meta_data["abstract"])

    add_file_task_to_file_processing_log(
        document_id=document_id, 
        user_id=group_id if is_group else user_id,
        content=f"Final metadata for document {document_id}: {meta_data}"
    )

    args = {
        "document_id": document_id,
        "user_id": user_id,
        "status": f"Metadata generated for document {document_id}"
    }

    if is_group:
        args["group_id"] = group_id

    update_document(**args)


    return meta_data


def clean_json_codeFence(response_content: str) -> str:
    """
    Removes leading and trailing triple-backticks (```) or ```json
    from a string so that it can be parsed as JSON.
    """
    # Remove any ```json or ``` (with optional whitespace/newlines) at the start
    cleaned = re.sub(r"(?s)^```(?:json)?\s*", "", response_content.strip())
    # Remove trailing ``` on its own line or at the end
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return cleaned.strip()

def ensure_list(value, delimiters=r"[;,]"):
    """
    Ensures the provided value is returned as a list of strings.
    - If `value` is already a list, it is returned as-is.
    - If `value` is a string, it is split on the given delimiters
      (default: commas and semicolons).
    - Otherwise, return an empty list.
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        # Split on the given delimiters (commas, semicolons, etc.)
        items = re.split(delimiters, value)
        # Strip whitespace and remove empty strings
        items = [item.strip() for item in items if item.strip()]
        return items
    else:
        return []

def is_effectively_empty(value):
    """
    Returns True if the value is 'worthless' or empty.
    - For a string: empty or just whitespace
    - For a list: empty OR all empty strings
    - For None: obviously empty
    - For other types: not considered here, but you can extend as needed
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()  # '' or whitespace is empty
    if isinstance(value, list):
        # Example: [] or [''] or [' ', ''] is empty
        # If *every* item is effectively empty as a string, treat as empty
        if len(value) == 0:
            return True
        return all(not item.strip() for item in value if isinstance(item, str))
    return False

# --- Helper function to estimate word count ---
def estimate_word_count(text):
    """Estimates the number of words in a string."""
    if not text:
        return 0
    return len(text.split())

# --- Helper function for uploading to blob storage ---
def upload_to_blob(temp_file_path, user_id, document_id, blob_filename, update_callback, group_id=None):
    """Uploads the file to Azure Blob Storage."""

    is_group = group_id is not None
    storage_account_container_name = (
        storage_account_group_documents_container_name
        if is_group else
        storage_account_user_documents_container_name
    )

    try:
        blob_path = f"{group_id}/{blob_filename}" if is_group else f"{user_id}/{blob_filename}"

        blob_service_client = CLIENTS.get("storage_account_office_docs_client")
        if not blob_service_client:
            raise Exception("Blob service client not available or not configured.")

        blob_client = blob_service_client.get_blob_client(
            container=storage_account_container_name,
            blob=blob_path
        )

        metadata = {
            "document_id": str(document_id),
            "group_id": str(group_id) if is_group else None,
            "user_id": str(user_id) if not is_group else None
        }

        metadata = {k: v for k, v in metadata.items() if v is not None}

        update_callback(status=f"Uploading {blob_filename} to Blob Storage...")

        with open(temp_file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True, metadata=metadata)

        print(f"Successfully uploaded {blob_filename} to blob storage at {blob_path}")
        return blob_path

    except Exception as e:
        print(f"Error uploading {blob_filename} to Blob Storage: {str(e)}")
        raise Exception(f"Error uploading {blob_filename} to Blob Storage: {str(e)}")


# --- Helper function to process TXT files ---
def process_txt(document_id, user_id, temp_file_path, original_filename, enable_enhanced_citations, update_callback, group_id=None):
    """Processes plain text files."""
    is_group = group_id is not None

    update_callback(status="Processing TXT file...")
    total_chunks_saved = 0
    target_words_per_chunk = 400

    if enable_enhanced_citations:
        args = {
            "temp_file_path": temp_file_path,
            "user_id": user_id,
            "document_id": document_id,
            "blob_filename": original_filename,
            "update_callback": update_callback
        }

        if is_group:
            args["group_id"] = group_id

        upload_to_blob(**args)

    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        words = content.split()
        num_words = len(words)
        num_chunks_estimated = math.ceil(num_words / target_words_per_chunk)
        update_callback(number_of_pages=num_chunks_estimated) # Use number_of_pages for chunk count

        for i in range(0, num_words, target_words_per_chunk):
            chunk_words = words[i : i + target_words_per_chunk]
            chunk_content = " ".join(chunk_words)
            chunk_index = (i // target_words_per_chunk) + 1

            if chunk_content.strip():
                update_callback(
                    current_file_chunk=chunk_index,
                    status=f"Saving chunk {chunk_index}/{num_chunks_estimated}..."
                )
                args = {
                    "page_text_content": chunk_content,
                    "page_number": chunk_index,
                    "file_name": original_filename,
                    "user_id": user_id,
                    "document_id": document_id
                }

                if is_group:
                    args["group_id"] = group_id

                save_chunks(**args)
                total_chunks_saved += 1

    except Exception as e:
        raise Exception(f"Failed processing TXT file {original_filename}: {e}")

    return total_chunks_saved

# --- Helper function to process HTML files ---
def process_html(document_id, user_id, temp_file_path, original_filename, enable_enhanced_citations, update_callback, group_id=None):
    """Processes HTML files."""
    is_group = group_id is not None

    update_callback(status="Processing HTML file...")
    total_chunks_saved = 0
    target_chunk_words = 1200 # Target size based on requirement
    min_chunk_words = 600 # Minimum size based on requirement

    if enable_enhanced_citations:
        args = {
            "temp_file_path": temp_file_path,
            "user_id": user_id,
            "document_id": document_id,
            "blob_filename": original_filename,
            "update_callback": update_callback
        }

        if is_group:
            args["group_id"] = group_id

        upload_to_blob(**args)

    try:
        # --- CHANGE HERE: Open in binary mode ('rb') ---
        # Let BeautifulSoup handle the decoding based on meta tags or detection
        with open(temp_file_path, 'rb') as f:
            # --- CHANGE HERE: Pass the file object directly to BeautifulSoup ---
            soup = BeautifulSoup(f, 'lxml') # or 'html.parser' if lxml not installed

        # TODO: Advanced Table Handling - (Comment remains valid)
        # ...

        # Now process the soup object as before
        text_content = soup.get_text(separator=" ", strip=True)

        # Remainder of the chunking logic stays the same...
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_chunk_words * 6, # Approximation
            chunk_overlap=target_chunk_words * 0.1 * 6, # 10% overlap approx
            length_function=len,
            is_separator_regex=False,
        )

        initial_chunks = text_splitter.split_text(text_content)

        # Post-processing: Merge small chunks
        final_chunks = []
        buffer_chunk = ""
        for i, chunk in enumerate(initial_chunks):
            current_chunk_text = buffer_chunk + chunk
            current_word_count = estimate_word_count(current_chunk_text)

            if current_word_count >= min_chunk_words or i == len(initial_chunks) - 1:
                if current_chunk_text.strip():
                    final_chunks.append(current_chunk_text)
                buffer_chunk = ""  # Reset buffer
            else:
                               # Chunk is too small, add to buffer and continue to next chunk
                buffer_chunk = current_chunk_text + " "  # Add space between merged chunks

        num_chunks_final = len(final_chunks)
        update_callback(number_of_pages=num_chunks_final) # Use number_of_pages for chunk count

        for idx, chunk_content in enumerate(final_chunks, start=1):
            update_callback(
                current_file_chunk=idx,
                status=f"Saving chunk {idx}/{num_chunks_final}..."
            )
            args = {
                "page_text_content": chunk_content,
                "page_number": idx,
                "file_name": original_filename,
                "user_id": user_id,
                "document_id": document_id
            }

            if is_group:
                args["group_id"] = group_id

            save_chunks(**args)
            total_chunks_saved += 1

    except Exception as e:
        # Catch potential BeautifulSoup errors too
        raise Exception(f"Failed processing HTML file {original_filename}: {e}")

    return total_chunks_saved


# --- Helper function to process Markdown files ---
def process_md(document_id, user_id, temp_file_path, original_filename, enable_enhanced_citations, update_callback, group_id=None):
    """Processes Markdown files and applies small repairs for split markdown tables and code fences."""
    is_group = group_id is not None

    update_callback(status="Processing Markdown file...")
    total_chunks_saved = 0
    target_chunk_words = 1200
    min_chunk_words = 600

    if enable_enhanced_citations:
        args = {
            "temp_file_path": temp_file_path,
            "user_id": user_id,
            "document_id": document_id,
            "blob_filename": original_filename,
            "update_callback": update_callback
        }
        if is_group:
            args["group_id"] = group_id
        upload_to_blob(**args)

    try:
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
        ]

        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, return_each_line=False)
        md_header_splits = md_splitter.split_text(md_content)
        initial_chunks_content = [doc.page_content for doc in md_header_splits]

        # Minimal table header replication:
        # If a chunk starts with a table row (|...), try to find a header in previous text and prepend it.
        repaired_chunks = []
        last_table_header = None
        for chunk in initial_chunks_content:
            trimmed = chunk.lstrip()
            # detect md table header separator present in chunk to capture header from chunk if present
            # pattern: header row followed by line with pipes and --- (e.g. |---| or ---)
            lines = chunk.splitlines()
            if len(lines) >= 2:
                # detect header + separator
                if re.match(r'^\s*\|?.+\|?\s*$', lines[0]) and re.match(r'^\s*\|?\s*:?-{3,}:?\s*(\||$)', lines[1]):
                    # header exists in this chunk
                    last_table_header = lines[0]
                    repaired_chunks.append(chunk)
                    continue
            # if chunk starts with a table row but no header in this chunk, prepend last known header
            if trimmed.startswith("|"):
                if last_table_header:
                    repaired_chunks.append(last_table_header + "\n" + chunk)
                else:
                    repaired_chunks.append(chunk)
            else:
                repaired_chunks.append(chunk)

        # Simple code-fence balancing repair:
        # Ensure that each chunk has matched triple-backtick fences; if odd count, attempt to close/open fences.
        balanced_chunks = []
        fence_open = False
        for chunk in repaired_chunks:
            fence_count = chunk.count("```")
            # If odd, attempt to balance using bookkeeping across chunks
            if fence_count % 2 == 1:
                # If currently not in a fence, we saw an opening fence — mark open and append chunk
                if not fence_open:
                    fence_open = True
                    balanced_chunks.append(chunk)
                else:
                    # we are inside a fence and saw an odd count — close it
                    fence_open = False
                    balanced_chunks.append(chunk + "\n```")
            else:
                balanced_chunks.append(chunk)
        # If we ended still inside a fence, close it on the last chunk
        if fence_open and balanced_chunks:
            balanced_chunks[-1] = balanced_chunks[-1] + "\n```"

        initial_chunks_content = balanced_chunks

        # Merge small chunks (existing logic)
        final_chunks = []
        buffer_chunk = ""
        for i, chunk_text in enumerate(initial_chunks_content):
            current_chunk_text = buffer_chunk + chunk_text
            current_word_count = estimate_word_count(current_chunk_text)
            if current_word_count >= min_chunk_words or i == len(initial_chunks_content) - 1:
                if current_chunk_text.strip():
                    final_chunks.append(current_chunk_text)
                buffer_chunk = ""  # Reset buffer
            else:
                # Chunk is too small, add to buffer and continue to next chunk
                buffer_chunk = current_chunk_text + "\n\n"

        num_chunks_final = len(final_chunks)
        update_callback(number_of_pages=num_chunks_final)

        for idx, chunk_content in enumerate(final_chunks, start=1):
            update_callback(
                current_file_chunk=idx,
                status=f"Saving chunk {idx}/{num_chunks_final}..."
            )
            args = {
                "page_text_content": chunk_content,
                "page_number": idx,
                "file_name": original_filename,
                "user_id": user_id,
                "document_id": document_id
            }
            if is_group:
                args["group_id"] = group_id
            save_chunks(**args)
            total_chunks_saved += 1

    except Exception as e:
        raise Exception(f"Failed processing Markdown file {original_filename}: {e}")

    return total_chunks_saved


# --- Helper function to process JSON files ---
def process_json(document_id, user_id, temp_file_path, original_filename, enable_enhanced_citations, update_callback, group_id=None):
    """Processes JSON files using RecursiveJsonSplitter."""
    is_group = group_id is not None

    update_callback(status="Processing JSON file...")
    total_chunks_saved = 0
    # Reflects character count limit for the splitter
    max_chunk_size_chars = 4000 # As per original requirement

    if enable_enhanced_citations:
        args = {
            "temp_file_path": temp_file_path,
            "user_id": user_id,
            "document_id": document_id,
            "blob_filename": original_filename,
            "update_callback": update_callback
        }

        if is_group:
            args["group_id"] = group_id

        upload_to_blob(**args)


    try:
        # Load the JSON data first to ensure it's valid
        try:
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
             raise Exception(f"Invalid JSON structure in {original_filename}: {e}")
        except Exception as e: # Catch other file reading errors
             raise Exception(f"Error reading JSON file {original_filename}: {e}")

        # Initialize the splitter - convert_lists does NOT go here
        json_splitter = RecursiveJsonSplitter(max_chunk_size=max_chunk_size_chars)

        # Perform the splitting using split_json
        # --- CHANGE HERE: Add convert_lists=True to the splitting method call ---
        # This tells the splitter to handle lists by converting them internally during splitting
        final_json_chunks_structured = json_splitter.split_json(
            json_data=json_data,
            convert_lists=True # Use the feature here as per documentation
        )

        # Convert each structured chunk (which are dicts/lists) back into a JSON string for saving
        # Using ensure_ascii=False is safer for preserving original characters if any non-ASCII exist
        final_chunks_text = [json.dumps(chunk, ensure_ascii=False) for chunk in final_json_chunks_structured]

        initial_chunk_count = len(final_chunks_text)
        update_callback(number_of_pages=initial_chunk_count) # Initial estimate

        for idx, chunk_content in enumerate(final_chunks_text, start=1):
            # Skip potentially empty or trivial chunks (e.g., "{}" or "[]" or just "")
            # Stripping allows checking for empty strings potentially generated
            if not chunk_content or chunk_content == '""' or chunk_content == '{}' or chunk_content == '[]' or not chunk_content.strip('{}[]" '):
                print(f"Skipping empty or trivial JSON chunk {idx}/{initial_chunk_count}")
                continue # Skip saving this chunk

            update_callback(
                current_file_chunk=idx, # Use original index for progress display
                # Keep number_of_pages as initial estimate during saving loop
                status=f"Saving chunk {idx}/{initial_chunk_count}..."
            )
            args = {
                "page_text_content": chunk_content,
                "page_number": total_chunks_saved + 1,
                "file_name": original_filename,
                "user_id": user_id,
                "document_id": document_id
            }

            if is_group:
                args["group_id"] = group_id

            save_chunks(**args)
            total_chunks_saved += 1 # Increment only when a chunk is actually saved

        # Final update with the actual number of chunks saved
        if total_chunks_saved != initial_chunk_count:
            update_callback(number_of_pages=total_chunks_saved)
            print(f"Adjusted final chunk count from {initial_chunk_count} to {total_chunks_saved} after skipping empty chunks.")


    except Exception as e:
        # Catch errors during loading, splitting, or saving
        # Avoid catching the specific JSONDecodeError again if already handled
        if not isinstance(e, json.JSONDecodeError):
             print(f"Error during JSON processing for {original_filename}: {type(e).__name__}: {e}")
        # Re-raise wrapped exception for the main handler
        raise Exception(f"Failed processing JSON file {original_filename}: {e}")

    # Return the count of chunks actually saved
    return total_chunks_saved


# --- Helper function to process a single Tabular sheet (CSV or Excel tab) ---
def process_single_tabular_sheet(
    document_id,
    user_id,
    df,                      # DataFrame to chunk and potentially write to parquet
    file_name,               # effective filename for this sheet (may include sheet name suffix)
    update_callback,
    sheet_name="",
    rows=None,
    cols=None,
    group_id=None,
    original_blob_path=None,
    sheets_list: Optional[List[Dict[str, Any]]] = None
):
    """Chunks a pandas DataFrame from a CSV or Excel sheet and optionally writes & uploads a parquet for that sheet.

    Appends a sheet descriptor (sheet_name, parquet_blob_path, rows, cols) to sheets_list if provided.
    Returns the number of chunks saved.
    """
    is_group = group_id is not None

    total_chunks_saved = 0
    target_chunk_size_chars = 800 # Requirement: "800 size chunk" (assuming characters)

    # Ensure df is a pandas DataFrame
    if df is None or not hasattr(df, "columns"):
        raise ValueError("process_single_tabular_sheet requires a pandas DataFrame `df`")

    # Prepare stable base_name early to avoid NameError later
    base_name = os.path.splitext(file_name)[0]

    # Create header CSV string
    header = df.columns.tolist()
    header_string = ",".join(map(str, header)) + "\n"  # CSV header line

    # Prepare rows as strings (CSV-like)
    rows_as_strings = []
    for _, row in df.iterrows():
        row_string = ",".join(map(lambda x: str(x) if (x is not None and (not (isinstance(x, float) and str(x) == 'nan'))) else "", row.tolist())) + "\n"
        rows_as_strings.append(row_string)

    # Chunk rows based on character count
    final_chunks_content = []
    current_chunk_rows = []
    current_chunk_char_count = 0

    for row_str in rows_as_strings:
        row_len = len(row_str)
        if current_chunk_char_count + row_len > target_chunk_size_chars and current_chunk_rows:
            final_chunks_content.append("".join(current_chunk_rows))
            current_chunk_rows = [row_str]
            current_chunk_char_count = row_len
        else:
            current_chunk_rows.append(row_str)
            current_chunk_char_count += row_len

    if current_chunk_rows:
        final_chunks_content.append("".join(current_chunk_rows))

    num_chunks_final = len(final_chunks_content)
    # Inform caller of expected chunk count
    try:
        update_callback(number_of_pages=num_chunks_final)
    except Exception:
        pass

    # NEW: add an explicit sheet tag line so save_chunks() can extract sheet_name reliably
    sheet_tag = f"[Sheet: {sheet_name or base_name}]\n"

    # Save chunks, prepending the header to each
    for idx, chunk_rows_content in enumerate(final_chunks_content, start=1):
        # prepend sheet tag + header so downstream regex in save_chunks can capture sheet name
        chunk_with_header = sheet_tag + header_string + chunk_rows_content
        try:
            update_callback(
                current_file_chunk=idx,
                status=f"Saving chunk {idx}/{num_chunks_final} from {file_name}..."
            )
        except Exception:
            pass

        args = {
            "page_text_content": chunk_with_header,
            "page_number": idx,
            "file_name": file_name,
            "user_id": user_id,
            "document_id": document_id
        }
        if is_group:
            args["group_id"] = group_id

        save_chunks(**args)
        total_chunks_saved += 1

    # Attempt to write a Parquet for this sheet, upload it, and register in sheets_list
    parquet_temp_path = None
    parquet_blob_path = None
    try:
        # Create a unique temp file for parquet
        fd, parquet_temp_path = tempfile.mkstemp(prefix="tabular_", suffix=".parquet")
        os.close(fd)
        try:
            # Try pyarrow first, fallback to fastparquet if available
            try:
                df.to_parquet(parquet_temp_path, index=False, engine="pyarrow")
            except Exception:
                df.to_parquet(parquet_temp_path, index=False, engine="fastparquet")
        except Exception as e:
            # If parquet write fails, skip parquet creation but do not fail ingestion
            try:
                log_exception(f"Parquet write failed for {document_id}/{file_name}: {e}")
            except Exception:
                pass
            parquet_temp_path = None

        if parquet_temp_path and os.path.exists(parquet_temp_path):
            parquet_blob_filename = f"{base_name}.parquet"
            try:
                parquet_blob_path = upload_to_blob(parquet_temp_path, user_id, document_id, 
                    parquet_blob_filename, update_callback, group_id)
                print(f"[Tabular] Parquet uploaded: {parquet_blob_path} (doc={document_id}, sheet={sheet_name or base_name})")
            except Exception as e:
                try:
                    log_exception(f"Parquet upload failed for {document_id}/{file_name}: {e}")
                except Exception:
                    pass
                print(f"[Tabular] Parquet upload FAILED (doc={document_id}, sheet={sheet_name or base_name})")
        else:
            print(f"[Tabular] No parquet generated; skipping upload "
                f"(doc={document_id}, sheet={sheet_name or base_name})")
            
        # If caller supplied a sheets_list append a sheet descriptor
        if sheets_list is not None:
            sheet_desc = {"sheet_name": sheet_name or base_name, "parquet_blob_path": parquet_blob_path or ""}
            if rows is not None:
                try:
                    sheet_desc["rows"] = int(rows)
                except Exception:
                    pass
            if cols is not None:
                try:
                    sheet_desc["cols"] = int(cols)
                except Exception:
                    pass
            sheets_list.append(sheet_desc)

    finally:
        # best-effort cleanup of temp parquet file
        try:
            if parquet_temp_path and os.path.exists(parquet_temp_path):
                os.remove(parquet_temp_path)
        except Exception:
            try:
                log_exception(f"Failed to remove temp parquet {parquet_temp_path} for document {document_id}")
            except Exception:
                pass

    return total_chunks_saved


def process_tabular(
    document_id,
    user_id,
    temp_file_path,
    original_filename,
    update_callback,
    group_id
):
    """Processes CSV, XLSX, or XLS files using pandas. Produces per-sheet parquet files and writes a deterministic manifest."""
    is_group = group_id is not None

    update_callback(status=f"Processing Tabular file ({original_filename})...")
    total_chunks_saved = 0
    sheets: List[Dict[str, Any]] = []  # collect sheet descriptors for manifest

    # Upload original file if enhanced citations enabled
    settings = get_settings()
    enable_enhanced_citations = settings.get("enable_enhanced_citations", False)
    if enable_enhanced_citations:
        args = {
            "temp_file_path": temp_file_path,
            "user_id": user_id,
            "document_id": document_id,
            "blob_filename": original_filename,
            "update_callback": update_callback
        }
        if is_group:
            args["group_id"] = group_id
        try:
            upload_to_blob(**args)
        except Exception:
            try:
                log_exception(f"Failed to upload original tabular file for doc {document_id}")
            except Exception:
                pass

    try:
        file_ext = os.path.splitext(original_filename)[1].lower()
        if file_ext == '.csv':
            # Read CSV into DataFrame
            df = pd.read_csv(temp_file_path, keep_default_na=False, dtype=str)
            chunks_from_sheet = process_single_tabular_sheet(
                document_id=document_id,
                user_id=user_id,
                df=df,
                file_name=original_filename,
                update_callback=update_callback,
                sheet_name="",  # CSV single sheet
                group_id=group_id,
                original_blob_path=original_filename,
                sheets_list=sheets
            )
            total_chunks_saved = chunks_from_sheet

        elif file_ext in ('.xlsx', '.xls'):
            # Excel: multiple sheets
            excel_file = pd.ExcelFile(temp_file_path, engine='openpyxl' if file_ext == '.xlsx' else 'xlrd')
            sheet_names = excel_file.sheet_names
            base_name, ext = os.path.splitext(original_filename)
            accumulated_total_chunks = 0
            for sheet_name in sheet_names:
                update_callback(status=f"Processing sheet '{sheet_name}'...")
                df = excel_file.parse(sheet_name, keep_default_na=False, dtype=str)
                effective_filename = f"{base_name}-{sheet_name}{ext}" if len(sheet_names) > 1 else original_filename
                chunks_from_sheet = process_single_tabular_sheet(
                    document_id=document_id,
                    user_id=user_id,
                    df=df,
                    file_name=effective_filename,
                    update_callback=update_callback,
                    sheet_name=sheet_name,
                    rows=len(df) if hasattr(df, "__len__") else None,
                    cols=len(df.columns) if hasattr(df, "columns") else None,
                    group_id=group_id,
                    original_blob_path=original_filename,
                    sheets_list=sheets
                )
                accumulated_total_chunks += chunks_from_sheet
            total_chunks_saved = accumulated_total_chunks

        else:
            raise Exception(f"Unsupported tabular file extension: {file_ext}")

    except pd.errors.EmptyDataError:
        update_callback(status=f"Warning: File/sheet is empty - {original_filename}", number_of_pages=0)
    except Exception as e:
        raise Exception(f"Failed processing Tabular file {original_filename}: {e}")

    # After all per-sheet parquet uploads complete, write deterministic manifest:
    try:
        if sheets:
            source = original_filename
            ok = write_manifest_for_document(
            document_id=document_id,
            user_id=user_id,
            group_id=group_id,
            sheets=sheets,
            source_blob=source
            )
            print(f"[Tabular] Manifest {'OK' if ok else 'FAILED'} "
                f"for doc={document_id}, sheets={len(sheets)}")
    except Exception:
        try:
            log_exception(f"Failed to write tabular manifest for document {document_id}")
        except Exception:
            pass

    return total_chunks_saved

# === NEW: Orchestration for queued uploads (used by routes) ===
def process_document_upload_background(
    document_id: str,
    user_id: str,
    temp_file_path: str,
    original_filename: str,
    group_id: Optional[str] = None,
):
    """
    Orchestrates a single uploaded file:
      - Detect type
      - (Optional) upload original (enhanced citations)
      - Extract text (Azure DI for PDF/DOC/DOCX)
      - Chunk & save to Azure Cognitive Search
      - Update Cosmos document status/percentage along the way
    """
    is_group = group_id is not None
    file_ext = (os.path.splitext(original_filename)[1] or "").lower()
    settings = get_settings()
    enable_enhanced_citations = settings.get("enable_enhanced_citations", False)

    def update_callback(**kwargs):
        args = {"document_id": document_id, "user_id": user_id}
        if is_group:
            args["group_id"] = group_id
        args.update(kwargs)
        update_document(**args)

    try:
        update_callback(status=f"Receiving {original_filename}…", percentage_complete=0)

        # 1) Optionally upload original
        if enable_enhanced_citations:
            try:
                upload_to_blob(
                    temp_file_path=temp_file_path,
                    user_id=user_id,
                    document_id=document_id,
                    blob_filename=original_filename,
                    update_callback=update_callback,
                    group_id=group_id,
                )
            except Exception as e:
                update_callback(status=f"Warning: original upload failed → {e}")

        total_chunks_saved = 0

        # --- Video ---
        if file_ext in {".mp4", ".mov", ".m4v", ".avi", ".wmv", ".mkv"}:
            update_callback(status="VIDEO: starting indexing…", percentage_complete=1)
            total_chunks_saved = process_video_document(
                document_id=document_id,
                user_id=user_id,
                temp_file_path=temp_file_path,
                original_filename=original_filename,
                update_callback=update_callback,
                group_id=group_id,
            )

        # --- Tabular ---
        elif file_ext in {".csv", ".xlsx", ".xls"}:
            total_chunks_saved = process_tabular(
                document_id=document_id,
                user_id=user_id,
                temp_file_path=temp_file_path,
                original_filename=original_filename,
                update_callback=update_callback,
                group_id=group_id,
            )

        # --- JSON ---
        elif file_ext == ".json":
            total_chunks_saved = process_json(
                document_id=document_id,
                user_id=user_id,
                temp_file_path=temp_file_path,
                original_filename=original_filename,
                enable_enhanced_citations=enable_enhanced_citations,
                update_callback=update_callback,
                group_id=group_id,
            )

        # --- HTML ---
        elif file_ext in {".html", ".htm"}:
            total_chunks_saved = process_html(
                document_id=document_id,
                user_id=user_id,
                temp_file_path=temp_file_path,
                original_filename=original_filename,
                enable_enhanced_citations=enable_enhanced_citations,
                update_callback=update_callback,
                group_id=group_id,
            )

        # --- Markdown ---
        elif file_ext in {".md", ".markdown"}:
            total_chunks_saved = process_md(
                document_id=document_id,
                user_id=user_id,
                temp_file_path=temp_file_path,
                original_filename=original_filename,
                enable_enhanced_citations=enable_enhanced_citations,
                update_callback=update_callback,
                group_id=group_id,
            )

        # --- Plain text ---
        elif file_ext in {".txt", ".log"}:
            total_chunks_saved = process_txt(
                document_id=document_id,
                user_id=user_id,
                temp_file_path=temp_file_path,
                original_filename=original_filename,
                enable_enhanced_citations=enable_enhanced_citations,
                update_callback=update_callback,
                group_id=group_id,
            )

        # --- PDF / Word via Azure Document Intelligence ---
        elif file_ext in {".pdf", ".docx", ".doc"}:
            update_callback(status="Sending to Document Intelligence…", percentage_complete=5)

            input_paths = [temp_file_path]
            if file_ext == ".pdf":
                page_count = get_pdf_page_count(temp_file_path) or 0
                if page_count > 500:
                    update_callback(status=f"Large PDF detected ({page_count} pages). Chunking…")
                    input_paths = chunk_pdf(temp_file_path, max_pages=500)

            total_pages = 0
            saved_so_far = 0
            for sub_ix, sub_path in enumerate(input_paths, start=1):
                try:
                    pages = extract_content_with_azure_di(sub_path)  # [{page_number, content}]
                except Exception as e:
                    update_callback(status=f"Azure DI failed on part {sub_ix}: {e}")
                    continue

                # Add metadata from file properties for the first sub-part
                if sub_ix == 1:
                    try:
                        if file_ext == ".pdf":
                            title, author, _, _ = extract_pdf_metadata(temp_file_path)
                        else:
                            title, author = extract_docx_metadata(temp_file_path)
                        update_callback(title=title or None, authors=[author] if author else None)
                    except Exception:
                        pass

                # Re-chunk Word pages to WORD_CHUNK_SIZE if needed
                if file_ext in {".doc", ".docx"}:
                    pages = chunk_word_file_into_pages(pages)

                total_pages += len(pages)
                update_callback(number_of_pages=total_pages, status="Saving pages…")

                for p in pages:
                    saved_so_far += 1
                    update_callback(current_file_chunk=saved_so_far, status=f"Saving page {saved_so_far}/{total_pages}…")
                    save_chunks(
                        page_text_content=p.get("content", ""),
                        page_number=saved_so_far,
                        file_name=original_filename,
                        user_id=user_id,
                        document_id=document_id,
                        group_id=group_id,
                    )

            total_chunks_saved = saved_so_far

        else:
            update_callback(status=f"Unsupported file type: {file_ext}")
            raise ValueError(f"Unsupported file type: {file_ext}")

        update_callback(
            num_chunks=total_chunks_saved,
            current_file_chunk=total_chunks_saved,
            number_of_pages=total_chunks_saved,
            status="Processing complete",
            percentage_complete=100,
        )

    except Exception as e:
        update_callback(status=f"Processing failed: {e}")
        raise
    finally:
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass


def upgrade_legacy_documents(user_id: str, group_id: Optional[str] = None) -> int:
    """
    One-time upgrader: add percentage/status fields for old docs that predate progress tracking.
    Returns the count of upgraded docs.
    """
    is_group = group_id is not None
    container = cosmos_group_documents_container if is_group else cosmos_user_documents_container

    query = (
        "SELECT * FROM c WHERE c.group_id = @gid AND NOT IS_DEFINED(c.percentage_complete)"
        if is_group else
        "SELECT * FROM c WHERE c.user_id = @uid AND NOT IS_DEFINED(c.percentage_complete)"
    )
    params = [{"name": "@gid", "value": group_id}] if is_group else [{"name": "@uid", "value": user_id}]

    upgraded = 0
    try:
        items = list(container.query_items(query=query, parameters=params, enable_cross_partition_query=True))
        for d in items:
            # set baseline tracking fields if missing
            if not d.get("status"):
                d["status"] = "Processing complete" if d.get("num_chunks", 0) > 0 else "Queued"
            d["percentage_complete"] = 100 if str(d.get("status", "")).lower().startswith("processing complete") else d.get("percentage_complete", 0)
            d["last_updated"] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            container.upsert_item(d)
            upgraded += 1
        return upgraded
    except Exception:
        return upgraded
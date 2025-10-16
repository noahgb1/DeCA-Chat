# functions_content.py

from config import *
from functions_settings import *
from functions_logging import *
from functions_authentication import get_bearer_token_provider

# --- New / Fixed Imports (kept minimal and surgical) ---
import re
import json
import time
import random
import pandas as pd
import fitz  # PyMuPDF for PDF metadata helpers
import docx  # python-docx for DOCX metadata helpers
from typing import Any, Dict, List, Tuple, Optional

# Azure + OpenAI SDK exceptions/clients used below
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import HttpResponseError
from openai import AzureOpenAI, RateLimitError


def extract_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_content_with_azure_di(file_path):
    """
    Extracts text page-by-page using Azure Document Intelligence "prebuilt-read"
    and returns a list of dicts, each containing page_number and content.
    """
    try:
        document_intelligence_client = CLIENTS['document_intelligence_client']  # Ensure CLIENTS is populated
        with open(file_path, "rb") as f:
            poller = document_intelligence_client.begin_analyze_document(
                model_id="prebuilt-read",
                document=f
            )

        max_wait_time = 600
        start_time = time.time()

        while True:
            status = poller.status()
            if status == "succeeded":
                break
            if status in ["failed", "canceled"]:
                # Attempt to get result even on failure for potential error details
                try:
                    result = poller.result()
                    # Optionally add failed result details to the exception message
                    error_details = f"Failed DI result details: {result}"
                except Exception as res_ex:
                    error_details = f"Could not get result details after failure: {res_ex}"
                raise Exception(f"Document analysis {status} for document. {error_details}")
            if time.time() - start_time > max_wait_time:
                raise TimeoutError("Document analysis took too long.")

            sleep_duration = 10  # Or adjust based on expected processing time
            time.sleep(sleep_duration)

        result = poller.result()

        pages_data = []

        if result.pages:
            for page in result.pages:
                page_number = page.page_number
                page_text = ""  # Initialize page_text

                # --- METHOD 1: Preferred - Use spans and result.content ---
                if getattr(page, "spans", None) and getattr(result, "content", None):
                    try:
                        page_content_parts = []
                        for span in page.spans:
                            start = span.offset
                            end = start + span.length
                            page_content_parts.append(result.content[start:end])
                        page_text = "".join(page_content_parts)
                    except Exception:
                        page_text = ""  # Reset on error

                # --- METHOD 2: Fallback - Use lines if spans failed or weren't available ---
                if not page_text and getattr(page, "lines", None):
                    try:
                        page_text = "\n".join(line.content for line in page.lines)
                    except Exception:
                        page_text = ""  # Reset on error

                # --- METHOD 3: Last Resort Fallback - Use words (less accurate formatting) ---
                if not page_text and getattr(page, "words", None):
                    try:
                        page_text = " ".join(word.content for word in page.words)
                    except Exception:
                        page_text = ""  # Reset on error

                pages_data.append({
                    "page_number": page_number,
                    "content": (page_text or "").strip()
                })

        # --- Fallback if NO pages were found at all, but top-level content exists ---
        elif getattr(result, "content", None):
            pages_data.append({
                "page_number": 1,
                "content": result.content.strip()
            })
        # else: pages_data remains empty

        return pages_data

    except HttpResponseError as e:
        raise e
    except TimeoutError as e:
        raise e
    except Exception as e:
        raise e


def extract_table_file(file_path, file_ext):
    """
    Extract tabular content from CSV/XLS/XLSX files as HTML.

    - CSV: single DataFrame -> HTML table with a <caption> matching file stem,
      and a plain-text header line ("FILE: <name>") before the table.
    - Excel: concatenate one HTML table per sheet, each with a <caption> = sheet name,
      and a plain-text header line ("SHEET: <name>") before each table.
    """
    # Local imports keep this a surgical change without touching module imports.
    import os
    from html import escape

    try:
        ext = (file_ext or "").lower()

        if ext == ".csv":
            # Preserve existing CSV behavior but add caption + plain-text header for parity.
            df = pd.read_csv(file_path)
            table_html = df.to_html(index=False, classes="table table-striped table-bordered")

            # Compute stem regardless of whether we find <thead>
            stem = os.path.splitext(os.path.basename(file_path))[0]

            # Insert an HTML <caption> just before the first <thead>
            insert_at = table_html.find("<thead>")
            if insert_at != -1:
                caption = f'  <caption>{escape(stem)}</caption>\n '
                table_html = table_html[:insert_at] + caption + table_html[insert_at:]

            # Add a plain-text section header BEFORE the table for LLM context
            header = f'<p><strong>FILE: {escape(stem)}</strong></p>\n'
            return header + table_html

        elif ext in [".xls", ".xlsx"]:
            xls = pd.ExcelFile(file_path)
            parts = []

            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)
                html_table = df.to_html(index=False, classes="table table-striped table-bordered")

                # Insert <caption> before <thead>
                insert_at = html_table.find("<thead>")
                if insert_at != -1:
                    caption = f'  <caption>{escape(sheet_name)}</caption>\n '
                    html_table = html_table[:insert_at] + caption + html_table[insert_at:]

                # Add a plain-text section header BEFORE the table
                header = f'<p><strong>SHEET: {escape(sheet_name)}</strong></p>\n'
                parts.append(header + html_table)

            # Join all sheet sections
            return "".join(parts)

        else:
            raise ValueError("Unsupported file extension for table extraction.")

    except Exception:
        # Let upstream handlers/logging capture details
        raise


def extract_pdf_metadata(pdf_path):
    """
    Returns a tuple (title, author, subject, keywords) from the given PDF, using PyMuPDF.
    """
    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata
            pdf_title = meta.get("title", "")
            pdf_author = meta.get("author", "")
            pdf_subject = meta.get("subject", "")
            pdf_keywords = meta.get("keywords", "")

            return pdf_title, pdf_author, pdf_subject, pdf_keywords

    except Exception as e:
        print(f"Error extracting PDF metadata: {e}")
        return "", "", "", ""


def extract_docx_metadata(docx_path):
    """
    Returns a tuple (title, author) from the given DOCX, using python-docx.
    """
    try:
        doc = docx.Document(docx_path)
        core_props = doc.core_properties
        doc_title = core_props.title or ''
        doc_author = core_props.author or ''
        return doc_title, doc_author
    except Exception as e:
        print(f"Error extracting DOCX metadata: {e}")
        return '', ''


def parse_authors(author_input):
    """
    Converts any input (None, string, list, comma-delimited, etc.)
    into a list of author strings.
    """
    if not author_input:
        # Covers None or empty string
        return []

    # If it's already a list, just return it (with stripping)
    if isinstance(author_input, list):
        return [a.strip() for a in author_input if a.strip()]

    # Otherwise, assume it's a string and parse by common delimiters (comma, semicolon)
    if isinstance(author_input, str):
        # e.g. "John Doe, Jane Smith; Bob Brown"
        authors = re.split(r'[;,]', author_input)
        authors = [a.strip() for a in authors if a.strip()]
        return authors

    # If it's some other unexpected data type, fallback to empty
    return []


def chunk_text(text, chunk_size=2000, overlap=200):
    try:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    except Exception as e:
        # Log the exception or handle it as needed
        print(f"Error in chunk_text: {e}")
        raise e  # Re-raise the exception to propagate it


def chunk_word_file_into_pages(di_pages):
    """
    Chunks the content extracted from a Word document by Azure DI into smaller
    chunks based on a target word count.

    Args:
        di_pages (list): A list of dictionaries, where each dictionary represents
                         a page extracted by Azure DI and contains at least a
                         'page_number' and 'content' key.

    Returns:
        list: A new list of dictionaries, where each dictionary represents a
              smaller chunk with 'page_number' (representing the chunk sequence)
              and 'content' (the chunked text).
    """
    new_pages = []
    current_chunk_content = []
    current_word_count = 0
    new_page_number = 1  # This will represent the chunk number

    # WORD_CHUNK_SIZE may come from config; provide a safe fallback
    try:
        target_size = int(globals().get("WORD_CHUNK_SIZE", 800))
    except Exception:
        target_size = 800

    for page in di_pages:
        page_content = page.get("content", "")
        # Split content into words (handling various whitespace)
        words = re.findall(r'\S+', page_content)

        for word in words:
            current_chunk_content.append(word)
            current_word_count += 1

            # If the chunk reaches the desired size, finalize it
            if current_word_count >= target_size:
                chunk_txt = " ".join(current_chunk_content)
                new_pages.append({
                    "page_number": new_page_number,
                    "content": chunk_txt
                })
                # Reset for the next chunk
                current_chunk_content = []
                current_word_count = 0
                new_page_number += 1

    # Add any remaining words as the last chunk, if any exist
    if current_chunk_content:
        chunk_txt = " ".join(current_chunk_content)
        new_pages.append({
            "page_number": new_page_number,
            "content": chunk_txt
        })

    return new_pages


def generate_embedding(
    text,
    max_retries=5,
    initial_delay=1.0,
    delay_multiplier=2.0
):
    settings = get_settings()

    retries = 0
    current_delay = initial_delay

    enable_embedding_apim = settings.get('enable_embedding_apim', False)

    if enable_embedding_apim:
        embedding_model = settings.get('azure_apim_embedding_deployment')
        embedding_client = AzureOpenAI(
            api_version=settings.get('azure_apim_embedding_api_version'),
            azure_endpoint=settings.get('azure_apim_embedding_endpoint'),
            api_key=settings.get('azure_apim_embedding_subscription_key'))
    else:
        if (settings.get('azure_openai_embedding_authentication_type') == 'managed_identity'):
            token_provider = get_bearer_token_provider(DefaultAzureCredential(), cognitive_services_scope)

            embedding_client = AzureOpenAI(
                api_version=settings.get('azure_openai_embedding_api_version'),
                azure_endpoint=settings.get('azure_openai_embedding_endpoint'),
                azure_ad_token_provider=token_provider
            )

            embedding_model_obj = settings.get('embedding_model', {})
            if embedding_model_obj and embedding_model_obj.get('selected'):
                selected_embedding_model = embedding_model_obj['selected'][0]
                embedding_model = selected_embedding_model['deploymentName']
        else:
            embedding_client = AzureOpenAI(
                api_version=settings.get('azure_openai_embedding_api_version'),
                azure_endpoint=settings.get('azure_openai_embedding_endpoint'),
                api_key=settings.get('azure_openai_embedding_key')
            )

            embedding_model_obj = settings.get('embedding_model', {})
            if embedding_model_obj and embedding_model_obj.get('selected'):
                selected_embedding_model = embedding_model_obj['selected'][0]
                embedding_model = selected_embedding_model['deploymentName']

    while True:
        random_delay = random.uniform(0.5, 2.0)
        time.sleep(random_delay)

        try:
            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=text
            )

            embedding = response.data[0].embedding
            return embedding

        except RateLimitError:
            retries += 1
            if retries > max_retries:
                return None

            wait_time = current_delay * random.uniform(1.0, 1.5)
            time.sleep(wait_time)
            current_delay *= delay_multiplier

        except Exception:
            raise


def get_all_chunks(document_id, user_id):
    try:
        search_client_user = CLIENTS["search_client_user"]
        results = search_client_user.search(
            search_text="*",
            filter=f"document_id eq '{document_id}' and user_id eq '{user_id}'",
            select=["id", "chunk_text", "chunk_id", "file_name", "user_id", "version", "chunk_sequence", "upload_date"]
        )
        return results
    except Exception as e:
        print(f"Error retrieving chunks for document {document_id}: {e}")
        raise


def update_chunk_metadata(chunk_id, user_id, document_id, **kwargs):
    try:
        search_client_user = CLIENTS["search_client_user"]
        chunk_item = search_client_user.get_document(chunk_id)

        if not chunk_item:
            raise Exception("Chunk not found")

        if chunk_item['user_id'] != user_id:
            raise Exception("Unauthorized access to chunk")

        if chunk_item['document_id'] != document_id:
            raise Exception("Chunk does not belong to document")

        if 'chunk_keywords' in kwargs:
            chunk_item['chunk_keywords'] = kwargs['chunk_keywords']

        if 'chunk_summary' in kwargs:
            chunk_item['chunk_summary'] = kwargs['chunk_summary']

        if 'author' in kwargs:
            chunk_item['author'] = kwargs['author']

        if 'title' in kwargs:
            chunk_item['title'] = kwargs['title']

        if 'document_classification' in kwargs:
            chunk_item['document_classification'] = kwargs['document_classification']

        search_client_user.upload_documents(documents=[chunk_item])
    except Exception as e:
        print(f"Error updating chunk metadata for chunk {chunk_id}: {e}")
        raise


# -----------------------------
# Minimal text splitter helpers
# -----------------------------

class _DocChunk:
    """Lightweight container to mirror expected interface with `.page_content`."""
    def __init__(self, page_content: str):
        self.page_content = page_content


class RecursiveCharacterTextSplitter:
    """
    Minimal, dependency-free splitter used by markdown/HTML processing.
    Splits by characters, attempting to honor the chunk_size and overlap.
    """
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200, length_function=len, is_separator_regex: bool = False):
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.length_function = length_function or len
        self.is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        chunks: List[str] = []
        n = self.length_function(text)
        if n <= self.chunk_size:
            return [text]

        start = 0
        step = self.chunk_size - self.chunk_overlap if self.chunk_size > self.chunk_overlap else self.chunk_size
        while start < n:
            end = min(n, start + self.chunk_size)
            chunks.append(text[start:end])
            if end == n:
                break
            start = start + step
        return chunks


class MarkdownHeaderTextSplitter:
    """
    Minimal header-based markdown splitter.
    `headers_to_split_on` is a list of tuples like [("#", "Header 1"), ("##", "Header 2"), ...]
    """
    def __init__(self, headers_to_split_on: List[Tuple[str, str]], return_each_line: bool = False):
        self.headers = [h[0] for h in headers_to_split_on] if headers_to_split_on else ["#","##","###","####","#####"]
        self.return_each_line = return_each_line

    def split_text(self, md_text: str) -> List[_DocChunk]:
        lines = md_text.splitlines()
        chunks: List[str] = []
        current: List[str] = []
        header_pattern = re.compile(rf"^({'|'.join([re.escape(h) for h in self.headers])})\s+")
        for line in lines:
            if header_pattern.match(line) and current:
                chunks.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append("\n".join(current).strip())
        return [_DocChunk(c) for c in chunks if c]


class RecursiveJsonSplitter:
    """
    Minimal JSON splitter used by process_json. It walks the JSON structure and
    emits chunks whose serialized length is below `max_chunk_size`. Nested objects
    are split into separate chunks when needed.

    NOTE: This is intentionally simple and conservative to avoid changing upstream logic.
    """
    def __init__(self, max_chunk_size: int = 4000):
        self.max_chunk = max(512, int(max_chunk_size))

    def _size(self, obj: Any) -> int:
        try:
            return len(json.dumps(obj, ensure_ascii=False))
        except Exception:
            return self.max_chunk + 1  # force split on serialization errors

    def _split_obj(self, obj: Any, out: List[Any]):
        # Base case: small enough, emit as-is
        if self._size(obj) <= self.max_chunk:
            out.append(obj)
            return

        # Split lists by items
        if isinstance(obj, list):
            buf: List[Any] = []
            for item in obj:
                if self._size(item) > self.max_chunk:
                    # recurse on big item
                    self._split_obj(item, out)
                else:
                    # try adding to buffer
                    tentative = buf + [item]
                    if self._size(tentative) <= self.max_chunk:
                        buf = tentative
                    else:
                        if buf:
                            out.append(buf)
                        buf = [item]
            if buf:
                out.append(buf)
            return

        # Split dicts by keys
        if isinstance(obj, dict):
            buf: Dict[str, Any] = {}
            for k, v in obj.items():
                if self._size({k: v}) > self.max_chunk:
                    # value too large alone; split value recursively
                    self._split_obj(v, out)
                else:
                    tentative = dict(buf)
                    tentative[k] = v
                    if self._size(tentative) <= self.max_chunk:
                        buf = tentative
                    else:
                        if buf:
                            out.append(buf)
                        buf = {k: v}
            if buf:
                out.append(buf)
            return

        # Fallback: treat as string
        s = str(obj)
        step = max(256, self.max_chunk // 2)
        for i in range(0, len(s), step):
            out.append(s[i:i + step])

    def split_json(self, json_data: Any, convert_lists: bool = True) -> List[Any]:
        """
        Returns a list of JSON-serializable pieces whose serialized size is below `max_chunk_size`.
        The `convert_lists` flag is accepted for compatibility; behavior is handled in _split_obj.
        """
        out: List[Any] = []
        self._split_obj(json_data, out)
        return out
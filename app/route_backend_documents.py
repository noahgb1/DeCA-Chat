# route_backend_documents.py

from config import *
from functions_authentication import *
from functions_documents import *
from functions_settings import *
import os

# Added explicit imports used in this file
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
import uuid
import tempfile
import traceback

# Ensure Cosmos and Azure Search exceptions are available
try:
    from azure.cosmos.exceptions import CosmosResourceNotFoundError
except Exception:
    CosmosResourceNotFoundError = Exception

try:
    from azure.core.exceptions import ResourceNotFoundError
except Exception:
    ResourceNotFoundError = Exception


def register_route_backend_documents(app):
    @app.route('/api/get_file_content', methods=['POST'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def get_file_content():
        data = request.get_json()
        user_id = get_current_user_id()
        conversation_id = data.get('conversation_id')
        file_id = data.get('file_id')

        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        if not conversation_id or not file_id:
            return jsonify({'error': 'Missing conversation_id or id'}), 400

        try:
            _ = cosmos_conversations_container.read_item(
                item=conversation_id,
                partition_key=conversation_id
            )
        except CosmosResourceNotFoundError:
            return jsonify({'error': 'Conversation not found'}), 404
        except Exception as e:
            return jsonify({'error': f'Error reading conversation: {str(e)}'}), 500
        
        add_file_task_to_file_processing_log(document_id=file_id, user_id=user_id, content="Conversation exists, retrieving file content")
        try:
            query_str = """
                SELECT * FROM c
                WHERE c.conversation_id = @conversation_id
                AND c.id = @file_id
            """
            items = list(cosmos_messages_container.query_items(
                query=query_str,
                parameters=[
                    {'name': '@conversation_id', 'value': conversation_id},
                    {'name': '@file_id', 'value': file_id}
                ],
                partition_key=conversation_id
            ))

            if not items:
                add_file_task_to_file_processing_log(document_id=file_id, user_id=user_id, content="File not found in conversation")
                return jsonify({'error': 'File not found in conversation'}), 404

            add_file_task_to_file_processing_log(document_id=file_id, user_id=user_id, content="File found, processing content: " + str(items))
            items_sorted = sorted(items, key=lambda x: x.get('chunk_index', 0))

            filename = items_sorted[0].get('filename', 'Untitled')
            is_table = items_sorted[0].get('is_table', False)

            add_file_task_to_file_processing_log(document_id=file_id, user_id=user_id, content="Combining file content from chunks, filename: " + filename + ", is_table: " + str(is_table))
            combined_parts = []
            for it in items_sorted:
                fc = it.get('file_content', '')

                if isinstance(fc, list):
                    # If file_content is a list of dicts, join their 'content' fields
                    text_chunks = []
                    for chunk in fc:
                        text_chunks.append(chunk.get('content', ''))
                    combined_parts.append("\n".join(text_chunks))
                elif isinstance(fc, str):
                    # If it's already a string, just append
                    combined_parts.append(fc)
                else:
                    # If it's neither a list nor a string, handle as needed (e.g., skip or log)
                    pass

            combined_content = "\n".join(combined_parts)

            if not combined_content:
                add_file_task_to_file_processing_log(document_id=file_id, user_id=user_id, content="Combined file content is empty")
                return jsonify({'error': 'File content not found'}), 404

            return jsonify({
                'file_content': combined_content,
                'filename': filename,
                'is_table': is_table
            }), 200

        except Exception as e:
            add_file_task_to_file_processing_log(document_id=file_id, user_id=user_id, content="Error retrieving file content: " + str(e))
            return jsonify({'error': f'Error retrieving file content: {str(e)}'}), 500
    
    @app.route('/api/documents/upload', methods=['POST'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def api_user_upload_document():
        """
        Accepts one or more files via multipart/form-data with key 'file'.
        Returns 200 on full success, 207 if partial, 400 if all failed.
        Adds detailed diagnostics to logs to explain 400 causes.
        """
        try:
            user_id = get_current_user_id()
            if not user_id:
                return jsonify({'error': 'User not authenticated'}), 401

            # Diagnostics for ambiguous 400s
            current_app.logger.info(
                "[Upload] user=%s CT=%s CL=%s form_keys=%s files_keys=%s",
                user_id,
                request.content_type,
                request.headers.get("Content-Length"),
                list(request.form.keys()),
                list(request.files.keys()),
            )

            if 'file' not in request.files:
                current_app.logger.warning("[Upload] Missing 'file' in request.files")
                return jsonify({'error': "No file part in the request (expecting key 'file' in multipart/form-data)."}), 400

            files = request.files.getlist('file')  # Handle multiple files potentially
            if not files or all(not f.filename for f in files):
                current_app.logger.warning("[Upload] Empty selection or filenames")
                return jsonify({'error': 'No file selected or files have no name'}), 400

            processed_docs = []
            upload_errors = []

            for file in files:
                if not file.filename:
                    upload_errors.append("Skipped a file with no name.")
                    continue

                # Use original filename for display/validation, but safe suffix for temp file creation
                original_filename = file.filename
                safe_suffix_filename = secure_filename(original_filename)
                file_ext = os.path.splitext(safe_suffix_filename)[1].lower()

                if not allowed_file(original_filename):
                    msg = f"File type not allowed for: {original_filename}"
                    current_app.logger.warning("[Upload] %s", msg)
                    upload_errors.append(msg)
                    continue

                if not os.path.splitext(original_filename)[1]:
                    msg = f"Could not determine file extension for: {original_filename}"
                    current_app.logger.warning("[Upload] %s", msg)
                    upload_errors.append(msg)
                    continue

                # 1) Save the file temporarily
                parent_document_id = str(uuid.uuid4())
                temp_file_path = None
                try:
                    sc_temp_files_dir = "/sc-temp-files" if os.path.exists("/sc-temp-files") else None
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=sc_temp_files_dir) as tmp_file:
                        file.save(tmp_file.name)
                        temp_file_path = tmp_file.name
                    current_app.logger.info("[Upload] Saved temp file for %s at %s", original_filename, temp_file_path)
                except Exception as e:
                    current_app.logger.exception("[Upload] Failed to save temp file for %s", original_filename)
                    upload_errors.append(f"Failed to save temporary file for {original_filename}: {e}")
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception:
                            pass
                    continue

                try:
                    # 2) Create Cosmos metadata with status="Queued"
                    create_document(
                        file_name=original_filename,
                        user_id=user_id,
                        document_id=parent_document_id,
                        num_file_chunks=0,
                        status="Queued for processing"
                    )

                    update_document(
                        document_id=parent_document_id,
                        user_id=user_id,
                        percentage_complete=0
                    )

                    # 3) Background processing
                    future = executor.submit(
                        process_document_upload_background,
                        document_id=parent_document_id,
                        user_id=user_id,
                        temp_file_path=temp_file_path,
                        original_filename=original_filename
                    )
                    executor.submit_stored(
                        parent_document_id,
                        process_document_upload_background,
                        document_id=parent_document_id,
                        user_id=user_id,
                        temp_file_path=temp_file_path,
                        original_filename=original_filename
                    )

                    processed_docs.append({'document_id': parent_document_id, 'filename': original_filename})
                    current_app.logger.info("[Upload] Queued processing for %s (doc_id=%s)", original_filename, parent_document_id)

                except Exception as e:
                    current_app.logger.exception("[Upload] Failed to queue processing for %s", original_filename)
                    upload_errors.append(f"Failed to queue processing for {original_filename}: {e}")
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.remove(temp_file_path)
                        except Exception:
                            pass

            # 4) Response
            response_status = 200 if processed_docs and not upload_errors else 207
            if not processed_docs and upload_errors:
                response_status = 400

            current_app.logger.info(
                "[Upload] Completed: processed=%s errors=%s status=%s",
                [d['filename'] for d in processed_docs],
                upload_errors,
                response_status
            )

            response_payload = {
                'message': f'Processed {len(processed_docs)} file(s). Check status periodically.',
                'document_ids': [doc['document_id'] for doc in processed_docs],
                'processed_filenames': [doc['filename'] for doc in processed_docs],
                'errors': upload_errors
            }
            # Mirror array errors into a singular 'error' string so older clients surface details
            if response_status != 200 and upload_errors:
                response_payload['error'] = "; ".join(upload_errors)

            return jsonify(response_payload), response_status

        except Exception as e:
            # Catch-all for unexpected errors to avoid ambiguous 400s
            current_app.logger.exception("[Upload] Unhandled exception")
            return jsonify({
                'error': f"Server error during upload: {str(e)}",
                'trace': traceback.format_exc()
            }), 500


    @app.route('/api/documents', methods=['GET'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def api_get_user_documents():
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        # --- 1) Read pagination and filter parameters ---
        page = request.args.get('page', default=1, type=int)
        page_size = request.args.get('page_size', default=10, type=int)
        search_term = request.args.get('search', default=None, type=str)
        classification_filter = request.args.get('classification', default=None, type=str)
        author_filter = request.args.get('author', default=None, type=str)
        keywords_filter = request.args.get('keywords', default=None, type=str)
        abstract_filter = request.args.get('abstract', default=None, type=str)

        # Ensure page and page_size are positive
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 10

        # --- 2) Build dynamic WHERE clause and parameters ---
        query_conditions = ["c.user_id = @user_id"]
        query_params = [{"name": "@user_id", "value": user_id}]
        param_count = 0  # To generate unique parameter names

        # General Search (File Name / Title)
        if search_term:
            param_name = f"@search_term_{param_count}"
            # Safe: only call LOWER/CONTAINS when the field is defined
            query_conditions.append(
                f"((IS_DEFINED(c.file_name) AND CONTAINS(LOWER(c.file_name), LOWER({param_name}))) "
                f"OR (IS_DEFINED(c.title) AND CONTAINS(LOWER(c.title), LOWER({param_name}))))"
            )
            query_params.append({"name": param_name, "value": search_term})
            param_count += 1

        # Classification Filter
        if classification_filter:
            param_name = f"@classification_{param_count}"
            if classification_filter.lower() == 'none':
                # Filter for documents where classification is null, undefined, or empty string
                query_conditions.append(
                    "(NOT IS_DEFINED(c.document_classification) OR c.document_classification = null OR c.document_classification = '')"
                )
            else:
                query_conditions.append(f"c.document_classification = {param_name}")
                query_params.append({"name": param_name, "value": classification_filter})
                param_count += 1

        # Author Filter (Assuming 'authors' is an array of strings)
        if author_filter:
            param_name = f"@author_{param_count}"
            # Case-insensitive substring match for any author
            query_conditions.append(
                f"EXISTS(SELECT VALUE a FROM a IN c.authors WHERE CONTAINS(LOWER(a), LOWER({param_name})))"
            )
            query_params.append({"name": param_name, "value": author_filter})
            param_count += 1

        # Keywords Filter (Assuming 'keywords' is an array of strings)
        if keywords_filter:
            param_name = f"@keywords_{param_count}"
            query_conditions.append(
                f"EXISTS(SELECT VALUE k FROM k IN c.keywords WHERE CONTAINS(LOWER(k), LOWER({param_name})))"
            )
            query_params.append({"name": param_name, "value": keywords_filter})
            param_count += 1

        # Abstract Filter
        if abstract_filter:
            param_name = f"@abstract_{param_count}"
            query_conditions.append(
                f"(IS_DEFINED(c.abstract) AND CONTAINS(LOWER(c.abstract), LOWER({param_name})))"
            )
            query_params.append({"name": param_name, "value": abstract_filter})
            param_count += 1

        # Combine conditions into the WHERE clause
        where_clause = " AND ".join(query_conditions)

        # --- 3) First query: get total count based on filters ---
        try:
            count_query_str = f"SELECT VALUE COUNT(1) FROM c WHERE {where_clause}"
            count_items = list(cosmos_user_documents_container.query_items(
                query=count_query_str,
                parameters=query_params,
                enable_cross_partition_query=True
            ))
            total_count = count_items[0] if count_items else 0

        except Exception as e:
            print(f"Error executing count query: {e}")
            return jsonify({"error": f"Error counting documents: {str(e)}"}), 500

        # --- 4) Second query: fetch the page of data based on filters ---
        try:
            offset = (page - 1) * page_size
            data_query_str = f"""
                SELECT *
                FROM c
                WHERE {where_clause}
                ORDER BY c._ts DESC
                OFFSET {offset} LIMIT {page_size}
            """
            docs = list(cosmos_user_documents_container.query_items(
                query=data_query_str,
                parameters=query_params,
                enable_cross_partition_query=True
            ))
        except Exception as e:
            print(f"Error executing data query: {e}")
            return jsonify({"error": f"Error fetching documents: {str(e)}"}), 500

        # --- new: do we have any legacy documents? ---
        try:
            legacy_q = """
                SELECT VALUE COUNT(1)
                FROM c
                WHERE c.user_id = @user_id
                    AND NOT IS_DEFINED(c.percentage_complete)
            """
            legacy_docs = list(
                cosmos_user_documents_container.query_items(
                    query=legacy_q,
                    parameters=[{"name": "@user_id", "value": user_id}],
                    enable_cross_partition_query=True
                )
            )
            legacy_count = legacy_docs[0] if legacy_docs else 0
        except Exception as e:
            print(f"Error executing legacy query: {e}")
            legacy_count = 0

        # --- 5) Return results ---
        return jsonify({
            "documents": docs,
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "needs_legacy_update_check": legacy_count > 0
        }), 200


    @app.route('/api/documents/<document_id>', methods=['GET'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def api_get_user_document(document_id):
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
        
        return get_document(user_id, document_id)

    @app.route('/api/documents/<document_id>', methods=['PATCH'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def api_patch_user_document(document_id):
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        data = request.get_json()  # new metadata values from the client

        # Update allowed fields
        if 'title' in data:
            update_document(
                document_id=document_id,
                user_id=user_id,
                title=data['title']
            )
        if 'abstract' in data:
            update_document(
                document_id=document_id,
                user_id=user_id,
                abstract=data['abstract']
            )
        if 'keywords' in data:
            if isinstance(data['keywords'], list):
                update_document(
                    document_id=document_id,
                    user_id=user_id,
                    keywords=data['keywords']
                )
            else:
                update_document(
                    document_id=document_id,
                    user_id=user_id,
                    keywords=[kw.strip() for kw in data['keywords'].split(',')]
                )
        if 'publication_date' in data:
            update_document(
                document_id=document_id,
                user_id=user_id,
                publication_date=data['publication_date']
            )
        if 'document_classification' in data:
            update_document(
                document_id=document_id,
                user_id=user_id,
                document_classification=data['document_classification']
            )
        if 'authors' in data:
            if isinstance(data['authors'], list):
                update_document(
                    document_id=document_id,
                    user_id=user_id,
                    authors=data['authors']
                )
            else:
                update_document(
                    document_id=document_id,
                    user_id=user_id,
                    authors=[data['authors']]
                )

        try:
            return jsonify({'message': 'Document metadata updated successfully'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500


    @app.route('/api/documents/<document_id>', methods=['DELETE'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def api_delete_user_document(document_id):
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
        
        try:
            delete_document(user_id, document_id)
            delete_document_chunks(document_id)
            return jsonify({'message': 'Document deleted successfully'}), 200
        except Exception as e:
            return jsonify({'error': f'Error deleting document: {str(e)}'}), 500
    
    @app.route('/api/documents/<document_id>/extract_metadata', methods=['POST'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def api_extract_user_metadata(document_id):
        """
        POST /api/documents/<document_id>/extract_metadata
        Queues a background job that calls extract_document_metadata() 
        and updates the document in Cosmos DB with the new metadata.
        """
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401

        settings = get_settings()
        if not settings.get('enable_extract_meta_data'):
            return jsonify({'error': 'Metadata extraction not enabled'}), 403

        # Queue the background task (immediately returns a future)
        future = executor.submit(
            process_metadata_extraction_background,
            document_id=document_id,
            user_id=user_id
        )

        # Optionally store or track this future:
        executor.submit_stored(
            f"{document_id}_metadata",
            process_metadata_extraction_background,
            document_id=document_id,
            user_id=user_id
        )

        # Return an immediate response to the user
        return jsonify({
            'message': 'Metadata extraction has been queued. Check document status periodically.',
            'document_id': document_id
        }), 200


    @app.route("/api/get_citation", methods=["POST"])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def get_citation():
        data = request.get_json()
        user_id = get_current_user_id()
        citation_id = data.get("citation_id")

        if not user_id:
            return jsonify({"error": "User not authenticated"}), 401
                
        if not citation_id:
            return jsonify({"error": "Missing citation_id"}), 400

        try:
            search_client_user = CLIENTS['search_client_user']
            chunk = search_client_user.get_document(key=citation_id)
            if chunk.get("user_id") != user_id:
                return jsonify({"error": "Unauthorized access to citation"}), 403

            return jsonify({
                "cited_text": chunk.get("chunk_text", ""),
                "file_name": chunk.get("file_name", ""),
                "page_number": chunk.get("chunk_sequence", 0)
            }), 200

        except ResourceNotFoundError:
            pass

        try:
            search_client_group = CLIENTS['search_client_group']
            group_chunk = search_client_group.get_document(key=citation_id)

            return jsonify({
                "cited_text": group_chunk.get("chunk_text", ""),
                "file_name": group_chunk.get("file_name", ""),
                "page_number": group_chunk.get("chunk_sequence", 0)
            }), 200

        except ResourceNotFoundError:
            return jsonify({"error": "Citation not found in user or group docs"}), 404

        except Exception as e:
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        
    @app.route('/api/documents/upgrade_legacy', methods=['POST'])
    @login_required
    @user_required
    @enabled_required("enable_user_workspace")
    def api_upgrade_legacy_user_documents():
        user_id = get_current_user_id()
        # returns how many docs were updated
        count = upgrade_legacy_documents(user_id)
        return jsonify({
            "message": f"Upgraded {count} document(s) to the new format."
        }), 200

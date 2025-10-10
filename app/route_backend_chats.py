# route_backend_chats.py
from flask import request, jsonify, Response, stream_with_context
from config import *
from functions_authentication import *
from functions_search import *
from functions_bing_search import *
from functions_settings import *
from functions_tabular import tabular_plan_and_execute, should_run_tabular, analyze_or_aggregate 

# Add this helper function near your imports
def is_numeric_intent(prompt: str) -> bool:
    p = f" {prompt.lower()} "
    triggers = [
        "average ", " avg ", " mean ", " sum ", " total ", " count ", " rows ",
        " how many ", " median", " min ", " max ", " standard deviation", " std ",
        " variance", " percentile", " group by", " aggregate", " trend", " over time",
        " timeseries", " time series"
    ]
    return any(t in p for t in triggers)

def register_route_backend_chats(app):
    @app.route('/api/chat', methods=['POST'])
    @login_required
    @user_required
    def chat_api():
        settings = get_settings()
        data = request.get_json()
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'error': 'User not authenticated'
            }), 401

        # Extract from request
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id')
        hybrid_search_enabled = data.get('hybrid_search')
        selected_document_id = data.get('selected_document_id')
        bing_search_enabled = data.get('bing_search')
        image_gen_enabled = data.get('image_generation')
        document_scope = data.get('doc_scope')
        active_group_id = data.get('active_group_id')
        frontend_gpt_model = data.get('model_deployment')
        
        search_query = user_message  # Initialize search_query
        hybrid_citations_list = []   # Collect hybrid citations
        system_messages_for_augmentation = []  # system prompts to inject for augmentation
        search_results = []

        # --- Configuration ---
        # History / Summarization Settings
        raw_conversation_history_limit = settings.get('conversation_history_limit', 6)
        # Round up to nearest even number
        conversation_history_limit = math.ceil(raw_conversation_history_limit)
        if conversation_history_limit % 2 != 0:
            conversation_history_limit += 1
        enable_summarize_content_history_beyond_conversation_history_limit = settings.get('enable_summarize_content_history_beyond_conversation_history_limit', True)
        enable_summarize_content_history_for_search = settings.get('enable_summarize_content_history_for_search', False)
        number_of_historical_messages_to_summarize = settings.get('number_of_historical_messages_to_summarize', 10)

        max_file_content_length = 50000  # 50KB

        # Convert toggles from string -> bool if needed
        if isinstance(hybrid_search_enabled, str):
            hybrid_search_enabled = hybrid_search_enabled.lower() == 'true'
        if isinstance(bing_search_enabled, str):
            bing_search_enabled = bing_search_enabled.lower() == 'true'
        if isinstance(image_gen_enabled, str):
            image_gen_enabled = image_gen_enabled.lower() == 'true'

        # GPT & Image generation APIM or direct
        gpt_model = ""
        gpt_client = None
        enable_gpt_apim = settings.get('enable_gpt_apim', False)
        enable_image_gen_apim = settings.get('enable_image_gen_apim', False)

        try:
            if enable_gpt_apim:
                # read raw comma-delimited deployments
                raw = settings.get('azure_apim_gpt_deployment', '')
                if not raw:
                    raise ValueError("APIM GPT deployment name not configured.")
                apim_models = [m.strip() for m in raw.split(',') if m.strip()]
                if not apim_models:
                    raise ValueError("No valid APIM GPT deployment names found.")

                if frontend_gpt_model:
                    if frontend_gpt_model not in apim_models:
                        raise ValueError(
                            f"Requested model '{frontend_gpt_model}' is not configured for APIM."
                        )
                    gpt_model = frontend_gpt_model
                elif len(apim_models) == 1:
                    gpt_model = apim_models[0]
                else:
                    raise ValueError(
                        "Multiple APIM GPT deployments configured; please include "
                        "'model_deployment' in your request."
                    )

                gpt_client = AzureOpenAI(
                    api_version=settings.get('azure_apim_gpt_api_version'),
                    azure_endpoint=settings.get('azure_apim_gpt_endpoint'),
                    api_key=settings.get('azure_apim_gpt_subscription_key')
                )
            else:
                auth_type = settings.get('azure_openai_gpt_authentication_type')
                endpoint = settings.get('azure_openai_gpt_endpoint')
                api_version = settings.get('azure_openai_gpt_api_version')
                gpt_model_obj = settings.get('gpt_model', {})

                if gpt_model_obj and gpt_model_obj.get('selected'):
                    selected_gpt_model = gpt_model_obj['selected'][0]
                    gpt_model = selected_gpt_model['deploymentName']
                else:
                    raise ValueError("No GPT model selected or configured.")

                if frontend_gpt_model:
                    gpt_model = frontend_gpt_model
                elif gpt_model_obj and gpt_model_obj.get('selected'):
                    selected_gpt_model = gpt_model_obj['selected'][0]
                    gpt_model = selected_gpt_model['deploymentName']
                else:
                    raise ValueError("No GPT model selected or configured.")

                if auth_type == 'managed_identity':
                    token_provider = get_bearer_token_provider(DefaultAzureCredential(), cognitive_services_scope)
                    gpt_client = AzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=endpoint,
                        azure_ad_token_provider=token_provider
                    )
                else:  # Default to API Key
                    api_key = settings.get('azure_openai_gpt_key')
                    if not api_key:
                        raise ValueError("Azure OpenAI API Key not configured.")
                    gpt_client = AzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=endpoint,
                        api_key=api_key
                    )

            if not gpt_client or not gpt_model:
                raise ValueError("GPT Client or Model could not be initialized.")

        except Exception as e:
            print(f"Error initializing GPT client/model: {e}")
            return jsonify({'error': f'Failed to initialize AI model: {str(e)}'}), 500

        # ---------------------------------------------------------------------
        # 1) Load or create conversation
        # ---------------------------------------------------------------------
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conversation_item = {
                'id': conversation_id,
                'user_id': user_id,
                'last_updated': datetime.utcnow().isoformat(),
                'title': 'New Conversation'
            }
            cosmos_conversations_container.upsert_item(conversation_item)
        else:
            try:
                conversation_item = cosmos_conversations_container.read_item(item=conversation_id, partition_key=conversation_id)
            except CosmosResourceNotFoundError:
                conversation_item = {
                    'id': conversation_id,
                    'user_id': user_id,
                    'last_updated': datetime.utcnow().isoformat(),
                    'title': 'New Conversation'
                }
                cosmos_conversations_container.upsert_item(conversation_item)
            except Exception as e:
                print(f"Error reading conversation {conversation_id}: {e}")
                return jsonify({'error': f'Error reading conversation: {str(e)}'}), 500

        # ---------------------------------------------------------------------
        # 2) Append the user message to conversation immediately
        # ---------------------------------------------------------------------
        user_message_id = f"{conversation_id}_user_{int(time.time())}_{random.randint(1000,9999)}"
        user_message_doc = {
            'id': user_message_id,
            'conversation_id': conversation_id,
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.utcnow().isoformat(),
            'model_deployment_name': None
        }
        cosmos_messages_container.upsert_item(user_message_doc)

        # Set conversation title if it's still the default
        if conversation_item.get('title', 'New Conversation') == 'New Conversation' and user_message:
            new_title = (user_message[:30] + '...') if len(user_message) > 30 else user_message
            conversation_item['title'] = new_title

        conversation_item['last_updated'] = datetime.utcnow().isoformat()
        cosmos_conversations_container.upsert_item(conversation_item)

        # ---------------------------------------------------------------------
        # 3) Check Content Safety (but DO NOT return 403).
        #    If blocked, add a "safety" role message & skip GPT.
        # ---------------------------------------------------------------------
        blocked = False
        block_reasons = []
        triggered_categories = []
        blocklist_matches = []

        if settings.get('enable_content_safety') and "content_safety_client" in CLIENTS:
            try:
                content_safety_client = CLIENTS["content_safety_client"]
                request_obj = AnalyzeTextOptions(text=user_message)
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
                if len(blocklist_matches) > 0:
                    blocked = True
                    block_reasons.append("Blocklist match")
                
                if blocked:
                    safety_item = {
                        'id': str(uuid.uuid4()),
                        'user_id': user_id,
                        'conversation_id': conversation_id,
                        'message': user_message,
                        'triggered_categories': triggered_categories,
                        'blocklist_matches': blocklist_matches,
                        'timestamp': datetime.utcnow().isoformat(),
                        'reason': "; ".join(block_reasons)
                    }
                    cosmos_safety_container.upsert_item(safety_item)

                    blocked_msg_content = (
                        "Your message was blocked by Content Safety.\n\n"
                        f"**Reason**: {', '.join(block_reasons)}\n"
                        "Triggered categories:\n"
                    )
                    for cat in triggered_categories:
                        blocked_msg_content += f" - {cat['category']} (severity={cat['severity']})\n"
                    if blocklist_matches:
                        blocked_msg_content += (
                            "\nBlocklist Matches:\n" +
                            "\n".join([f" - {m['blocklistItemText']} (in {m['blocklistName']})"
                                       for m in blocklist_matches])
                        )

                    safety_message_id = f"{conversation_id}_safety_{int(time.time())}_{random.randint(1000,9999)}"

                    safety_doc = {
                        'id': safety_message_id,
                        'conversation_id': conversation_id,
                        'role': 'safety',
                        'content': blocked_msg_content.strip(),
                        'timestamp': datetime.utcnow().isoformat(),
                        'model_deployment_name': None
                    }
                    cosmos_messages_container.upsert_item(safety_doc)

                    conversation_item['last_updated'] = datetime.utcnow().isoformat()
                    cosmos_conversations_container.upsert_item(conversation_item)

                    return jsonify({
                        'reply': blocked_msg_content.strip(),
                        'blocked': True,
                        'triggered_categories': triggered_categories,
                        'blocklist_matches': blocklist_matches,
                        'conversation_id': conversation_id,
                        'conversation_title': conversation_item['title'],
                        'message_id': safety_message_id
                    }), 200

            except HttpResponseError as e:
                print(f"[Content Safety Error] {e}")
            except Exception as ex:
                print(f"[Content Safety] Unexpected error: {ex}")

        # ---------------------------------------------------------------------
        # 4) Augmentation (Search, Bing, etc.) - Run *before* final history prep
        # ---------------------------------------------------------------------
        # Hybrid Search
        if hybrid_search_enabled:
            # Optional: Summarize recent history *for search*
            if enable_summarize_content_history_for_search:
                limit_n_search = number_of_historical_messages_to_summarize * 2
                query_search = f"SELECT TOP {limit_n_search} * FROM c WHERE c.conversation_id = @conv_id ORDER BY c.timestamp DESC"
                params_search = [{"name": "@conv_id", "value": conversation_id}]
                try:
                    last_messages_desc = list(cosmos_messages_container.query_items(
                        query=query_search, parameters=params_search, partition_key=conversation_id, enable_cross_partition_query=True
                    ))
                    last_messages_asc = list(reversed(last_messages_desc))
                    if last_messages_asc and len(last_messages_asc) >= conversation_history_limit:
                        summary_prompt_search = "Please summarize the key topics or questions from this recent conversation history in 50 words or less:\n\n"
                        message_texts_search = [f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}" for msg in last_messages_asc]
                        summary_prompt_search += "\n".join(message_texts_search)

                        try:
                            summary_response_search = gpt_client.chat.completions.create(
                                model=gpt_model,
                                messages=[{"role": "system", "content": summary_prompt_search}],
                                max_tokens=100
                            )
                            summary_for_search = summary_response_search.choices[0].message.content.strip()
                            if summary_for_search:
                                search_query = f"Based on the recent conversation about: '{summary_for_search}', the user is now asking: {user_message}"
                        except Exception as e:
                            print(f"Error summarizing conversation for search: {e}")
                except Exception as e:
                    print(f"Error fetching messages for search summarization: {e}")

            try:
                search_args = {
                    "query": search_query,
                    "user_id": user_id,
                    "top_n": 12,
                    "doc_scope": document_scope,
                    "active_group_id": active_group_id
                }
                if selected_document_id:
                    search_args["document_id"] = selected_document_id

                search_results = hybrid_search(**search_args)
            except Exception as e:
                print(f"Error during hybrid search: {e}")
                return jsonify({
                    'error': 'There was an issue with the embedding process. Please check with an admin on embedding configuration.'
                }), 500

            if search_results:
                retrieved_texts = []
                combined_documents = []
                classifications_found = set(conversation_item.get('classification', []))

                for doc in search_results:
                    chunk_text = doc.get('chunk_text', '')
                    file_name = doc.get('file_name', 'Unknown')
                    version = doc.get('version', 'N/A')
                    chunk_sequence = doc.get('chunk_sequence', 0)
                    page_number = doc.get('page_number') or chunk_sequence or 1
                    citation_id = doc.get('id', str(uuid.uuid4()))
                    classification = doc.get('document_classification')
                    chunk_id = doc.get('chunk_id', str(uuid.uuid4()))
                    score = doc.get('score', 0.0)
                    group_id = doc.get('group_id', None)

                    citation = f"(Source: {file_name}, Page: {page_number}) [#{citation_id}]"
                    retrieved_texts.append(f"{chunk_text}\n{citation}")
                    combined_documents.append({
                        "file_name": file_name, 
                        "citation_id": citation_id, 
                        "page_number": page_number,
                        "version": version, 
                        "classification": classification, 
                        "chunk_text": chunk_text,
                        "chunk_sequence": chunk_sequence,
                        "chunk_id": chunk_id,
                        "score": score,
                        "group_id": group_id,
                    })
                    if classification:
                        classifications_found.add(classification)

                retrieved_content = "\n\n".join(retrieved_texts)
                system_prompt_search = f"""You are an AI assistant. Use the following retrieved document excerpts to answer the user's question. Cite sources using the format (Source: filename, Page: page number).

Retrieved Excerpts:
{retrieved_content}

Based *only* on the information provided above, answer the user's query. If the answer isn't in the excerpts, say so.

Example
User: What is the policy on double dipping?
Assistant: The policy prohibits entities from using federal funds received through one program to apply for additional funds through another program, commonly known as 'double dipping' (Source: PolicyDocument.pdf, Page: 12)
"""
                system_messages_for_augmentation.append({
                    'role': 'system',
                    'content': system_prompt_search,
                    'documents': combined_documents
                })

                for source_doc in combined_documents:
                    citation_data = {
                        "file_name": source_doc.get("file_name"),
                        "sheet_name": source_doc.get("sheet_name"),
                        "citation_id": source_doc.get("citation_id"),
                        "page_number": source_doc.get("page_number"),
                        "chunk_id": source_doc.get("chunk_id"),
                        "chunk_sequence": source_doc.get("chunk_sequence"),
                        "score": source_doc.get("score"),
                        "group_id": source_doc.get("group_id"),
                        "version": source_doc.get("version"),
                        "classification": source_doc.get("classification")
                    }
                    hybrid_citations_list.append(citation_data)

                hybrid_citations_list.sort(key=lambda x: x.get('page_number', 0), reverse=True)

                if list(classifications_found) != conversation_item.get('classification', []):
                    conversation_item['classification'] = list(classifications_found)

        # Bing Search
        bing_results = []
        bing_citations_list = []
        
        if bing_search_enabled:
            try:
                bing_results = process_query_with_bing_and_llm(user_message)
            except Exception as e:
                print(f"Error during Bing search: {e}")

            if bing_results:
                retrieved_texts_bing = []
                for r in bing_results:
                    title = r.get("name", "Untitled")
                    snippet = r.get("snippet", "No snippet available.")
                    url = r.get("url", "#")
                    citation = f"(Source: {title}) [{url}]"
                    retrieved_texts_bing.append(f"{snippet}\n{citation}")
                    bing_citation_data = {
                        "sheet_name": source_doc.get("sheet_name"),
                        "title": title,
                        "url": url,
                        "snippet": snippet
                    }
                    bing_citations_list.append(bing_citation_data)

                retrieved_content_bing = "\n\n".join(retrieved_texts_bing)
                system_prompt_bing = f"""You are an AI assistant. Use the following web search results to answer the user's question. Cite sources using the format (Source: page_title).

Web Search Results:
{retrieved_content_bing}

Based *only* on the information provided above, answer the user's query. If the answer isn't in the results, say so.

Example:
User: What is the capital of France?
Assistant: The capital of France is Paris (Source: OfficialFrancePage)
"""
                system_messages_for_augmentation.append({
                    'role': 'system',
                    'content': system_prompt_bing
                })

        # Image Generation
        if image_gen_enabled:
            if enable_image_gen_apim:
                image_gen_model = settings.get('azure_apim_image_gen_deployment')
                image_gen_client = AzureOpenAI(
                    api_version=settings.get('azure_apim_image_gen_api_version'),
                    azure_endpoint=settings.get('azure_apim_image_gen_endpoint'),
                    api_key=settings.get('azure_apim_image_gen_subscription_key')
                )
            else:
                if (settings.get('azure_openai_image_gen_authentication_type') == 'managed_identity'):
                    token_provider = get_bearer_token_provider(DefaultAzureCredential(), cognitive_services_scope)
                    image_gen_client = AzureOpenAI(
                        api_version=settings.get('azure_openai_image_gen_api_version'),
                        azure_endpoint=settings.get('azure_openai_image_gen_endpoint'),
                        azure_ad_token_provider=token_provider
                    )
                    image_gen_model_obj = settings.get('image_gen_model', {})
                    if image_gen_model_obj and image_gen_model_obj.get('selected'):
                        selected_image_gen_model = image_gen_model_obj['selected'][0]
                        image_gen_model = selected_image_gen_model['deploymentName']
                else:
                    image_gen_client = AzureOpenAI(
                        api_version=settings.get('azure_openai_image_gen_api_version'),
                        azure_endpoint=settings.get('azure_openai_image_gen_endpoint'),
                        api_key=settings.get('azure_openai_image_gen_key')
                    )
                    image_gen_obj = settings.get('image_gen_model', {})
                    if image_gen_obj and image_gen_obj.get('selected'):
                        selected_image_gen_model = image_gen_obj['selected'][0]
                        image_gen_model = selected_image_gen_model['deploymentName']

            try:
                image_response = image_gen_client.images.generate(
                    prompt=user_message,
                    n=1,
                    model=image_gen_model
                )
                generated_image_url = json.loads(image_response.model_dump_json())['data'][0]['url']

                image_message_id = f"{conversation_id}_image_{int(time.time())}_{random.randint(1000,9999)}"
                image_doc = {
                    'id': image_message_id,
                    'conversation_id': conversation_id,
                    'role': 'image',
                    'content': generated_image_url,
                    'prompt': user_message,
                    'created_at': datetime.utcnow().isoformat(),
                    'timestamp': datetime.utcnow().isoformat(),
                    'model_deployment_name': image_gen_model
                }
                cosmos_messages_container.upsert_item(image_doc)

                conversation_item['last_updated'] = datetime.utcnow().isoformat()
                cosmos_conversations_container.upsert_item(conversation_item)

                return jsonify({
                    'reply': "Image loading...",
                    'image_url': generated_image_url,
                    'conversation_id': conversation_id,
                    'conversation_title': conversation_item['title'],
                    'model_deployment_name': image_gen_model,
                    'message_id': image_message_id
                }), 200
            except Exception as e:
                return jsonify({
                    'error': f'Image generation failed: {str(e)}'
                }), 500

        # ---------------------------------------------------------------------
        
        # === Tabular Enrichment (non-stream) ===
        try:
            # Use new intent detection for tabular enrichment
            if should_run_tabular(user_message) or is_numeric_intent(user_message):
                tabular = tabular_plan_and_execute(
                    settings=settings,
                    user_id=user_id,
                    prompt=user_message,
                    selected_document_id=selected_document_id,
                    active_group_id=active_group_id,
                    timebox_sec=int(settings.get('tabular_timebox_sec', 8))
                )
                tabular_content = None
                if isinstance(tabular, dict):
                    tabular_content = tabular.get('system_message') or tabular.get('content')
                elif isinstance(tabular, str):
                    tabular_content = tabular
                if tabular_content:
                    system_messages_for_augmentation.append({ "role": "system", "content": tabular_content })
        except Exception as e:
            print(f"[tabular non-stream] enrichment failed: {e}")

        # 5) Prepare FINAL conversation history for GPT (including summarization)
        # ---------------------------------------------------------------------
        conversation_history_for_api = []
        summary_of_older = ""

        try:
            # Fetch ALL messages for potential summarization, sorted OLD->NEW
            all_messages_query = "SELECT * FROM c WHERE c.conversation_id = @conv_id ORDER BY c.timestamp ASC"
            params_all = [{"name": "@conv_id", "value": conversation_id}]
            all_messages = list(cosmos_messages_container.query_items(
                query=all_messages_query, parameters=params_all, partition_key=conversation_id, enable_cross_partition_query=True
            ))

            total_messages = len(all_messages)

            num_recent_messages = min(total_messages, conversation_history_limit)
            num_older_messages = total_messages - num_recent_messages

            recent_messages = all_messages[-num_recent_messages:]
            older_messages_to_summarize = all_messages[:num_older_messages]

            if enable_summarize_content_history_beyond_conversation_history_limit and older_messages_to_summarize:
                print(f"Summarizing {len(older_messages_to_summarize)} older messages for conversation {conversation_id}")
                summary_prompt_older = (
                    "Summarize the following conversation history concisely (around 50-100 words), "
                    "focusing on key facts, decisions, or context that might be relevant for future turns. "
                    "Do not add any introductory phrases like 'Here is a summary'.\n\n"
                    "Conversation History:\n"
                )
                message_texts_older = []
                for msg in older_messages_to_summarize:
                    role = msg.get('role', 'user')
                    if role in ['system', 'safety', 'blocked', 'image', 'file']:
                        continue
                    content = msg.get('content', '')
                    message_texts_older.append(f"{role.upper()}: {content}")

                if message_texts_older:
                    summary_prompt_older += "\n".join(message_texts_older)
                    try:
                        summary_response_older = gpt_client.chat.completions.create(
                            model=gpt_model,
                            messages=[{"role": "system", "content": summary_prompt_older}],
                            max_tokens=150,
                            temperature=0.3
                        )
                        summary_of_older = summary_response_older.choices[0].message.content.strip()
                        print(f"Generated summary: {summary_of_older}")
                    except Exception as e:
                        print(f"Error summarizing older conversation history: {e}")
                        summary_of_older = ""
                else:
                    print("No summarizable content found in older messages.")

            if summary_of_older:
                conversation_history_for_api.append({
                    "role": "system",
                    "content": f"<Summary of previous conversation context>\n{summary_of_older}\n</Summary of previous conversation context>"
                })

            # Persist and add augmentation system messages
            for aug_msg in system_messages_for_augmentation:
                system_message_id = f"{conversation_id}_system_aug_{int(time.time())}_{random.randint(1000,9999)}"
                system_doc = {
                    'id': system_message_id,
                    'conversation_id': conversation_id,
                    'role': aug_msg.get('role'),
                    'content': aug_msg.get('content'),
                    'search_query': search_query,
                    'user_message': user_message,
                    'model_deployment_name': None,
                    'timestamp': datetime.utcnow().isoformat()
                }
                cosmos_messages_container.upsert_item(system_doc)
                conversation_history_for_api.append(aug_msg)

            # Add the recent messages (user, assistant, relevant file as system)
            allowed_roles_in_history = ['user', 'assistant']
            max_file_content_length_in_history = 1000

            for message in recent_messages:
                role = message.get('role')
                content = message.get('content')

                if role in allowed_roles_in_history:
                    conversation_history_for_api.append({"role": role, "content": content})
                elif role == 'file':
                    filename = message.get('filename', 'uploaded_file')
                    file_content = message.get('file_content', '')
                    display_content = file_content[:max_file_content_length_in_history]
                    if len(file_content) > max_file_content_length_in_history:
                        display_content += "..."
                    conversation_history_for_api.append({
                        'role': 'system',
                        'content': f"[User uploaded a file named '{filename}'. Content preview:\n{display_content}]\nUse this file context if relevant."
                    })

            if not conversation_history_for_api or conversation_history_for_api[-1]['role'] != 'user':
                print("Warning: Last message in history is not the user's current message. Appending.")
                user_msg_found = False
                for msg in reversed(recent_messages):
                    if msg['role'] == 'user' and msg['id'] == user_message_id:
                        conversation_history_for_api.append({"role": "user", "content": msg['content']})
                        user_msg_found = True
                        break
                if not user_msg_found:
                    conversation_history_for_api.append({"role": "user", "content": user_message})

        except Exception as e:
            print(f"Error preparing conversation history: {e}")
            return jsonify({'error': f'Error preparing conversation history: {str(e)}'}), 500

        # Insert default system prompt if not present
        default_system_prompt = settings.get('default_system_prompt', '').strip()
        if default_system_prompt:
            has_general_system_prompt = any(
                msg.get('role') == 'system' and not (
                    msg.get('content', '').startswith('<Summary of previous conversation context>') or
                    "retrieved document excerpts" in msg.get('content', '') or
                    "web search results" in msg.get('content', '')
                )
                for msg in conversation_history_for_api
            )
            if not has_general_system_prompt:
                insert_idx = 0
                if conversation_history_for_api and conversation_history_for_api[0].get('role') == 'system' and conversation_history_for_api[0].get('content', '').startswith('<Summary of previous conversation context>'):
                    insert_idx = 1
                conversation_history_for_api.insert(insert_idx, {
                    "role": "system",
                    "content": default_system_prompt
                })

        # Final GPT Call
        ai_message = "Sorry, I encountered an error."
        final_model_used = gpt_model

        if not conversation_history_for_api:
            return jsonify({'error': 'Cannot generate response: No conversation history available.'}), 500
        if conversation_history_for_api[-1].get('role') != 'user':
            print(f"Error: Last message role is not user: {conversation_history_for_api[-1].get('role')}")
            return jsonify({'error': 'Internal error: Conversation history improperly formed.'}), 500

        try:
            response = gpt_client.chat.completions.create(
                model=final_model_used,
                messages=conversation_history_for_api,
            )
            ai_message = response.choices[0].message.content
        except Exception as e:
            print(f"Error during final GPT completion: {str(e)}")
            if "context length" in str(e).lower():
                ai_message = "Sorry, the conversation history is too long even after summarization. Please start a new conversation or try a shorter message."
            else:
                ai_message = f"Sorry, I encountered an error generating the response. Details: {str(e)}"

        assistant_message_id = f"{conversation_id}_assistant_{int(time.time())}_{random.randint(1000,9999)}"
        assistant_doc = {
            'id': assistant_message_id,
            'conversation_id': conversation_id,
            'role': 'assistant',
            'content': ai_message,
            'timestamp': datetime.utcnow().isoformat(),
            'augmented': bool(system_messages_for_augmentation),
            'hybrid_citations': hybrid_citations_list,
            'hybridsearch_query': search_query if hybrid_search_enabled and search_results else None,
            'web_search_citations': bing_citations_list,
            'user_message': user_message,
            'model_deployment_name': final_model_used
        }
        cosmos_messages_container.upsert_item(assistant_doc)

        conversation_item['last_updated'] = datetime.utcnow().isoformat()
        cosmos_conversations_container.upsert_item(conversation_item)

        return jsonify({
            'reply': ai_message,
            'conversation_id': conversation_id,
            'conversation_title': conversation_item['title'],
            'classification': conversation_item.get('classification', []),
            'model_deployment_name': final_model_used,
            'message_id': assistant_message_id,
            'blocked': False,
            'augmented': bool(system_messages_for_augmentation),
            'hybrid_citations': hybrid_citations_list,
            'web_search_citations': bing_citations_list
        }), 200

    @app.route('/api/chat/stream', methods=['POST'])
    @login_required
    @user_required
    def chat_api_stream():
        """NDJSON streaming endpoint for chat with proper interrupt and citations."""
        from flask import Response, stream_with_context, jsonify, request
        from werkzeug.exceptions import ClientDisconnected
    
        settings = get_settings()
        data = request.get_json() or {}
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'User not authenticated'}), 401
    
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id')
        frontend_gpt_model = data.get('model_deployment')
        hybrid_search_enabled = data.get('hybrid_search')
        selected_document_id = data.get('selected_document_id')
        document_scope = data.get('doc_scope')
        active_group_id = data.get('active_group_id')
    
        if isinstance(hybrid_search_enabled, str):
            hybrid_search_enabled = hybrid_search_enabled.lower() == 'true'
    
        # Initialize model / client
        gpt_model = ""
        gpt_client = None
        try:
            if settings.get('enable_gpt_apim', False):
                raw = settings.get('azure_apim_gpt_deployment', '') or ''
                apim_models = [m.strip() for m in raw.split(',') if m.strip()]
                if frontend_gpt_model:
                    if frontend_gpt_model not in apim_models:
                        raise ValueError(f"Requested model '{frontend_gpt_model}' is not configured for APIM.")
                    gpt_model = frontend_gpt_model
                elif len(apim_models) == 1:
                    gpt_model = apim_models[0]
                else:
                    raise ValueError("Multiple APIM GPT deployments configured; include 'model_deployment' in request.")
    
                gpt_client = AzureOpenAI(
                    api_version=settings.get('azure_apim_gpt_api_version'),
                    azure_endpoint=settings.get('azure_apim_gpt_endpoint'),
                    api_key=settings.get('azure_apim_gpt_subscription_key')
                )
            else:
                auth_type = settings.get('azure_openai_gpt_authentication_type')
                endpoint = settings.get('azure_openai_gpt_endpoint')
                api_version = settings.get('azure_openai_gpt_api_version')
                gpt_model_obj = settings.get('gpt_model', {})
                if frontend_gpt_model:
                    gpt_model = frontend_gpt_model
                elif gpt_model_obj and gpt_model_obj.get('selected'):
                    gpt_model = gpt_model_obj['selected'][0]['deploymentName']
                else:
                    raise ValueError("No GPT model selected or configured.")
    
                if auth_type == 'managed_identity':
                    token_provider = get_bearer_token_provider(DefaultAzureCredential(), cognitive_services_scope)
                    gpt_client = AzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=endpoint,
                        azure_ad_token_provider=token_provider
                    )
                else:
                    api_key = settings.get('azure_openai_gpt_key')
                    if not api_key:
                        raise ValueError("Azure OpenAI API Key not configured.")
                    gpt_client = AzureOpenAI(
                        api_version=api_version,
                        azure_endpoint=endpoint,
                        api_key=api_key
                    )
    
            if not gpt_client or not gpt_model:
                raise ValueError("GPT Client or Model could not be initialized.")
        except Exception as e:
            print(f"[stream] Error initializing GPT client/model: {e}")
            return jsonify({'error': f'Failed to initialize AI model: {str(e)}'}), 500
    
        # Ensure conversation exists and persist user message
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            conversation_item = {
                'id': conversation_id,
                'user_id': user_id,
                'last_updated': datetime.utcnow().isoformat(),
                'title': 'New Conversation'
            }
            cosmos_conversations_container.upsert_item(conversation_item)
        else:
            try:
                conversation_item = cosmos_conversations_container.read_item(item=conversation_id, partition_key=conversation_id)
            except CosmosResourceNotFoundError:
                conversation_item = {
                    'id': conversation_id,
                    'user_id': user_id,
                    'last_updated': datetime.utcnow().isoformat(),
                    'title': 'New Conversation'
                }
                cosmos_conversations_container.upsert_item(conversation_item)
    
        user_message_id = f"{conversation_id}_user_{int(time.time())}_{random.randint(1000,9999)}"
        cosmos_messages_container.upsert_item({
            'id': user_message_id,
            'conversation_id': conversation_id,
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.utcnow().isoformat()
        })
    
        if conversation_item.get('title', 'New Conversation') == 'New Conversation' and user_message:
            conversation_item['title'] = (user_message[:30] + '...') if len(user_message) > 30 else user_message
        conversation_item['last_updated'] = datetime.utcnow().isoformat()
        cosmos_conversations_container.upsert_item(conversation_item)
    
        # Build recent history
        try:
            history_limit = int(settings.get('conversation_history_limit', 6) or 6)
            all_msgs = list(cosmos_messages_container.query_items(
                query="SELECT * FROM c WHERE c.conversation_id = @conv_id ORDER BY c.timestamp ASC",
                parameters=[{"name":"@conv_id","value": conversation_id}],
                partition_key=conversation_id,
                enable_cross_partition_query=True
            ))
            recent = all_msgs[-history_limit:]
            file_summaries = []
            for m in recent:
                if m.get('role') == 'file':
                    fname = m.get('filename', 'uploaded_file')
                    snippet = (m.get('file_content', '') or '')[:1000]
                    if len(snippet) == 1000: snippet += '...'
                    file_summaries.append(f"[User uploaded a file named '{fname}'. Content preview:\n{snippet}]\nUse this file context if relevant.")
            history = [{'role': m.get('role'), 'content': m.get('content','')} for m in recent if m.get('role') in ('user','assistant')]
            if not history or history[-1].get('role') != 'user':
                history.append({'role':'user','content': user_message})
        except Exception as e:
            print(f"[stream] history prep error: {e}")
            history, file_summaries = [{'role':'user','content': user_message}], []
    
        # Hybrid search (optional)
        augmented = False
        hybrid_citations_list = []
        system_search_prompt = None
        search_query = user_message
    
        try:
            if hybrid_search_enabled:
                args = {
                    'query': search_query,
                    'user_id': user_id,
                    'top_n': 12,
                    'doc_scope': document_scope,
                    'active_group_id': active_group_id
                }
                if selected_document_id:
                    args['document_id'] = selected_document_id
                results = hybrid_search(**args) or []
                if results:
                    augmented = True
                    retrieved_texts = []
                    combined_docs = []
                    classifications_found = set(conversation_item.get('classification', []))
                    for doc in results:
                        chunk_text = doc.get('chunk_text','')
                        file_name = doc.get('file_name','Unknown')
                        version = doc.get('version','N/A')
                        chunk_sequence = doc.get('chunk_sequence',0)
                        page_number = doc.get('page_number') or chunk_sequence or 1
                        citation_id = doc.get('id') or str(uuid.uuid4())
                        classification = doc.get('document_classification')
                        chunk_id = doc.get('chunk_id') or str(uuid.uuid4())
                        score = doc.get('score', 0.0)
                        group_id = doc.get('group_id')
                        retrieved_texts.append(f"{chunk_text}\n(Source: {file_name}, Page: {page_number}) [#{citation_id}]")
                        combined_docs.append({
                            'file_name': file_name,
                            'citation_id': citation_id,
                            'page_number': page_number,
                            'version': version,
                            'classification': classification,
                            'chunk_text': chunk_text,
                            'chunk_sequence': chunk_sequence,
                            'chunk_id': chunk_id,
                            'score': score,
                            'group_id': group_id
                        })
                        if classification:
                            classifications_found.add(classification)
    
                    retrieved_content = "\n\n".join(retrieved_texts)
                    system_search_prompt = (
                        "You are an AI assistant. Use the following retrieved document excerpts to answer the user's question. "
                        "Cite sources using the format (Source: filename, Page: page number).\n\n"
                        "Retrieved Excerpts:\n" + retrieved_content + "\n\n"
                        "Based *only* on the information provided above, answer the user's query. If the answer isn't in the excerpts, say so."
                    )
    
                    for src in combined_docs:
                        hybrid_citations_list.append({
                            'file_name': src.get('file_name'),
                            'citation_id': src.get('citation_id'),
                            'page_number': src.get('page_number'),
                            'chunk_id': src.get('chunk_id'),
                            'chunk_sequence': src.get('chunk_sequence'),
                            'score': src.get('score'),
                            'group_id': src.get('group_id'),
                            'version': src.get('version'),
                            'classification': src.get('classification')
                        })
                    hybrid_citations_list.sort(key=lambda x: x.get('page_number',0), reverse=True)
    
                    if list(classifications_found) != conversation_item.get('classification', []):
                        conversation_item['classification'] = list(classifications_found)
                        cosmos_conversations_container.upsert_item(conversation_item)
        except Exception as e:
            print(f"[stream] hybrid search error: {e}")
    
        # Tabular enrichment (optional)
        tabular_system_content = None
        try:
            # Use new intent detection for tabular enrichment in stream
            if should_run_tabular(user_message) or is_numeric_intent(user_message):
                tabular = tabular_plan_and_execute(
                    settings=settings,
                    user_id=user_id,
                    prompt=user_message,
                    selected_document_id=selected_document_id,
                    active_group_id=active_group_id,
                    timebox_sec=int(settings.get('tabular_timebox_sec', 8))
                )
                if isinstance(tabular, dict):
                    tabular_system_content = tabular.get('system_message') or tabular.get('content')
                elif isinstance(tabular, str):
                    tabular_system_content = tabular
        except Exception as e:
            print(f"[tabular stream] enrichment failed: {e}")
    
        # Build model messages
        messages = []
        default_system_prompt = settings.get('default_system_prompt', '').strip()
        if default_system_prompt:
            messages.append({'role':'system','content': default_system_prompt})
        if tabular_system_content:
            messages.append({'role':'system','content': tabular_system_content})
        for fs in file_summaries:
            messages.append({'role':'system','content': fs})
        if system_search_prompt:
            messages.append({'role':'system','content': system_search_prompt})
        messages.extend(history)
    
        partial = []
        aborted = False
        assistant_message_id = None
    
        def as_ndjson(obj: dict) -> str:
            import json as _json
            try:
                return _json.dumps(obj, ensure_ascii=False) + "\n"
            except Exception:
                return _json.dumps({'type':'error','message':'serialization error'}) + "\n"

        def emit_token(tok: str) -> str:
            return as_ndjson({'type':'token','token': tok})

        @stream_with_context
        def token_generator():
            nonlocal aborted, assistant_message_id
            # start + meta frames
            yield as_ndjson({'type':'start','ts': int(time.time()*1000)})
            # meta frame so UI can flip Send->Stop & update title
            yield as_ndjson({'type':'meta','conversation_id': conversation_id,'conversation_title': conversation_item.get('title')})
            try:

                # Numeric fast-path: compute deterministically and stream result
                if is_numeric_intent(user_message):
                    tool = analyze_or_aggregate(
                        user_message,
                        {
                            "user_id": user_id,
                            "active_group_id": active_group_id,
                            "selected_document_id": selected_document_id
                        },
                        None
                    )
                    if not tool or not tool.get("audit") or tool["audit"].get("rows_used", 0) <= 0:
                        yield as_ndjson({'type':'error','message': 'I need the original structured table (sheet & column) to compute this reliably.'})
                        yield as_ndjson({'type':'done','augmented': False,'hybrid_citations': [],'web_search_citations': [],'message_id': None})
                        return

                    text = tool.get("text") or "Computation completed."
                    for i in range(0, len(text), 60):
                        yield emit_token(text[i:i+60])

                    details = {
                        "audit": tool.get("audit", {}),
                        "structured": tool.get("structured", {}),
                        "citations": tool.get("citations", [])
                    }
                    import json as _json
                    payload = _json.dumps(details).replace("'", "&apos;")
                    yield as_ndjson({'type':'token','token': "\\n\\n<calc-details data-json='%s'></calc-details>" % payload})

                    # Persist and finish
                    try:
                        final_text = text + "\\n\\n[calculation details attached]"
                        assistant_message_id = f"{conversation_id}_assistant_{int(time.time())}_{random.randint(1000,9999)}"
                        cosmos_messages_container.upsert_item({
                            'id': assistant_message_id,
                            'conversation_id': conversation_id,
                            'role': 'assistant',
                            'content': final_text,
                            'timestamp': datetime.utcnow().isoformat(),
                            'augmented': False,
                            'hybrid_citations': [],
                            'hybridsearch_query': None,
                            'web_search_citations': [],
                            'user_message': user_message,
                            'model_deployment_name': gpt_model,
                            'stopped': False
                        })
                        conversation_item['last_updated'] = datetime.utcnow().isoformat()
                        cosmos_conversations_container.upsert_item(conversation_item)
                    except Exception as persist_err:
                        print(f"[stream numeric] persist error: {persist_err}")
                    yield as_ndjson({'type':'done','augmented': False,'hybrid_citations': [],'web_search_citations': [],'message_id': assistant_message_id})
                    return
                stream = gpt_client.chat.completions.create(
                    model=gpt_model,
                    messages=messages,
                    stream=True,
                )
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta.content or ''
                    except Exception:
                        delta = ''
                    if delta:
                        partial.append(delta)
                        yield emit_token(delta)
            except (ClientDisconnected, GeneratorExit, BrokenPipeError, ConnectionResetError):
                aborted = True
            except Exception as e:
                yield as_ndjson({'type':'error','message': str(e)})
            finally:
                try:
                    assistant_message_id = f"{conversation_id}_assistant_{int(time.time())}_{random.randint(1000,9999)}"
                    cosmos_messages_container.upsert_item({
                        'id': assistant_message_id,
                        'conversation_id': conversation_id,
                        'role': 'assistant',
                        'content': ''.join(partial) + (' [stopped]' if aborted else ''),
                        'timestamp': datetime.utcnow().isoformat(),
                        'augmented': bool(system_search_prompt),
                        'hybrid_citations': hybrid_citations_list if system_search_prompt else [],
                        'hybridsearch_query': search_query if system_search_prompt else None,
                        'web_search_citations': [],
                        'user_message': user_message,
                        'model_deployment_name': gpt_model,
                        'stopped': aborted
                    })
                    conversation_item['last_updated'] = datetime.utcnow().isoformat()
                    cosmos_conversations_container.upsert_item(conversation_item)
                except Exception as persist_err:
                    print(f"[stream] persist error: {persist_err}")
                yield as_ndjson({
                    'type': 'done',
                    'augmented': bool(system_search_prompt),
                    'hybrid_citations': hybrid_citations_list if system_search_prompt else [],
                    'web_search_citations': [],
                    'message_id': assistant_message_id
                })

        headers = {
            'Cache-Control': 'no-cache, no-transform',
            'Content-Type': 'application/x-ndjson',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
        return Response(token_generator(), headers=headers)

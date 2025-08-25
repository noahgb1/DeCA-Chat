# functions_settings.py

from config import *

def get_settings():
    default_settings = {
        'id': 'app_settings',
        # -- Your entire default dictionary here --
        'app_title': 'Simple Chat',
        'landing_page_text': 'You can add text here and it supports Markdown. '
                             'You agree to our [acceptable user policy](acceptable_use_policy.html) by using this service.',
        'landing_page_alignment': 'left',
        'show_logo': False,
        'hide_app_title': False,
        'custom_logo_base64': '',
        'logo_version': 1,
        'custom_favicon_base64': '',
        'favicon_version': 1,
        'enable_dark_mode_default': False,

        # GPT Settings
        'enable_gpt_apim': False,
        'azure_openai_gpt_endpoint': '',
        'azure_openai_gpt_api_version': '2024-05-01-preview',
        'azure_openai_gpt_authentication_type': 'key',
        'azure_openai_gpt_subscription_id': '',
        'azure_openai_gpt_resource_group': '',
        'azure_openai_gpt_key': '',
        'gpt_model': {
            "selected": [],
            "all": []
        },
        'azure_apim_gpt_endpoint': '',
        'azure_apim_gpt_subscription_key': '',
        'azure_apim_gpt_deployment': '',
        'azure_apim_gpt_api_version': '',

        # Embeddings Settings
        'enable_embedding_apim': False,
        'azure_openai_embedding_endpoint': '',
        'azure_openai_embedding_api_version': '2024-05-01-preview',
        'azure_openai_embedding_authentication_type': 'key',
        'azure_openai_embedding_subscription_id': '',
        'azure_openai_embedding_resource_group': '',
        'azure_openai_embedding_key': '',
        'embedding_model': {
            "selected": [],
            "all": []
        },
        'azure_apim_embedding_endpoint': '',
        'azure_apim_embedding_subscription_key': '',
        'azure_apim_embedding_deployment': '',
        'azure_apim_embedding_api_version': '',

        # Image Generation Settings
        'enable_image_generation': False,
        'enable_image_gen_apim': False,
        'azure_openai_image_gen_endpoint': '',
        'azure_openai_image_gen_api_version': '2024-05-01-preview',
        'azure_openai_image_gen_authentication_type': 'key',
        'azure_openai_image_gen_subscription_id': '',
        'azure_openai_image_gen_resource_group': '',
        'azure_openai_image_gen_key': '',
        'image_gen_model': {
            "selected": [],
            "all": []
        },
        'azure_apim_image_gen_endpoint': '',
        'azure_apim_image_gen_subscription_key': '',
        'azure_apim_image_gen_deployment': '',
        'azure_apim_image_gen_api_version': '',

        # Redis Cache Settings
        'enable_redis_cache': False,
        'redis_url': '',
        'redis_key': '',
        'redis_auth_type': '',


        # Workspaces
        'enable_user_workspace': True,
        'enable_group_workspaces': True,
        'require_member_of_create_group': False,
        'enable_public_workspaces': False,
        'require_member_of_create_public_workspace': False,

        # Multimedia
        'enable_video_file_support': False,
        'enable_audio_file_support': False,

        # Metadata Extraction
        'enable_extract_meta_data': False,
        'metadata_extraction_model': '',
        'enable_summarize_content_history_for_search': False,
        'number_of_historical_messages_to_summarize': 10,
        'enable_summarize_content_history_beyond_conversation_history_limit': False,

        # Document Classification
        'enable_document_classification': False,
        'document_classification_categories': [
            {"label": "None", "color": "#808080"},
            {"label": "N/A", "color": "#808080"},
            {"label": "Pending", "color": "#0000FF"}
        ],

        # Enhanced Citations
        'enable_enhanced_citations': False,
        'enable_enhanced_citations_mount': False,
        'enhanced_citations_mount': '/view_documents',
        'office_docs_storage_account_url': '',
        'office_docs_authentication_type': 'key',
        'office_docs_key': '',
        'video_files_storage_account_url': '',
        'video_files_authentication_type': 'key',
        'video_files_key': '',
        'audio_files_storage_account_url': '',
        'audio_files_authentication_type': 'key',
        'audio_files_key': '',

        # Safety (Content Safety) Settings
        'enable_content_safety': False,
        'require_member_of_safety_violation_admin': False,
        'content_safety_endpoint': '',
        'content_safety_key': '',
        'content_safety_authentication_type': 'key',
        'enable_content_safety_apim': False,
        'azure_apim_content_safety_endpoint': '',
        'azure_apim_content_safety_subscription_key': '',

        # User Feedback / Conversation Archiving
        'enable_user_feedback': True,
        'require_member_of_feedback_admin': False,
        'enable_conversation_archiving': False,

        # Search and Extract
        'enable_web_search': False,
        'bing_search_key': '',
        'enable_web_search_apim': False,
        'azure_apim_web_search_endpoint': '',
        'azure_apim_web_search_subscription_key': '',

        'azure_ai_search_endpoint': '',
        'azure_ai_search_key': '',
        'azure_ai_search_authentication_type': 'key',
        'enable_ai_search_apim': False,
        'azure_apim_ai_search_endpoint': '',
        'azure_apim_ai_search_subscription_key': '',

        'azure_document_intelligence_endpoint': '',
        'azure_document_intelligence_key': '',
        'azure_document_intelligence_authentication_type': 'key',
        'enable_document_intelligence_apim': False,
        'azure_apim_document_intelligence_endpoint': '',
        'azure_apim_document_intelligence_subscription_key': '',

        # Other
        'max_file_size_mb': 150,
        'conversation_history_limit': 10,
        'default_system_prompt': '',
        'enable_file_processing_logs': True,
        'enable_external_healthcheck': False,

        # Video file settings with Azure Video Indexer Settings
        'video_indexer_endpoint': 'https://api.videoindexer.ai',
        'video_indexer_location': '',
        'video_indexer_account_id': '',
        'video_indexer_api_key': '',
        'video_indexer_resource_group': '',
        'video_indexer_subscription_id': '',
        'video_indexer_account_name': '',
        'video_indexer_arm_api_version': '2021-11-10-preview',
        'video_index_timeout': 600,

        # Audio file settings with Azure speech service
        "speech_service_endpoint": "https://eastus.api.cognitive.microsoft.com",
        "speech_service_location": "eastus",
        "speech_service_locale": "en-US",
        "speech_service_key": ""
    }

    try:
        # Attempt to read the existing doc
        settings_item = cosmos_settings_container.read_item(
            item="app_settings",
            partition_key="app_settings"
        )
        #print("Successfully retrieved settings from Cosmos DB.")

        # Merge default_settings in, to fill in any missing or nested keys
        merged = deep_merge_dicts(default_settings, settings_item)

        # If merging added anything new, upsert back to Cosmos so future reads remain up to date
        if merged != settings_item:
            cosmos_settings_container.upsert_item(merged)
            print("App Settings had missing keys and was updated in Cosmos DB.")
            return merged
        else:
            # If merged is unchanged, no new keys needed
            return merged

    except CosmosResourceNotFoundError:
        # If there's no doc, create it from scratch:
        cosmos_settings_container.create_item(body=default_settings)
        print("Default settings created in Cosmos and returned.")
        return default_settings

    except Exception as e:
        print(f"Error retrieving settings: {str(e)}")
        return None


def update_settings(new_settings):
    try:
        # always fetch the latest settings doc, which includes your merges
        settings_item = get_settings()
        settings_item.update(new_settings)
        cosmos_settings_container.upsert_item(settings_item)
        print("Settings updated successfully.")
        return True
    except Exception as e:
        print(f"Error updating settings: {str(e)}")
        return False

def compare_versions(v1_str, v2_str):
    """
    Manually compares two version strings (e.g., "1.0.0", "1.1").
    Returns:
        1 if v1 > v2
       -1 if v1 < v2
        0 if v1 == v2
       None if parsing fails or formats are invalid.
    """
    if not v1_str or not v2_str:
        return None # Cannot compare empty strings

    # Basic cleanup (remove potential 'v' prefix and whitespace)
    v1_str = v1_str.strip().lstrip('vV')
    v2_str = v2_str.strip().lstrip('vV')

    try:
        # Use regex to ensure parts are only digits before converting
        if not re.match(r'^\d+(\.\d+)*$', v1_str) or not re.match(r'^\d+(\.\d+)*$', v2_str):
             raise ValueError("Invalid characters in version string")
        v1_parts = [int(part) for part in v1_str.split('.')]
        v2_parts = [int(part) for part in v2_str.split('.')]
    except ValueError:
        # Handle cases where parts are not integers or contain invalid chars
        print(f"Invalid version format encountered: '{v1_str}' or '{v2_str}'")
        return None

    # Compare parts element by element
    len_v1 = len(v1_parts)
    len_v2 = len(v2_parts)
    max_len = max(len_v1, len_v2)

    for i in range(max_len):
        part1 = v1_parts[i] if i < len_v1 else 0 # Treat missing parts as 0
        part2 = v2_parts[i] if i < len_v2 else 0

        if part1 > part2:
            return 1
        if part1 < part2:
            return -1

    # If all compared parts are equal, they are the same version
    return 0

def extract_latest_version_from_html(html_content):
    """
    Parses HTML content (expected from GitHub releases page) to find the latest version tag.

    Args:
        html_content (str): The HTML content as a string.

    Returns:
        str: The latest version string (e.g., "0.203.16") found, or None if no
             valid versions are found or an error occurs.
    """
    if not html_content:
        print("HTML content is empty.")
        return None

    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        versions_found = set() # Use a set to store unique version strings

        # Find all <a> tags which are likely candidates for version tags
        # Looking for links with '/releases/tag/v' in href seems most reliable
        links = soup.find_all('a', href=True)

        for link in links:
            href = link.get('href')
            # Check if the link points to a release tag URL
            if href and '/releases/tag/v' in href:
                try:
                    # Extract the part after '/tag/' which should be like 'vX.Y.Z'
                    tag_part = href.split('/tag/')[-1]
                    # Ensure it starts with 'v' and has content after 'v'
                    if tag_part.startswith('v') and len(tag_part) > 1:
                        version_str = tag_part[1:] # Remove the leading 'v'
                        # Validate the format (digits and dots only) using regex
                        if re.match(r'^\d+(\.\d+)*$', version_str):
                            versions_found.add(version_str)
                        # else:
                        #     print(f"Skipping invalid version format from href '{href}': '{version_str}'")

                except (IndexError, ValueError):
                    # Ignore links where splitting or processing fails
                    # print(f"Could not process href: {href}")
                    continue # Skip to the next link

        if not versions_found:
            print("No valid version tags found in HTML matching the pattern.")
            return None

        # Now compare the found versions to find the latest
        latest_version = None
        for current_version in versions_found:
            if latest_version is None:
                latest_version = current_version
                # print(f"Initial latest version set to: {latest_version}")
            else:
                # print(f"Comparing '{current_version}' with current latest '{latest_version}'")
                comparison_result = compare_versions(current_version, latest_version)

                if comparison_result == 1: # current_version > latest_version
                    # print(f"  -> New latest version: {current_version}")
                    latest_version = current_version
                elif comparison_result is None:
                     # Log if comparison fails, but continue trying others
                     print(f"Warning: Could not compare version '{current_version}' with '{latest_version}'. Skipping this comparison.")
                # else: comparison is -1 or 0, keep existing latest_version
                #     print(f"  -> '{latest_version}' remains latest.")


        print(f"Latest version identified from HTML: {latest_version}")
        return latest_version

    except Exception as e:
        print(f"Error parsing HTML or finding latest version: {e}")
        return None
    
def deep_merge_dicts(default_dict, existing_dict):
    for k, default_val in default_dict.items():
        if k not in existing_dict:
            existing_dict[k] = default_val
        else:
            existing_val = existing_dict[k]
            if isinstance(default_val, dict) and isinstance(existing_val, dict):
                deep_merge_dicts(default_val, existing_val)
            # For lists or other types, we skip overwriting.
    return existing_dict

def encrypt_key(key):
    cipher_suite = Fernet(app.config['SECRET_KEY'])
    encrypted_key = cipher_suite.encrypt(key.encode())
    return encrypted_key.decode()

def decrypt_key(encrypted_key):
    cipher_suite = Fernet(app.config['SECRET_KEY'])
    try:
        encrypted_key_bytes = base64.urlsafe_b64decode(encrypted_key.encode())
        decrypted_key = cipher_suite.decrypt(encrypted_key_bytes).decode()
        return decrypted_key
    except InvalidToken:
        print("Decryption failed: Invalid token")
        return None

def get_user_settings(user_id):
    """Fetches the user settings document from Cosmos DB."""
    try:
        doc = cosmos_user_settings_container.read_item(item=user_id, partition_key=user_id)
        # Ensure the settings key exists for consistency downstream
        if 'settings' not in doc:
            doc['settings'] = {}
        return doc
    except exceptions.CosmosResourceNotFoundError:
        # Return a default structure if the user has no settings saved yet
        return {"id": user_id, "settings": {}}
    except Exception as e:
        print(f"Error in get_user_settings for {user_id}: {e}")
        raise # Re-raise the exception to be handled by the route
    
def update_user_settings(user_id, settings_to_update):
    """
    Updates or creates user settings in Cosmos DB, merging new settings
    into the existing 'settings' sub-dictionary and updating 'lastUpdated'.

    Args:
        user_id (str): The ID of the user.
        settings_to_update (dict): A dictionary containing the specific
                                   settings key/value pairs to update.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    log_prefix = f"User settings update for {user_id}:"


    try:
        # Try to read the existing document
        try:
            doc = cosmos_user_settings_container.read_item(item=user_id, partition_key=user_id)

            # Ensure the 'settings' key exists and is a dictionary
            if 'settings' not in doc or not isinstance(doc.get('settings'), dict):
                doc['settings'] = {}


        except exceptions.CosmosResourceNotFoundError:

            # Document doesn't exist, create the basic structure
            doc = {
                "id": user_id,
                "settings": {} # Initialize the settings dictionary
                # Add any other default top-level fields if needed
            }

        # --- Merge the new settings into the 'settings' sub-dictionary ---
        doc['settings'].update(settings_to_update)

        # --- Update the timestamp ---
        # Use timezone-aware UTC time
        doc['lastUpdated'] = datetime.now(timezone.utc).isoformat()



        # Upsert the modified document
        cosmos_user_settings_container.upsert_item(body=doc) # Use body=doc for clarity


        return True

    except exceptions.CosmosHttpResponseError as e:
        print(f"{log_prefix} Cosmos DB HTTP error: {e}")

        return False
    except Exception as e:
        # Catch any other unexpected errors during the update process
        print(f"{log_prefix} Unexpected error during update: {e}")

        return False

def enabled_required(setting_key):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            settings = get_settings()
            if not settings.get(setting_key, False):
                setting_key_as_statement = setting_key.replace("_", " ").title()
                return jsonify({"error": f"{setting_key_as_statement} is disabled."}), 400
            return f(*args, **kwargs)
        return wrapper
    return decorator


def sanitize_settings_for_user(full_settings: dict) -> dict:
    keys_to_exclude = {
        'azure_document_intelligence_key',
        'azure_ai_search_key',
        'azure_openai_gpt_key',
        'azure_openai_embedding_key',
        'azure_openai_image_gen_key',
        'bing_search_key',
        'azure_apim_gpt_subscription_key',
        'azure_apim_embedding_subscription_key',
        'azure_apim_image_gen_subscription_key',
        'azure_apim_web_search_subscription_key',
        'azure_apim_ai_search_subscription_key',
        'azure_apim_document_intelligence_subscription_key',
        'redis_key',
        'azure_apim_redis_subscription_key',
        'azure_apim_content_safety_subscription_key',
        'content_safety_key',
        'office_docs_key',
        'video_files_key',
        'audio_files_key'
        # any others that are secrets
    }
    return {k:v for k,v in full_settings.items() if k not in keys_to_exclude}
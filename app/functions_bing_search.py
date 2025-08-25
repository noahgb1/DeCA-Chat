# functions_bing_search.py

from config import *
from functions_settings import *

def get_suggestions(query):
    settings = get_settings()
    if not settings.get('enable_web_search'):
        return []

    bing_key = settings.get('bing_search_key', '')
    if not bing_key:
        return []

    autosuggest_url = f"{bing_search_endpoint}/v7.0/suggestions"
    headers = {"Ocp-Apim-Subscription-Key": bing_key}
    params = {"q": query}
    response = requests.get(autosuggest_url, headers=headers, params=params)
    response.raise_for_status()
    suggestions = response.json()["suggestionGroups"][0]["searchSuggestions"]
    return [s["displayText"] for s in suggestions]

def get_search_results(query, top_n=10):
    settings = get_settings()
    if not settings.get('enable_web_search'):
        return []

    bing_key = settings.get('bing_search_key', '')
    if not bing_key:
        return []

    search_url = f"{bing_search_endpoint}/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_key}
    params = {"q": query, "count": top_n}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json().get("webPages", {}).get("value", [])
    return [{"name": r["name"], "url": r["url"], "snippet": r["snippet"]} for r in results]


def process_query_with_bing_and_llm(user_query, top_n=10):
    print(f"Original Query: {user_query}")
    suggestions = get_suggestions(user_query)
    if suggestions:
        refined_query = suggestions[0]
        print(f"Refined Query (from Autosuggest): {refined_query}")
    else:
        refined_query = user_query
        print("No suggestions available. Using the original query.")

    search_results = get_search_results(refined_query, top_n=top_n)
    print(f"Search Results: {search_results}")

    return search_results
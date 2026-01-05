from utils.decorators import tool
from ddgs import DDGS
import flyte
from typing import Optional
import httpx
from bs4 import BeautifulSoup
from config import TAVILY_API_KEY

# ----------------------------------
# Tavily Search Tool
# ----------------------------------
@tool(agent="web_search")
@flyte.trace
async def tavily_search(
    query: str,
    max_results: int = 10,
    include_answer: bool = False,
    include_raw_content: bool = False,
    search_depth: str = "basic"
) -> dict:
    """
    Search the web using Tavily for web results.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return (default: 10).
        include_answer (bool): Include AI-generated answer to the query (default: False).
        include_raw_content (bool): Include raw HTML content (default: False).
        search_depth (str): Search depth - "basic" or "advanced" (default: "basic").

    Returns:
        dict: Dictionary with 'results' (list), 'answer' (str if include_answer=True), and 'query' (str).
    """
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        include_answer=include_answer,
        include_raw_content=include_raw_content,
        search_depth=search_depth
    )

    search_results = []
    for result in response.get("results", []):
        search_results.append({
            "title": result.get("title", ""),
            "href": result.get("url", ""),
            "body": result.get("content", "")
        })

    result_dict = {
        "query": query,
        "results": search_results
    }

    # Add answer if requested
    if include_answer and "answer" in response:
        result_dict["answer"] = response["answer"]

    print(f"[Tavily] Found {len(search_results)} results for: {query}")
    if include_answer and "answer" in result_dict:
        print(f"[Tavily] Answer: {result_dict['answer'][:100]}...")

    return result_dict


# ----------------------------------
# DuckDuckGo Search Tool
# ----------------------------------
@tool(agent="web_search")
@flyte.trace
async def duck_duck_go(
    query: str,
    max_results: int = 10,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None
) -> list[dict]:
    """
    Search DuckDuckGo for web results.

    Args:
        query (str): The search query.
        max_results (int): Maximum number of results to return (default: 10).
        region (str): Region code for search results (default: "us-en").
            Examples: "us-en", "uk-en", "de-de", "fr-fr"
        safesearch (str): Safe search level (default: "moderate").
            Options: "off", "moderate", "strict"
        timelimit (str, optional): Time limit for results (default: None for all time).
            Options: "d" (day), "w" (week), "m" (month), "y" (year)

    Returns:
        list[dict]: List of search results with 'title', 'href', and 'body' keys.
    """
    ddgs = DDGS()

    results = ddgs.text(
        query=query,
        region=region,
        safesearch=safesearch,
        timelimit=timelimit,
        max_results=max_results
    )

    # Convert generator to list and extract relevant fields
    search_results = []
    for result in results:
        search_results.append({
            "title": result.get("title", ""),
            "href": result.get("href", ""),
            "body": result.get("body", "")
        })

    print(f"[DuckDuckGo] Found {len(search_results)} results for: {query}")
    return search_results

# ----------------------------------
# Webpage Fetching Tool
# ----------------------------------
@tool(agent="web_search")
@flyte.trace
async def fetch_webpage(url: str, max_length: int = 5000) -> dict:
    """
    Fetch and extract text content from a webpage.

    Args:
        url (str): The URL of the webpage to fetch.
        max_length (int): Maximum length of content to return (default: 5000 chars).

    Returns:
        dict: Dictionary with 'url', 'title', 'content', and 'error' keys.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text content
        title = soup.title.string if soup.title else "No title"
        text = soup.get_text(separator='\n', strip=True)

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "... [truncated]"

        print(f"[Fetch Webpage] Successfully fetched {len(text)} chars from: {url}")

        return {
            "url": url,
            "title": title,
            "content": text,
            "error": ""
        }

    except httpx.TimeoutException:
        error_msg = f"Timeout fetching {url}"
        print(f"[Fetch Webpage] {error_msg}")
        return {"url": url, "title": "", "content": "", "error": error_msg}

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code} error for {url}"
        print(f"[Fetch Webpage] {error_msg}")
        return {"url": url, "title": "", "content": "", "error": error_msg}

    except Exception as e:
        error_msg = f"Error fetching {url}: {str(e)}"
        print(f"[Fetch Webpage] {error_msg}")
        return {"url": url, "title": "", "content": "", "error": error_msg}
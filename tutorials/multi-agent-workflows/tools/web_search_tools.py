from utils.decorators import tool
from ddgs import DDGS
import flyte
from typing import Optional, Any
import httpx
from bs4 import BeautifulSoup
# from config import TAVILY_API_KEY

# Type aliases for clarity (Python 3.9+ style)
SearchResult = dict[str, str]  # Dict with 'title', 'href', 'body' keys
WebpageResult = dict[str, str]  # Dict with 'url', 'title', 'content', 'error' keys
TavilySearchResult = dict[str, Any]  # Dict with 'query' (str), 'results' (list), optional 'answer' (str)


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
) -> list[SearchResult]:
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
        list[SearchResult]: List of dicts with 'title', 'href', and 'body' keys.
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
async def fetch_webpage(url: str, max_length: int = 5000) -> WebpageResult:
    """
    Fetch and extract text content from a webpage.

    Args:
        url: The URL of the webpage to fetch.
        max_length: Maximum length of content to return (default: 5000 chars).

    Returns:
        WebpageResult: Dict with 'url', 'title', 'content', and 'error' keys.
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
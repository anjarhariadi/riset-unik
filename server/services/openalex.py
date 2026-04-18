import httpx
import urllib.parse

from config import config


async def search_openalex(topic: str, max_results=40):
    api_key = config.get_settings().openalex_api_key
    try:
        query = urllib.parse.quote(topic)
        url = f"https://api.openalex.org/works?search={query}&per_page={max_results}&api_key={api_key}"
        
        async with httpx.AsyncClient() as client:
            client.headers.update({"User-Agent": "Mozilla/5.0"})
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()

        results = []
        for item in data.get("results", [])[:max_results]:
            title = item.get("title", "No Title")
            
            doi = item.get("doi")
            if doi:
                link = f"https://doi.org/{doi}"
            else:
                openalex_id = item.get("id", "#")
                link = f"https://openalex.org/{openalex_id.split('/')[-1]}"
            
            results.append({"title": title, "link": link})

        return results

    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        print(f"HTTP error occurred: {e}")
        return []
    except ValueError as e:
        print(f"JSON decoding failed: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
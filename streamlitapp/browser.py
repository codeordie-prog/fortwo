import asyncio
import time
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from duckduckgo_search.exceptions import RatelimitException

# Set the event loop policy for Windows, if needed
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Create an instance of DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()

def perform_search(query):
    try:
        # Invoke the search
        response = search.invoke(query)
        return response
    except RatelimitException:
        time.sleep(60)  # Wait for 60 seconds before retrying
        return perform_search(query)  # Retry the search

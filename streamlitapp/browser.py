
import time
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from duckduckgo_search.exceptions import RatelimitException
from langchain_openai import ChatOpenAI

# Set the event loop policy for Windows, if needed
#asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Create an instance of DuckDuckGoSearchRun
search = DuckDuckGoSearchRun()


def query_prompt(query,api):
    llm = ChatOpenAI(api_key=api,model="gpt-4o")
    query_string = f"Given the following query by the user{query}, you should reply directly without including anything else, the most detailed efficient search prompt for a browser, that will definitely and accurately if send to the browser returns a reliable response"
    resp = llm.invoke(query_string)
    

    return resp.content


def perform_search(query):
    try:
        # Invoke the search
        response = search.invoke(query)
        return response
    except RatelimitException:
        time.sleep(60)  # Wait for 60 seconds before retrying
        return perform_search(query)  # Retry the search



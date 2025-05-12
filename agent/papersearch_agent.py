import autogen
from tools.paper_search_tool import search_papers
from config import LLM_CONFIG

paper_search_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Searches for research papers based on topic, publication year (and operator: 'in', 'before', 'after'), and minimum citations.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The research topic to search for."},
                "year": {"type": "integer", "description": "The target year for publication (e.g., 2020)."},
                "year_operator": {
                    "type": "string",
                    "description": "How to compare the publication year: 'in' (exact year), 'before' (e.g., year < 2020), 'after' (e.g., year > 2020).",
                    "enum": ["in", "before", "after"]
                },
                "min_citations": {"type": "integer", "description": "The minimum number of citations a paper should have."},
                "limit": {
                    "type": "integer",       
                    "description": "Maximum number of papers to return, defaults to 5 if not specified.",
                    "default": 5
                }
            },
            "required": ["topic"],
        },
    },
}

def create_paper_search_agent() -> autogen.ConversableAgent:
    current_llm_config = LLM_CONFIG.copy() 
    current_llm_config["tools"] = [paper_search_tool_schema]
    agent = autogen.ConversableAgent(
        name="Paper Search Agent",
        system_message="You are a helpful AI assistant that finds research papers. "
                       "The user will ask you to find papers using a phrase like: "
                       "'Find a research paper on [topic] that was published [in/before/after] [year] and has [number of citations] citations.' "
                       "Extract the topic, year, year operator, and minimum citations from the user's request "
                       "and use the 'search_papers' tool. "
                       "If 'limit' is not specified by the user, default to 5 results. "
                       "Present the found papers clearly, including title, authors, year, citations, and URL. "
                       "After providing all information and a list of papers (if any are found), or stating that no papers were found, end your response with the word TERMINATE.",
        llm_config=current_llm_config,
    )
    agent.register_for_llm(name="search_papers", description="Search for research papers.")(search_papers)
    return agent

def create_user_proxy_agent() -> autogen.UserProxyAgent:
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
        code_execution_config=False,
        llm_config=False,
    )
    user_proxy.register_for_execution(name="search_papers")(search_papers)
    return user_proxy

def main():
    paper_search_agent = create_paper_search_agent()
    user_proxy = create_user_proxy_agent()

    while True:
        task = input("Please enter your research paper query (or type 'quit' to exit): ")
        if task.strip().lower() == 'quit':
            break
        
        if not task.strip():
            print("No input received, please try again.")
            continue
        
        user_proxy.initiate_chat(
            recipient=paper_search_agent,
            message=task,
        )
        
        user_proxy.reset()
        paper_search_agent.reset()

if __name__ == "__main__":
    main()
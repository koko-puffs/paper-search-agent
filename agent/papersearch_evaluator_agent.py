import autogen # type: ignore
import json
import time
from agent.papersearch_agent import create_paper_search_agent, create_user_proxy_agent
from config import LLM_CONFIG

def main():
    paper_search_agent = create_paper_search_agent()
    user_proxy = create_user_proxy_agent()

    critic_agent = autogen.AssistantAgent(
        name="critic_agent",
        llm_config=LLM_CONFIG,
        system_message="You are an AI assistant that evaluates the responses of a paper search agent. Your evaluation should be based on the following criteria: Completeness, Quality, Robustness, Consistency, and Specificity. Provide your evaluation in JSON format."
    )

    prompts = [
        "Find research papers on 'machine learning in healthcare' published after 2020 with at least 50 citations.",
        "Search for articles about 'quantum computing algorithms' published in 2022.",
        "Show me papers on 'renewable energy sources' with a focus on solar power, minimum 20 citations.",
        "I need some good AI papers.",
        "Find me something about new tech.",
        "Latest developments in science.",
        "Find papers on 'the impact of social media on adolescent mental health' published before 2019, authored by 'Dr. Jane Doe' or 'Dr. John Smith', with over 100 citations.",
        "I'm looking for review articles on 'CRISPR gene editing applications in cancer therapy', preferably published in high-impact journals in the last 3 years.",
        "Find papers on 'time travel to the Jurassic period'.",
        "Search for research published tomorrow.",
        "Papers about 'the best programming language' with exactly 7 citations.",
        ""
    ]

    for prompt in prompts:
        critic_prompt = """
            You are evaluating an AI paper search agent.
        
            Evaluate the agent's response based on these criteria, using a scale of 1-5 (1=Poor, 5=Excellent):
            - Completeness: Did the agent fully address every aspect of the user's prompt? (e.g., topic, year, citations, number of papers)
            - Quality: Was the response accurate (correct papers, correct details), clear, well-organized, and easy to understand?
            - Robustness: How well did the agent handle ambiguous, incorrect, or challenging inputs? (e.g., did it ask for clarification, or state inability if appropriate?)
            - Consistency: If multiple constraints were given, are the results consistent with all of them?
            - Specificity: Did the agent offer precise details (title, authors, year, citations, URL) for each paper?

            Additionally:
            - Check if the agent clearly stated when no papers were found or if parameters were ignored.
            - Assess if the agent interpreted ambiguous prompts reasonably or asked for clarification.
            - Note if the agent defaulted to a reasonable number of results if not specified.

            User Prompt: {prompt}
            Agent Response: {agent_response}

            Provide your evaluation as a JSON object with the following fields:
            - completeness (integer, 1-5)
            - quality (integer, 1-5)
            - robustness (integer, 1-5)
            - consistency (integer, 1-5)
            - specificity (integer, 1-5)
            - feedback (string, a brief descriptive explanation of the ratings, including specific examples or issues from the response. Note if the agent TERMINATED as expected.)
        """
        
        user_proxy.initiate_chat(
            recipient=paper_search_agent,
            message=prompt,
            max_turns=3 
        )
        
        agent_response = "Agent did not provide a response."
        if paper_search_agent in user_proxy.chat_messages and user_proxy.chat_messages[paper_search_agent]:
            last_message = user_proxy.chat_messages[paper_search_agent][-1]
            if isinstance(last_message, dict) and "content" in last_message:
                agent_response = last_message["content"]
            elif isinstance(last_message, str):
                agent_response = last_message

        critic_prompt_filled = critic_prompt.format(
            prompt=prompt,
            agent_response=str(agent_response) 
        )

        critic_reply = critic_agent.generate_reply(
            messages=[{"role": "user", "content": critic_prompt_filled}]
        )

        json_string_to_parse = ""
        if isinstance(critic_reply, dict) and 'content' in critic_reply:
            json_string_to_parse = critic_reply['content']
        elif isinstance(critic_reply, str):
            json_string_to_parse = critic_reply
        else:
            json_string_to_parse = str(critic_reply) 

        if json_string_to_parse.strip().startswith("```json"):
            json_string_to_parse = json_string_to_parse.strip()[7:]
            if json_string_to_parse.endswith("```"):
                json_string_to_parse = json_string_to_parse[:-3]
        
        json_string_to_parse = json_string_to_parse.strip()
        
        result = {}
        if not json_string_to_parse:
            print("ERROR: json_string_to_parse is empty after processing. Critic did not provide valid JSON content.")
            result = {"error": "Critic did not provide valid JSON content / empty after processing."}
        else:
            try:
                result = json.loads(json_string_to_parse)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON from critic: {e}")
                print(f"ERROR: Offending string was: '{json_string_to_parse}'")
                result = {"error": "JSONDecodeError", "details": str(e), "original_string": json_string_to_parse}
        
        print(f"Prompt: {prompt}\nAgent Response: {agent_response}\nCritic Evaluation: {json.dumps(result, indent=2)}\n")

        user_proxy.reset()
        paper_search_agent.reset()
        
        print(f"--- Waiting for 5 seconds before next prompt ---")
        time.sleep(5)

if __name__ == "__main__":
    main()
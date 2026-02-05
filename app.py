import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

load_dotenv()

st.title("LangChain - Search Agent with Modern API")

groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", help="Your Groq API key here")

# Ensure the model is configured
if not groq_api_key:
    st.warning("Enter Groq API Key in sidebar to start the agent.")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

# Define the search tool
@tool
def duckduckgo_search(query: str) -> str:
    """Search the web for the query and return a summary."""
    search_tool = DuckDuckGoSearchRun(name="duckduckgo_search")
    return search_tool.run(query)

# Define ArXiv tool
@tool
def arxiv_query(query: str) -> str:
    """Search Arxiv for research papers"""
    wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    arxiv_runner = ArxivQueryRun(api_wrapper=wrapper)
    return arxiv_runner.run(query)

# Define Wikipedia tool
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for the query and return summary."""
    wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki_runner = WikipediaQueryRun(api_wrapper=wrapper)
    return wiki_runner.run(query)

tools = [duckduckgo_search, arxiv_query, wikipedia_search]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are an intelligent assistant that can search the web, scholarly papers, and encyclopedia. "
        "Use only the provided tools to answer queries."
    )
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask something…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        # Create a container for streaming agent steps
        callback_container = st.container()

        st_callback = StreamlitCallbackHandler(
            callback_container,
            expand_new_thoughts=True
        )

        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": user_input}]
            },
            config={
                "callbacks": [st_callback]
            }
        )

        last_msg = result["messages"][-1]
        reply = last_msg.content

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )
        st.write(reply)


from langchain_groq import ChatGroq
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

from dotenv import load_dotenv
import os
load_dotenv()

# Load API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the AI agent
tech_agent = ChatGroq(
    name="Technical Agent",
    api_key=groq_api_key,
    model='gemma2-9b-it',
)

# Define search tool
search = DuckDuckGoSearchRun()
tools = [search]

# Define prompt template
prompt_for_tech_agent = '''You are a cutting-edge AI assistant specializing in the latest technology trends. 
You have access to real-time internet search capabilities and can retrieve up-to-date information about advancements in {tech_interests}. 
You provide insightful, engaging, and well-structured responses in a conversational manner. 
You analyze search results and summarize them concisely for the user, avoiding outdated or irrelevant data. 
If no relevant search results are found, you rely on your existing knowledge and provide general trends. 
You maintain a professional yet engaging tone, and you can also discuss the impact of these trends on various industries. 
You ask follow-up questions to keep the conversation engaging and ensure the user gets the most relevant insights.'''

prompt_template = PromptTemplate(
    input_variables=["tech_interests"],
    template=prompt_for_tech_agent
)

# Streamlit UI
st.set_page_config(page_title="Agents' GC", page_icon='🤖')
st.title("AI Agents Groupchat")

tech_interests = ['Generative AI', 'Web Development', 'Machine Learning']
tech_int = st.selectbox(label="Select your Tech Interest", placeholder="Choose an option", options=tech_interests)

# Format the prompt with user-selected interest
formatted_prompt = prompt_template.format(tech_interests=tech_int)

# Initialize the agent
tech_agent_with_search = initialize_agent(
    tools=tools,
    llm=tech_agent,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Run the agent with the formatted prompt
response = tech_agent_with_search.invoke(formatted_prompt)

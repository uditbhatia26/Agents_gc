from langchain_groq import ChatGroq
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit Page Configuration
st.set_page_config(page_title="AI Agents Groupchat", page_icon='🤖', layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .big-title { font-size: 2.5rem; font-weight: bold; text-align: center; color: #4CAF50; }
        .stButton>button { width: 100%; font-size: 1.2rem; padding: 10px; }
        .stTextInput>div>div>input { font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">🤖 AI Agents Groupchat</p>', unsafe_allow_html=True)

# Sidebar for user preferences
with st.sidebar:
    st.header("📌 Select Your Interests")
    
    tech_interests = ['Generative AI', 'Web Development', 'Machine Learning', 'Cybersecurity', 'Blockchain']
    tech_int = st.selectbox("💻 Select Your Tech Interest", tech_interests)

    st.markdown("---")  # Divider
    st.info("🔍 Click 'Ask Agent' to explore the latest tech trends!", icon="ℹ️")

# Define AI Agent
tech_agent = ChatGroq(
    name="Technical Agent",
    api_key=groq_api_key,
    model='gemma2-9b-it',
)

# Define search tool
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["tech_interests"],
    template="""
        You are a cutting-edge AI assistant specializing in the latest technology trends.
        Your role is to provide up-to-date insights, analyze trends, and summarize findings for the user.
        The user is interested in {tech_interests}.
        Retrieve and summarize the latest updates from the internet and provide actionable insights.
    """
)

# Initialize agent
tech_agent_with_search = initialize_agent(
    tools=tools,
    llm=tech_agent,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Button to trigger response
if st.button("🚀 Ask Agent"):
    st.subheader(f"📢 Exploring {tech_int} Trends...")
    
    # Generate response
    formatted_prompt = prompt_template.format(tech_interests=tech_int)
    response = tech_agent_with_search.invoke(formatted_prompt)

    # Display the result in an expandable container
    with st.expander("🔎 Click to view AI Insights", expanded=True):
        st.write(response['output'])

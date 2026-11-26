import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# Page configuration
st.set_page_config(page_title="GlobeTrotter AI - Smart Travel Guide", page_icon="✈️", layout="centered")

st.title("🌍 GlobeTrotter AI: Smart Itinerary & Travel Guide")
st.markdown("Your personal AI travel companion. Upload a destination guide and plan your dream trip!")

# Sidebar for API Key & Trip Settings
st.sidebar.header("⚙️ Configuration")
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

st.sidebar.markdown("---")
trip_style = st.sidebar.selectbox("Select Travel Style", ["Balanced & Cultural", "Foodie & Culinary", "Budget Backpacker", "Luxury & Relaxation"])
trip_duration = st.sidebar.slider("Trip Duration (Days)", 1, 7, 3)

CHROMA_PATH = "chroma_db"

if not api_key:
    st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to get started.")
else:
    try:
        # Initialize Gemini for chat generation
        llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2, google_api_key=api_key)
        
        # Initialize local Hugging Face embeddings for vector search
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # User query input
        query = st.text_input("What would you like to know about your destination?", placeholder="e.g., Famous places to visit in Paris")
        
        if query:
            with st.spinner("✈️ Scouting the location and building your guide..."):
                # 1. Retrieve from Chroma DB locally
                db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                docs = db.similarity_search(query, k=3)
                context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

                # 2. Generate Travel Assistant Prompt
                prompt = f"""You are an expert global travel planner and local tour guide. 
Use the given travel document context to answer the user's question. 
Tailor your recommendations to a '{trip_style}' style for a {trip_duration}-day trip.
If the answer is not in the context, provide a helpful general travel recommendation while noting it.

Context:
{context_text}

Traveler Question:
{query}
"""
                ai_response = llm.invoke(prompt)
                content_text = ai_response.content
                if isinstance(content_text, list):
                    clean_text = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in content_text])
                else:
                    clean_text = str(content_text)

                # Display Results in a clean card layout
                st.markdown("### 🗺️ Your Personalized Travel Guide")
                st.success(clean_text)

                with st.expander("🔍 View Source Reference Chunks"):
                    st.write(context_text)

    except Exception as e:
        st.error(f"An error occurred: {e}")
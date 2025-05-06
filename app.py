import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime

# --- Configuration ---

# Set page config (do this FIRST)
st.set_page_config(
    page_title="Text-to-Solution Generator",
    page_icon="🧠",
    layout="wide"
)

# --- Firebase Initialization ---
# Initialize Firebase only once
@st.cache_resource # Use st.cache_resource for resources like DB connections
def init_firebase():
    """Initializes Firebase Admin SDK"""
    try:
        # Check if Firebase app is already initialized
        if not firebase_admin._apps:
            # Try to load credentials from Streamlit secrets first (for deployment)
            firebase_creds_dict = st.secrets.get("firebase_credentials")
            if firebase_creds_dict:
                cred = credentials.Certificate(dict(firebase_creds_dict))
                st.success("Firebase initialized from Streamlit secrets.")
            # Fallback to local firebase_config.json (for local development)
            elif os.path.exists("firebase_config.json"):
                cred = credentials.Certificate("firebase_config.json")
                st.success("Firebase initialized from local firebase_config.json.")
            else:
                st.error("Firebase configuration not found. "
                         "Please ensure 'firebase_config.json' is present or "
                         "'firebase_credentials' are set in Streamlit secrets.")
                return None, None
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        return db, True # Return db client and success status
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")
        return None, False

db, firebase_initialized = init_firebase()

# --- Gemini API Configuration ---
@st.cache_data # Cache the API key loading
def get_gemini_api_key():
    """Retrieves Gemini API key from Streamlit secrets."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key:
            st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
            return None
        return api_key
    except KeyError:
        st.error("GEMINI_API_KEY not found in Streamlit secrets. Please add it.")
        return None
    except Exception as e:
        st.error(f"Error accessing Gemini API key: {e}")
        return None

GEMINI_API_KEY = get_gemini_api_key()

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash') # Or your preferred Gemini model
        st.sidebar.success("Gemini API configured successfully!")
    except Exception as e:
        st.sidebar.error(f"Error configuring Gemini API: {e}")
        model = None
else:
    st.sidebar.warning("Gemini API key not configured. Solution generation will not work.")
    model = None


# --- Firestore Functions ---

def save_qa_to_firestore(question: str, solution: str):
    """Saves the question and solution to Firestore."""
    if not firebase_initialized or not db:
        st.error("Firebase not initialized. Cannot save data.")
        return False
    if not question or not solution:
        st.warning("Question or solution is empty. Nothing to save.")
        return False
    try:
        qa_collection = db.collection('q_and_a')
        qa_collection.add({
            'question': question,
            'solution': solution,
            'timestamp': firestore.SERVER_TIMESTAMP  # Use server timestamp
        })
        return True
    except Exception as e:
        st.error(f"Error saving to Firestore: {e}")
        return False

@st.cache_data(ttl=300) # Cache Firestore data for 5 minutes
def get_past_qa_from_firestore():
    """Retrieves all past questions and solutions from Firestore, ordered by timestamp."""
    if not firebase_initialized or not db:
        st.error("Firebase not initialized. Cannot retrieve data.")
        return []
    try:
        qa_collection_ref = db.collection('q_and_a').order_by(
            'timestamp', direction=firestore.Query.DESCENDING
        )
        docs = qa_collection_ref.stream()
        past_qa = []
        for doc in docs:
            data = doc.to_dict()
            # Ensure timestamp is serializable (convert to string if it's a datetime object)
            if 'timestamp' in data and isinstance(data['timestamp'], datetime):
                data['timestamp'] = data['timestamp'].isoformat()
            past_qa.append(data)
        return past_qa
    except Exception as e:
        st.error(f"Error retrieving from Firestore: {e}")
        return []

# --- Gemini Solution Generation ---

def generate_solution(question: str):
    """Generates a step-by-step solution using the Gemini model."""
    if not model:
        return "Error: Gemini model not initialized. Please check API key and configuration."
    if not question:
        return "Please enter a question."

    prompt = f"""
    You are an expert math and physics tutor.
    Provide a clear, step-by-step solution to the following question.
    Explain each step thoroughly. If formulas are used, state them first.
    Make the solution easy for a beginner to understand.

    Question: "{question}"

    Solution:
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating solution with Gemini: {e}")
        return f"Sorry, I couldn't generate a solution at this time. Error: {e}"

# --- Streamlit UI ---

st.title("📚 Text-to-Solution Generator")
st.markdown("Enter a math or physics question below, and AI will generate a step-by-step solution!")

# --- Main Application Flow ---

# Initialize session state for storing current question and solution
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_solution' not in st.session_state:
    st.session_state.current_solution = ""
if 'solution_generated' not in st.session_state:
    st.session_state.solution_generated = False

# --- Input Section ---
with st.container(border=True):
    st.subheader("❓ Ask a New Question")
    user_question = st.text_area(
        "Enter your math or physics question here:",
        value=st.session_state.current_question,
        height=100,
        key="question_input"
    )

    if st.button("✨ Generate Solution", type="primary", use_container_width=True):
        if not GEMINI_API_KEY:
            st.error("Cannot generate solution: Gemini API Key is not configured.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state.current_question = user_question
            with st.spinner("🔍 Generating solution... Please wait."):
                solution = generate_solution(user_question)
                st.session_state.current_solution = solution
                st.session_state.solution_generated = True

                # Save to Firebase only if solution generation was successful (basic check)
                if solution and not solution.startswith("Sorry, I couldn't generate a solution"):
                    if firebase_initialized:
                        if save_qa_to_firestore(user_question, solution):
                            st.success("Question and solution saved to history!")
                            # Clear cache for past Q&A to refresh the history section
                            st.cache_data.clear()
                        else:
                            st.error("Failed to save question and solution to history.")
                    else:
                        st.warning("Firebase not initialized. Solution not saved to history.")
                else:
                    st.error("Solution generation failed. Not saved to history.")


# --- Display Current Solution ---
if st.session_state.solution_generated:
    with st.container(border=True):
        st.subheader("💡 Generated Solution")
        st.markdown(f"**Question:** {st.session_state.current_question}")
        st.markdown("**Answer:**")
        st.markdown(st.session_state.current_solution, unsafe_allow_html=True) # Allow HTML for better formatting if Gemini provides it

# --- History Section ---
if firebase_initialized: # Only show history if Firebase is working
    st.divider()
    st.subheader("📜 Past Questions and Solutions")

    past_qas = get_past_qa_from_firestore()

    if not past_qas:
        st.info("No past questions and solutions found in the history.")
    else:
        # Pagination for history
        items_per_page = 5
        total_items = len(past_qas)
        total_pages = (total_items + items_per_page - 1) // items_per_page

        if 'history_page' not in st.session_state:
            st.session_state.history_page = 1

        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.history_page > 1:
                    st.session_state.history_page -= 1
        with col2:
            st.write(f"Page {st.session_state.history_page} of {total_pages}")
        with col3:
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.history_page < total_pages:
                    st.session_state.history_page += 1

        start_idx = (st.session_state.history_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_page_qas = past_qas[start_idx:end_idx]


        if not current_page_qas and st.session_state.history_page > 1: # If on an empty page (e.g. after deleting items)
            st.session_state.history_page = 1 # Go back to first page
            # Rerun to reflect change immediately
            st.rerun()


        for i, qa in enumerate(current_page_qas):
            with st.expander(f"**Q: {qa.get('question', 'N/A')}** (Asked on: {qa.get('timestamp', 'N/A')[:10] if qa.get('timestamp') else 'N/A'})"):
                st.markdown("**Solution:**")
                st.markdown(qa.get('solution', 'No solution stored.'), unsafe_allow_html=True)
else:
    st.sidebar.warning("History section disabled as Firebase is not initialized.")


# --- Sidebar for Status ---
st.sidebar.title("Status")
if firebase_initialized:
    st.sidebar.success("Firebase Connected")
else:
    st.sidebar.error("Firebase Disconnected")

if GEMINI_API_KEY and model:
    st.sidebar.success("Gemini Model Loaded")
else:
    st.sidebar.error("Gemini Model Not Loaded")

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses Gemini for solution generation and Firebase Firestore for storing history. "
    "Ensure your API keys and Firebase configuration are set up correctly."
)
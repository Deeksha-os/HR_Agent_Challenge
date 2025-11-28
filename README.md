🤖 HR Policy Assistant AI Agent

Overview of the Agent

This project implements a fully functional HR Assistant Agent using Retrieval-Augmented Generation (RAG) within a Streamlit interface. Its primary purpose is to provide employees with accurate, grounded answers to common policy, leave, and benefits queries by consulting a private repository of HR documents. It also incorporates a critical guardrail to ensure sensitive issues are escalated to human HR personnel.

Features & Limitations

Features

RAG-Based Grounding: Answers are sourced directly from policy documents using ChromaDB and HuggingFace embeddings, preventing LLM hallucination.

High-Risk Query Escalation (Guardrail): Queries containing sensitive keywords (e.g., 'harassment', 'complaint', 'legal action') are blocked and redirected to human HR support.

Source Citation: The agent cites the specific policy documents used to construct the answer, enhancing trust and auditability.

Live Chat Interface: Provides a user-friendly, real-time chat experience via Streamlit.

Limitations

No Conversational Memory: The agent treats each query as a new request; it cannot remember past interactions within the session.

Document Format: Currently only supports ingestion of plain text (.txt) files.

External API Dependence: Requires an active Gemini API key to function.

⚙️ Tech Stack & APIs Used

Component

Technology

Role

User Interface

Python / Streamlit

Front-end web application for the chat interface.

LLM

Google Gemini 2.5 Flash API

Generative model for synthesizing policy-based answers.

Framework

LangChain (Core & Community)

Orchestrates the RAG pipeline.

Vector Database

ChromaDB

Local, persistent storage for document embeddings.

Embeddings

HuggingFace all-MiniLM-L6-v2

Converts policy text into numerical vectors for search.

🧪 Testing Scenarios (Crucial for Evaluation)

To ensure the agent is working correctly and meets all challenge requirements, please use the following two types of queries:

1. RAG Success Test (Policy Answering)

These questions test the agent's ability to retrieve information from the HR_Policy_Docs folder and generate a detailed answer.

Query Type

Example Query

Expected Outcome

Leave Policy

"How many vacation days do I get a year?"

The agent should return a detailed, conversational answer, citing Vacation_Policy.txt and Sick_Leave_Policy.txt (or whichever files it uses).

Sick Leave

"What is the process for calling out sick?"

The agent should explain the notification procedure (e.g., call manager, email) citing the relevant policy document.

2. Guardrail Success Test (Escalation)

This test proves the mandatory escalation feature is working. The agent must not attempt to answer these questions.

Query Type

Example Query

Expected Outcome

Formal Complaint

"I need to file a formal complaint about my manager."

The agent must show the escalation message (as defined in app.py) and immediately stop, referring the user to human HR.

Harassment

"I have experienced harassment in the workplace, what do I do?"

The agent must show the escalation message and refuse to process the query further.

🚀 Setup and Run Instructions

Prerequisites

Python 3.8+

A valid Gemini API Key.

The project structure must include a folder named HR_Policy_Docs containing your .txt policy files.

Step 1: Clone the Repository and Set up Environment

# Clone your repository
git clone <YOUR_REPO_LINK>
cd HR_Agent_Challenge
# Create and activate the virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate


Step 2: Install Dependencies

pip install streamlit langchain langchain-community langchain-google-genai python-dotenv sentence-transformers


Step 3: Configure API Key

Create a file named .env in the root of your project directory and add your Gemini API key:

# .env file
GEMINI_API_KEY="AIzaSy...." 


Step 4: Run the Agent

The agent handles document ingestion on the first run.

streamlit run app.py


The application will open in your web browser, typically at http://localhost:8501.

📈 Potential Improvements

Adding Memory: Implement ConversationBufferMemory to maintain context across turns.

Document Loader Expansion: Add support for ingestion of PDF and DOCX files to increase utility.

Advanced Guardrail: Integrate LLM-based intent classification for sensitive topics instead of simple keyword matching, improving the quality of the escalation mechanism.
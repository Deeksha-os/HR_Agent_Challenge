**HR Policy Assistant Agent**

**Overview:**
The HR Policy Assistant is a Retrieval-Augmented Generation (RAG) agent built to instantly answer employee questions regarding company policies, benefits, and internal guidelines. It functions as a secure, always-on knowledge base, ensuring all answers are grounded strictly in the official policy documents provided in the HR_Policy_Docs directory. This eliminates the need for manual HR query resolution, improves employee self-service, and reduces response time.

**Agent Category:** People & HR (HR Assistant Agent)

**Features & Limitations**

**Features:**
* Grounded Generation (RAG): Answers are derived exclusively from the provided policy documents (Sick_Leave_Policy.txt, Vacation_Policy.txt). The agent is explicitly instructed to state when information is not available in the source material, preventing hallucinations.
* Chat History Awareness: The agent maintains conversational context (st.session_state.messages), allowing users to ask follow-up questions without repeating the core query.
* Source Attribution: Every response includes a list of the exact source files (e.g., Sick_Leave_Policy.txt) that were used to generate the answer, providing transparency and trust.
* High-Risk Keyword Filtering: Includes a basic guardrail that intercepts sensitive queries (e.g., "harassment," "termination") and directs the user to a human HR contact, ensuring compliance and confidentiality for critical issues.
* Persistent Vector Store: The policy data is processed once and saved to disk to ChromaDB in the chroma_db_manual_rag_v10 folder, allowing for rapid startup times on subsequent runs.

**Limitations:**
* Static Knowledge Base: The agent's knowledge is limited strictly to the .txt files present in the HR_Policy_Docs directory at the time of vector store creation.
* Document Format: Only supports plain text (.txt) files for indexing.
* Single-User Session: The chat history is maintained only within the current Streamlit session (st.session_state) and is reset if the user completely closes and re-opens the application.

**Tech Stack & APIs Used:**
| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend/UI** | Streamlit | Provides the simple, interactive web interface for the chat application. |
| **LLM** | Google Gemini 2.5 Flash | The generative AI model used to process the retrieved context and formulate the final, natural-language answer. |
| **Framework** | LangChain | Used for orchestrating the RAG components (Embedding, Document Loading, Retrieval). |
| **Vector DB** | ChromaDB | Used to store and manage the numerical representations (embeddings) of the policy documents, enabling semantic search. |
| **Embedding Model** | HuggingFace `all-MiniLM-L6-v2` | Converts policy text into dense vectors (embeddings) for efficient search and retrieval by ChromaDB. |.

**Setup & Run Instructions**

**Prerequisites:**
+ Python 3.9+
+ Git
+ A Gemini API Key (set as an environment variable or in Streamlit secrets).

**Local Setup:**
+ Clone the repository:
git clone https://github.com/Deeksha-os/HR_Agent_Challenge.git
cd HR_Agent_Challenge

+ Create a virtual environment (Recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

+ Install dependencies:
The requirements.txt file contains all necessary packages.
pip install -r requirements.txt

+ Set your API Key:
Create a file named .env in the root directory and add your key:
#.env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE" 
Note: If deploying to Streamlit Cloud, use secrets.toml instead.

+ Run the Streamlit App:
streamlit run app.py

The application will automatically open in your web browser. On the first run, it will process the documents and build the chroma_db_manual_rag_v10 folder.

**Test the Agent:** Ask a question related to the policies, such as "What is the maximum number of unused vacation days I can carry over into the next year?" to confirm the RAG chain is functioning.
          
**Potential Improvements:**
+ Support Diverse File Types: Implement document loaders for richer formats like .docx and .pdf to expand the knowledge base without manual conversion.
+ Persistent Chat History: Integrate a proper database (like Firebase or Supabase) to save chat history across sessions and users.
+ User Authentication: Add user login/authentication to restrict access to sensitive policy information.
+ Deployment Monitoring: Integrate tools to monitor LLM performance, latency, and cost in production.

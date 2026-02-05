# Search-Engine-LLM
An interactive AI-powered search agent built with **LangChain**, **Groq LLMs**, and **Streamlit**.

The app can intelligently search across:

- DuckDuckGo (web search)  
- Wikipedia (encyclopedia)  
- arXiv (research papers)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JayK327/Search-Engine-LLM.git
cd Search-Engine-LLM
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables (optional)
Create a .env file:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Running the App
Create a .env file:

```bash
streamlit run app.py
```

Then open:

http://localhost:8501


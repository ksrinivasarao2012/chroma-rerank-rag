from streamlit.proto import ClientState_pb2
import streamlit as st 
import requests
import json

# Point this to your FastAPI server
API_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="Portfolio RAG", page_icon="📚", layout="wide")
st.title("📚 Portfolio RAG Assistant")

# 1. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Sidebar: Document Upload
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf', 'PDF'])

    if st.button('Process File') and uploaded_file:
        with st.spinner("Chunking and generating embeddings..."):
            # Send file as multipart/form-data
            files = {'file' : (uploaded_file.name,uploaded_file.getvalue(),"application/pdf")}
            response = requests.post(f'{API_URL}/upload',files=files)
            
            if response.status_code == 200:
                st.success(f"Success! {response.json()['filename']} stored in vector DB.")
            else:
                st.error(f"Error: {response.text}")

# 3. Main UI: Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
    
# 4. Main UI: Query Input & API Call
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to session state & display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Fetch and render AI streaming response
    with st.chat_message('assistant'):
        # Prepare payload with query, top_k, and previous conversation history
        payload = {
            "query": prompt,
            "top_k": 3,
            "chat_history": st.session_state.messages[:-1]  # Exclude current prompt
        }
        
        try:
            response = requests.post(f'{API_URL}/query', json=payload, stream=True)
            
            if response.status_code == 200:
                citations = []
                full_text = ""
                response_placeholder = st.empty()
                
                # Read NDJSON lines as they stream in
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        chunk_type = chunk.get("type")
                        data = chunk.get("data")
                        
                        if chunk_type == "citations":
                            citations = data
                        elif chunk_type == "token":
                            full_text += data
                            response_placeholder.markdown(full_text + "▌")
                
                # Append citations footer only if the LLM found relevant information
                final_display = full_text
                if citations and "do not have enough information" not in full_text.lower():
                    final_display += "\n\n### Relevant Context\n"
                    for i, cite in enumerate(citations, 1):
                        final_display += f"**{i}. {cite['source_file']} (Page {cite['page_number']})**\n> {cite['text_snippet']}\n\n"
                
                response_placeholder.markdown(final_display)
                
                # Save assistant response to session state
                st.session_state.messages.append({'role': 'assistant', 'content': final_display})
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to communicate with backend: {e}")
                

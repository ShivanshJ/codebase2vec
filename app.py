import os
import streamlit as st
import code_chunker
import github_interface  # Ensure this is the correct import for your GitHub functions
from code_embedding import CodeEmbedding  # Import the CodeEmbedding class
from dotenv import load_dotenv

from vector_store import VectorStore, VectorNode

load_dotenv()


class CodebaseLoader:
    def __init__(self, local_dir=None, github_repo=None):
        self.local_dir = local_dir
        self.github_repo = github_repo
        self.snippets = []

    def load_codebase(self):
        if self.github_repo:
            self.snippets = github_interface.load_github_codebase(self.github_repo)
        elif self.local_dir:
            self.snippets = self.load_local_codebase(self.local_dir)
        return self.snippets

    def load_local_codebase(self, directory):
        snippets = []
        for filename in os.listdir(directory):
            if filename.startswith('.'):
                continue
            filepath = os.path.join(directory, filename)
            if os.path.isdir(filepath):
                snippets.extend(self.load_local_codebase(filepath))
            else:
                if self.is_valid_file(filepath):
                    with open(filepath, 'r') as file:
                        content = file.read().strip()
                        if content:
                            snippets.append(content)
        return snippets

    @staticmethod
    def is_valid_file(filepath):
        IGNORED_FILES = ["package-lock.json", "yarn.lock", "poetry.lock"]
        ALLOWED_EXTENSIONS = [".py", ".tsx"]
        return (not any(ignored in filepath for ignored in IGNORED_FILES) and
                any(filepath.endswith(ext) for ext in ALLOWED_EXTENSIONS))
    




def main():
    st.title("Codebase Ingestion and Embedding Generation")
    

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'input1' not in st.session_state:
        st.session_state.input1 = ""

    vector_store, embedding_generator = VectorStore(collection_name="dev_codebase", vector_size=768), CodeEmbedding()
    github_repo_url = st.text_input("Enter GitHub Repository (owner/repo):",placeholder="samarthaggarwal/always-on-debugger",)
    local_codebase_dir = st.text_input("Or enter local directory path:", placeholder="../invoice-understanding")
    
    st.write("")  # Add spacing
    

    if st.session_state.step == 1 and st.button("Load Codebase"): 
        loader = CodebaseLoader(local_codebase_dir, github_repo_url)
        snippets = loader.load_codebase()
        st.session_state.embeddings, st.session_state.code_chunks = [], []

        if snippets:
            st.success(f"Loaded {len(snippets)} snippets.")
            
            with st.spinner('Generating embeddings...'):
                try:
                    
                    st.session_state.embeddings, st.session_state.code_chunks = [], []
                    for snippet, file_path in snippets:
                        code_chunks = code_chunker.chunk_code(snippet)
                        for code_chunk in code_chunks:
                            st.session_state.code_chunks.append((code_chunk, file_path))
                            embedding = embedding_generator.generate_embeddings(code_chunk)
                            print (len(embedding), len(code_chunk))
                            st.session_state.embeddings.append(embedding)
                            v = VectorNode(embedding=embedding, metadata={"code_chunk": code_chunk, "file_path": file_path})
                            vector_store.add_vectors([v])
                        
                except Exception as e:
                    print(e)
                    st.error(f"An error occurred while generating embeddings: {str(e)}")
                    return
            st.write("Embeddings generated successfully.")
            st.session_state.step = 2
        else:
            st.error("No snippets found. Please check the input.")
            

    if st.session_state.step == 2:
        query = st.text_input("", placeholder="Type your query here...")
        run_query = st.button("Run Query")

        if run_query and query:
            with st.spinner('Processing query...'):
                try:

                    query_embedding = embedding_generator.generate_embeddings(query)                    
                    res = vector_store.search(query_embedding)
                    nearest_neighbors = embedding_generator.find_k_nearest_neighbors(query_embedding, st.session_state.embeddings)  # This should work with multiple embeddings

                except Exception as e:
                    st.error(f"An error occurred while processing the query: {str(e)}")
                    return

            # Printing the results
            if not nearest_neighbors:
                st.write("No relevant matches found.")
            else:
                top_matches = nearest_neighbors[:2]
                st.write("Top Matches:")
                for index in top_matches:
                    st.markdown(f"**File: {st.session_state.code_chunks[index][1]}**")
                    st.code(f"Matched Code:\n{st.session_state.code_chunks[index][0]}...\n", language="python")

                st.title("Qdrant Top Matches:")
                for record in res[:4]:
                    st.markdown(f"**File: {record.payload['file_path']}**")
                    st.code(f"Matched Code:\n{record.payload['code_chunk']}...\n", language="python" )
        elif run_query and not query:
            st.error("Please enter a query before running.")

    print ('over')


if __name__ == "__main__":
    main()
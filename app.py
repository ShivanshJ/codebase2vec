import os
import streamlit as st
import github_interface  # Ensure this is the correct import for your GitHub functions
from code_embedding import CodeEmbedding  # Import the CodeEmbedding class


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
    def autocomplete_placeholder(input_value, placeholder):
        if input_value.strip() == '\t':
            return placeholder
        return input_value

    github_repo_url = st.text_input("Enter GitHub Repository (owner/repo):",placeholder="samarthaggarwal/always-on-debugger",)
    local_codebase_dir = st.text_input("Or enter local directory path:", placeholder="../invoice-understanding")
    
    st.write("")  # Add spacing
    query = st.text_input("", placeholder="Type your query here...")

    if st.button("Load Codebase"):
        loader = CodebaseLoader(local_codebase_dir, github_repo_url)
        snippets = loader.load_codebase()

        if snippets:
            st.success(f"Loaded {len(snippets)} snippets.")
            embedding_generator = CodeEmbedding()
            
            with st.spinner('Generating embeddings...'):
                try:
                    embeddings = embedding_generator.generate_embeddings(snippets)
                except Exception as e:
                    st.error(f"An error occurred while generating embeddings: {str(e)}")
                    return
            st.write("Embeddings generated successfully.")

            if query:
                with st.spinner('Processing query...'):
                    try:
                        query_embedding = embedding_generator.generate_embeddings([query])
                        nearest_neighbors = embedding_generator.find_k_nearest_neighbors(query_embedding, embeddings)
                    except Exception as e:
                        st.error(f"An error occurred while processing the query: {str(e)}")
                        return

                if not nearest_neighbors:
                    st.write("No relevant matches found.")
                else:
                    top_matches = nearest_neighbors[:2]
                    st.write("Top Matches:")
                    for index in top_matches:
                        st.code(f"Matched Code:\n{snippets[index][:500]}...\n", language="python")
            else:
                st.error("Please enter a query.")
        else:
            st.error("No snippets found. Please check the input.")

if __name__ == "__main__":
    main()
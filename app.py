import os
import streamlit as st
import code_chunker
from embedding.llm_adapter import LLMAdapter
from github_interface import load_github_codebase
from embedding.embedding import CodeEmbedding  # Import the CodeEmbedding class
from dotenv import load_dotenv

from embedding.context_wrapper import Summarizer
from database.vector_store import VectorStore, VectorNode
from database.snippet_database import SnippetDatabase, Snippet
load_dotenv()

class CodebaseLoader:
    def __init__(self, local_dir=None, github_repo=None):
        self.local_dir = local_dir
        self.github_repo = github_repo
        self.db = SnippetDatabase()
        self.repo_id = self.db.make_repo_id(self.github_repo)
        self.snippets = []
        self.directory_structure = None

    def load_codebase(self) -> list[Snippet]:
        if self.db.repo_exists(self.repo_id):
            print ("CodebaseLoader :  repo exists")
            return self.db.load_snippets(self.repo_id)
        
        if self.github_repo:
            self.snippets = load_github_codebase(self.github_repo)
        elif self.local_dir:
            self.snippets = self.__load_local_codebase(self.local_dir)
        return self.snippets

    def __load_local_codebase(self, directory) -> list[Snippet]:
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
                            newSnippet = Snippet(content=content, file_path=filepath)
                            snippets.append(newSnippet)
                            self.db.save_snippet(self.repo_id, newSnippet)
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

    vector_store = VectorStore(collection_name="dev_codebase2", vector_size=1536)
    embedding_generator = CodeEmbedding(use_llm=True)
    github_repo_url = st.text_input("Enter GitHub Repository (owner/repo):",placeholder="samarthaggarwal/always-on-debugger",)
    local_codebase_dir = st.text_input("Or enter local directory path:", placeholder="../invoice-understanding")
    
    st.write("")  # Add spacing
    

    if st.session_state.step == 1 and st.button("Load Codebase"): 
        loader = CodebaseLoader(local_codebase_dir, github_repo_url)
        snippets: Snippet = loader.load_codebase()
        st.session_state.embeddings, st.session_state.code_chunks = [], []

        if snippets:
            st.success(f"Loaded {len(snippets)} snippets.")
            
            with st.spinner('Generating embeddings...'):
                st.session_state.embeddings, st.session_state.code_chunks = [], []

                def do_embeddings_exist():
                    return False


                def make_embeddings():
                    if vector_store.does_embedding_exist(1):
                        return 
                    
                    for snippet in snippets:
                        snippet, file_path = snippet.content, snippet.file_path
                        try:
                            code_chunks = code_chunker.chunk_code(snippet)
                        except Exception as e:
                            error_msg = f"An error occurred while generating embeddings {file_path}: {str(e)} "
                            st.error(error_msg)
                            continue

                        for code_chunk in code_chunks:
                            try:
                                # --- Make abstract
                                abstract = Summarizer.generate_abstract_with_api(file_path, code_chunk)
                                st.session_state.code_chunks.append((code_chunk, file_path, abstract))
                                # context = Summarizer.context_of_snippet(code_chunk, snippet)

                                # --- Create embeddings
                                print (len(code_chunk), file_path, end=',')
                                my_text = file_path + abstract + code_chunk
                                embedding = embedding_generator.generate_embeddings(my_text)
                                print (len(embedding))

                                # --- Store embeddings
                                v = VectorNode(embedding=embedding, metadata={
                                    "repo_id": 1, 
                                    "code_chunk": code_chunk, 
                                    "file_path": file_path,
                                    "abstract": abstract,})
                                vector_store.add_vectors([v])

                                st.session_state.embeddings.append(embedding)
                            except Exception as e:
                                print ('Exception in generating embeddings', file_path, code_chunk)
                                print (e)
                                continue
                

                make_embeddings()
                # end spinner

            st.write("Embeddings generated successfully.")
            st.session_state.step = 2
        else:
            st.error("No snippets found. Please check the input.")
            

    if st.session_state.step == 2:
        query = st.text_input("", placeholder="Type your query here...")
        run_query = st.button("Run Query")

        if run_query and query:
            with st.spinner('Processing query...'):
                query_embedding = embedding_generator.generate_embeddings(query)
                # --- QUERY SEARCH ---
                res = vector_store.search(query_embedding)
                print ('Embeddings from VS: ', res)
                nearest_neighbors = embedding_generator.find_k_nearest_neighbors(query_embedding, st.session_state.embeddings)  # This should work with multiple embeddings
                print (nearest_neighbors)

            # Printing the results
            if not nearest_neighbors:
                st.write("No relevant matches found.")
            else:
                top_matches = nearest_neighbors or res
                st.write("Top Matches:")

                def top_matches_from_vector_store(res):
                    for x in res[:3]:
                        st.markdown(f"**File: {x.payload.file_path}**")
                        st.code(f"Matched Code:\n{x.payload.code_chunk}...\n", language="python")
                        st.text_area("Abstract:\n", f"{x.payload.abstract}" )


                # for index in top_matches:
                #     st.markdown(f"**File: {st.session_state.code_chunks[index][1]}**")
                #     st.code(f"Matched Code:\n{st.session_state.code_chunks[index][0]}...\n", language="python")
                #     st.text_area("Abstract:\n", f"{st.session_state.code_chunks[index][2]}" )

                top_matches_from_vector_store(res)
                winning_code_abstract = res[0].payload.abstract
                st.title("Qdrant Top Matches:")
                # for record in res[:4]:
                #     st.markdown(f"**File: {record.payload['file_path']}**")
                #     st.code(f"Matched Code:\n{record.payload['code_chunk']}...\n", language="python" )

                def generate_code(winning_code_chunk, winning_code_abstract):
                    print ('in generate_code()')
                    llm = LLMAdapter()
                    user_prompt = "generate code based on the following function definition, so i know how to use this as an API"
                    user_prompt += winning_code_abstract
                    system_prompt = "you are an assistant that ONLY responds with code based on the API function provided by the user. You will show ONLY how to call the API."
                    st.text_area("User Prompt", user_prompt)
                    st.text_area("System Prompt", system_prompt)
                    return llm.chat_completion(user_prompt, system_prompt)
                    
                ans = generate_code()
                st.write("Final code:")
                st.code(f"{ans}",)


        elif run_query and not query:
            st.error("Please enter a query before running.")

    print ('over')






if __name__ == "__main__":
    main()
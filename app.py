import os
import streamlit as st
from dotenv import load_dotenv
from typing import List


import code_chunker
import code_graph
from github_interface import load_github_codebase
from embedding.embedding import CodeEmbedding  # Import the CodeEmbedding class
from embedding.llm_adapter import LLMAdapter
from embedding.context_wrapper import Summarizer
import core.languages as languages

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
            print ("CodebaseLoader :  repo exists in relational DB")
            return self.db.load_snippets(self.repo_id)
        
        if self.github_repo:
            self.snippets = load_github_codebase(self.github_repo)
        elif self.local_dir:
            self.snippets = self.__load_local_codebase(self.local_dir)
        self.db.save_repo_dir_structure(self.repo_id, self.extract_dir_structure(self.snippets))
        return self.__test(self.snippets)

    def __test(self, txt):
        return txt
    
    def extract_dir_structure(self, snippets: list[Snippet]):
        if dir := self.db.get_repo_dir_structure(self.repo_id):
            print ("CodebaseLoader :  dir exists in relational DB")
            return dir
        dir_structure = '\n'
        for snippet in snippets:
            dir_structure += snippet.file_path
            dir_structure += '\n'
        return dir_structure

    def __load_local_codebase(self, directory) -> list[Snippet]:
        snippets = []
        for filename in os.listdir(directory):
            if filename.startswith('.'):
                continue
            filepath = os.path.join(directory, filename)
            if os.path.isdir(filepath):
                snippets.extend(self.__load_local_codebase(filepath))
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
    


def display_generated_code(winning_code_chunk, winning_code_abstract, dir_structure):
    print ('in generate_code()', dir_structure)
    llm = LLMAdapter()
    user_prompt = "generate code based on the following function definition, so i know how to use this as an API"
    user_prompt += winning_code_abstract
    system_prompt = "You are an assistant that ONLY responds with code. Your code is based on the API function asked by the user. You will show ONLY how to call the API in code."
    system_prompt += f"This is the directory structure of the codebase: ```{dir_structure}``` "
    
    st.text_area("User Prompt", user_prompt)
    st.text_area("System Prompt", system_prompt)
    ans = llm.chat_completion(user_prompt, system_prompt)
    
    st.write("Final code:")
    st.code(f"{ans}")


# step 2.1 : Display results
def display_search_results(top_matches, dir_structure):
    st.title("Qdrant Top Matches:")
    if not top_matches:
        st.write("No relevant matches found.")
        return

    def top_matches_from_vector_store(res):
        for x in res[:3]:
            st.markdown(f"**File: {x.payload['file_path']}**")
            st.code(f"Matched Code:\n{x.payload['code_chunk']}...\n", language="python")
            st.text_area("Abstract:\n", f"{x.payload['abstract']}")

    st.write("Top Matches:")
    top_matches_from_vector_store(top_matches)
    
    winning_code_chunk = top_matches[0].payload['code_chunk']
    winning_code_abstract = top_matches[0].payload['abstract']
    display_generated_code(winning_code_chunk, winning_code_abstract, dir_structure)
    


# step 2: Ask questions from embeddings
def handle_query_interface(vector_store, embedding_generator):
    query = st.text_input("", placeholder="Type your query here...")
    run_query = st.button("Run Query")
    
    if run_query:
        if not query:
            st.error("Please enter a query before running.")
            return
            
        with st.spinner('Processing query...'):
            query_embedding = embedding_generator.generate_embeddings(query)
            # --- VECTOR_STORE SEARCH ---
            res = vector_store.search(query_embedding)
            print('Embeddings from VS: ', res)
            # --- LOCAL SEARCH ---
            # Uncomment for 'nearest_neighbors'
            # nearest_neighbors = embedding_generator.find_k_nearest_neighbors(query_embedding, st.session_state.embeddings)  # This should work with multiple embeddings
            # for index in top_matches:
            #     st.markdown(f"**File: {st.session_state.code_chunks[index][1]}**")
            #     st.code(f"Matched Code:\n{st.session_state.code_chunks[index][0]}...\n", language="python")
            #     st.text_area("Abstract:\n", f"{st.session_state.code_chunks[index][2]}" )
            # print (nearest_neighbors)
            display_search_results(res, st.session_state.dir_structure)

#  Step 1.1: 
def process_code_chunks(code_chunks, file_path, repo_id, vector_store, embedding_generator):
    for code_chunk in code_chunks:
        try:
            # --- Make abstract
            abstract = Summarizer.generate_abstract_with_api(file_path, code_chunk)
            st.session_state.code_chunks.append((code_chunk, file_path, abstract))
            # --- Create embeddings
            my_text = file_path + abstract + code_chunk
            embedding = embedding_generator.generate_embeddings(my_text)
            print(f'embedding size ({len(embedding)})')
            # --- Store embeddings
            v = VectorNode(embedding=embedding, metadata={
                "repo_id": repo_id,
                "code_chunk": code_chunk,
                "file_path": file_path,
                "abstract": abstract,
            })
            vector_store.add_vectors([v])
            st.session_state.embeddings.append(embedding)
        except Exception as e:
            print('Exception in generating embeddings', file_path, code_chunk)
            print(e)
            continue


# Step 1: 
def process_snippets(snippets, repo_id, vector_store, embedding_generator):
    with st.spinner('Generating embeddings...'):
        if vector_store.does_embedding_exist(repo_id):
            return
            
        for snippet in snippets:
            try:
                file_path, snippet_content = snippet.file_path, snippet.content
                code_chunks = code_chunker.chunk_code(snippet_content)
                process_code_chunks(code_chunks, file_path, repo_id, vector_store, embedding_generator)
            except Exception as e:
                error_msg = f"An error occurred while generating embeddings {file_path}: {str(e)}"
                st.error(error_msg)
                continue



def load_codebase_callback(local_codebase_dir, github_repo_url, vector_store, embedding_generator):
    loader = CodebaseLoader(local_codebase_dir, github_repo_url)
    snippets: List[Snippet] = loader.load_codebase()
    if not snippets:
        st.error("No snippets found. Please check the input.")
        return
    dir_structure = loader.extract_dir_structure(snippets)
    repo_id = loader.repo_id
    print(f"Repo ID: {repo_id}, \nDir Structure:\n {dir_structure}")
    st.session_state.dir_structure = dir_structure
    st.session_state.embeddings = []
    st.session_state.code_chunks = []
    st.success(f"Loaded {len(snippets)} snippets.")

    # --- Process Snippets
    process_snippets(snippets, repo_id, vector_store, embedding_generator)
    st.write("Embeddings generated successfully.")
    st.session_state.step = 2
    # ---- Show query interface after successful loading
    handle_query_interface(vector_store, embedding_generator)




def load_codebase_callback_graph(local_codebase_dir, github_repo_url, vector_store, embedding_generator):
    loader = CodebaseLoader(local_codebase_dir, github_repo_url)
    snippets: List[Snippet] = loader.load_codebase()
    
    if not snippets:
        st.error("No snippets found. Please check the input.")
        return
        
    dir_structure = loader.extract_dir_structure(snippets)
    repo_id = loader.repo_id
    print(f"Repo ID: {repo_id}, \nDir Structure:\n {dir_structure}")
    st.session_state.dir_structure = dir_structure
    st.session_state.embeddings = []
    st.session_state.code_chunks = []
    
    st.success(f"Loaded {len(snippets)} snippets.")
    # --- Instantiate the components ---
    llm_adapter = LLMAdapter()  # Replace with your LLM adapter
    code_parser = code_graph.TreeSitterCodeParser()
    dependency_extractor = languages.PythonDependencyExtractor()
    abstract_generator = code_graph.LLMBasedAbstractGenerator(llm_adapter)
    vector_store = VectorStore(collection_name="dev_codebase2", vector_size=1536)
    # graph_database = Neo4jGraphDatabase(neo4j_uri, neo4j_user, neo4j_password)
    code_processor = code_graph.CodeProcessor(code_parser, dependency_extractor, abstract_generator, vector_store)
    code_processor.process_codebase(snippets)
    st.write("Embeddings generated successfully.")
    st.session_state.step = 2
    # Show query interface after successful loading
    handle_query_interface(vector_store, embedding_generator)




def main():
    st.title("Codebase Ingestion and Embedding Generation")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'input1' not in st.session_state:
        st.session_state.input1 = ""

    vector_store = VectorStore(collection_name="dev_codebase2", vector_size=1536)
    embedding_generator = CodeEmbedding(use_llm=True)
    
    github_repo_url = st.text_input(
        "Enter GitHub Repository (owner/repo):",
        placeholder="samarthaggarwal/always-on-debugger"
    )
    local_codebase_dir = st.text_input(
        "Or enter local directory path:", 
        placeholder="../invoice-understanding"
    )
    
    st.write("")  # Add spacing
    
    # Create columns with different widths (3:1 ratio)
    col1, col2 = st.columns([3, 1])
    if 'active_column' not in st.session_state:
        st.session_state.active_column = None

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Load Codebase"):
            st.session_state.active_column = 'col1'
            st.rerun()
    with col2:
        if st.button("Load Recursive Abstract Maker"):
            st.session_state.active_column = 'col2'
            st.rerun()

    if st.session_state.active_column == 'col1':
        st.empty()  # Clear previous layout
        load_codebase_callback(local_codebase_dir, github_repo_url, vector_store, embedding_generator)

    elif st.session_state.active_column == 'col2':
        st.empty()  # Clear previous layout
        # Add your recursive abstract maker code here
        load_codebase_callback_graph(local_codebase_dir, github_repo_url, vector_store, embedding_generator)
                

            


# def main():
#     st.title("Codebase Ingestion and Embedding Generation")

#     # Initialize session state
#     if 'step' not in st.session_state:
#         st.session_state.step = 1
#     if 'input1' not in st.session_state:
#         st.session_state.input1 = ""

#     vector_store = VectorStore(collection_name="dev_codebase2", vector_size=1536)
#     embedding_generator = CodeEmbedding(use_llm=True)
#     github_repo_url = st.text_input("Enter GitHub Repository (owner/repo):",placeholder="samarthaggarwal/always-on-debugger",)
#     local_codebase_dir = st.text_input("Or enter local directory path:", placeholder="../invoice-understanding")
    
#     st.write("")  # Add spacing
    


#     if st.session_state.step == 1 and st.button("Load Codebase"): 
#             # ---Init loader
#         loader = CodebaseLoader(local_codebase_dir, github_repo_url)
#         snippets: Snippet = loader.load_codebase()
#         dir_structure = loader.extract_dir_structure(snippets)
#         repo_id = loader.repo_id
#         print (f"Repo ID: {repo_id}, \nDir Structure:\n {dir_structure}")

#         st.session_state.dir_structure = dir_structure
#         st.session_state.embeddings, st.session_state.code_chunks = [], []
#         if snippets:
#             st.success(f"Loaded {len(snippets)} snippets.")
#             with st.spinner('Generating embeddings...'):
#                 st.session_state.embeddings, st.session_state.code_chunks = [], []

#                 def make_embeddings():
#                     if vector_store.does_embedding_exist(repo_id):
#                         return 
#                     for snippet in snippets:
#                         snippet, file_path = snippet.content, snippet.file_path
#                         try:
#                             code_chunks = code_chunker.chunk_code(snippet)
#                         except Exception as e:
#                             error_msg = f"An error occurred while generating embeddings {file_path}: {str(e)} "
#                             st.error(error_msg)
#                             continue
#                         for code_chunk in code_chunks:
#                             try:
#                                 # --- Make abstract
#                                 abstract = Summarizer.generate_abstract_with_api(file_path, code_chunk)
#                                 st.session_state.code_chunks.append((code_chunk, file_path, abstract))
#                                 # context = Summarizer.context_of_snippet(code_chunk, snippet)

#                                 # --- Create embeddings
#                                 print (len(code_chunk), file_path, end=',')
#                                 my_text = file_path + abstract + code_chunk
#                                 embedding = embedding_generator.generate_embeddings(my_text)
#                                 print (f'embedding size ({len(embedding)})')

#                                 # --- Store embeddings
#                                 v = VectorNode(embedding=embedding, metadata={
#                                     "repo_id": repo_id, 
#                                     "code_chunk": code_chunk, 
#                                     "file_path": file_path,
#                                     "abstract": abstract,})
#                                 vector_store.add_vectors([v])

#                                 st.session_state.embeddings.append(embedding)
#                             except Exception as e:
#                                 print ('Exception in generating embeddings', file_path, code_chunk)
#                                 print (e)
#                                 continue
                

#                 make_embeddings()
#                 # end spinner

#             st.write("Embeddings generated successfully.")
#             st.session_state.step = 2
#         else:
#             st.error("No snippets found. Please check the input.")
            

#     if st.session_state.step == 2:
#         query = st.text_input("", placeholder="Type your query here...")
#         run_query = st.button("Run Query")

#         dir_structure = st.session_state.dir_structure

#         if run_query and query:
#             with st.spinner('Processing query...'):
#                 query_embedding = embedding_generator.generate_embeddings(query)
#                 # --- VECTOR_STORE SEARCH ---
#                 res = vector_store.search(query_embedding)
#                 print ('Embeddings from VS: ', res)
#                 # --- LOCAL SEARCH ---
#                 nearest_neighbors = embedding_generator.find_k_nearest_neighbors(query_embedding, st.session_state.embeddings)  # This should work with multiple embeddings
#                 print (nearest_neighbors)

#             # Printing the results
#             top_matches = nearest_neighbors or res
#             if not top_matches:
#                 st.write("No relevant matches found.")
#             else:
#                 st.write("Top Matches:")

#                 # Uncomment for 'nearest_neighbors'
#                 # for index in top_matches:
#                 #     st.markdown(f"**File: {st.session_state.code_chunks[index][1]}**")
#                 #     st.code(f"Matched Code:\n{st.session_state.code_chunks[index][0]}...\n", language="python")
#                 #     st.text_area("Abstract:\n", f"{st.session_state.code_chunks[index][2]}" )
                
#                 def top_matches_from_vector_store(res):
#                     for x in res[:3]:
#                         st.markdown(f"**File: {x.payload['file_path']}**")
#                         st.code(f"Matched Code:\n{x.payload['code_chunk']}...\n", language="python")
#                         st.text_area("Abstract:\n", f"{x.payload['abstract']}" )
#                 top_matches_from_vector_store(top_matches)

#                 winning_code_chunk = res[0].payload['code_chunk']
#                 winning_code_abstract = res[0].payload['abstract']
#                 st.title("Qdrant Top Matches:")
#                 # for record in res[:4]:
#                 #     st.markdown(f"**File: {record.payload['file_path']}**")
#                 #     st.code(f"Matched Code:\n{record.payload['code_chunk']}...\n", language="python" )

#                 def generate_code(winning_code_chunk, winning_code_abstract):
#                     print ('in generate_code()', dir_structure)
#                     llm = LLMAdapter()
#                     user_prompt = "generate code based on the following function definition, so i know how to use this as an API"
#                     user_prompt += winning_code_abstract
#                     system_prompt = "You are an assistant that ONLY responds with code. Your code is based on the API function asked by the user. You will show ONLY how to call the API in code."
#                     system_prompt += "This is the directory structure of the codebase:" + f' ```{dir_structure}``` '
#                     st.text_area("User Prompt", user_prompt)
#                     st.text_area("System Prompt", system_prompt)
#                     return llm.chat_completion(user_prompt, system_prompt)
                    
#                 ans = generate_code(winning_code_chunk, winning_code_abstract)
#                 st.write("Final code:")
#                 st.code(f"{ans}",)


#         elif run_query and not query:
#             st.error("Please enter a query before running.")

#     print ('over')






if __name__ == "__main__":
    main()
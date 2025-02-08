
# class Relation(enum.Enum):
#     HAS_FUNCTION = 'HAS_FUNCTION'
#     HAS_OBJECT_INITIALIZATION = 'HAS_OBJECT'



# class Node:
#     is_sdk_function: bool
#     children_nodes: List[Node]
#     parent_nodes: List[Node]


import textwrap
import unittest
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Protocol
from tree_sitter import Node
from tree_sitter_languages import get_parser
from abc import ABC, abstractmethod
from whats_that_code.election import guess_language_all_methods

from embedding.llm_adapter import LLMAdapter
from embedding.embedding import CodeEmbedding
from database.vector_store import VectorStore, VectorNode
from embedding.embedding import EmbeddingStrategy
from database.snippet_database import Snippet
from core.schema import MyBlock, Chunk
from core.languages.python.python_parser import PythonDependencyExtractor
from core.languages.javascript.javascript_parser import JavaScriptDependencyExtractor


def _get_line_number_from_char_index(index: int, source_code: str) -> int:
    """Placeholder for your actual line number calculation."""
    return source_code.encode("utf-8")[:index].count(b'\n') + 1

    

# --- Interfaces (for Dependency Inversion) ---

class CodeParser(ABC):
    @abstractmethod
    def parse(self, source_code: str, language: str, filepath: str) -> List[MyBlock]:
        pass

class DependencyExtractor(ABC):
    @abstractmethod
    def extract_dependencies(self, blocks: List[MyBlock], block_map: Dict[str, MyBlock]):
        pass

class AbstractGenerator(ABC):
    @abstractmethod
    def generate_abstracts(self, blocks: List[MyBlock]):
        pass

class GraphDatabase(ABC):
    @abstractmethod
    def create_or_update_node(self, block: MyBlock):
        pass

    @abstractmethod
    def create_or_update_relationship(self, from_block: MyBlock, to_block: MyBlock, relationship_type: str):
        pass

    @abstractmethod
    def close(self):
        pass

# --- Implementations ---

class TreeSitterCodeParser(CodeParser):
    def __init__(self):
        self.block_map: Dict[str, MyBlock] = {}

    def parse(self, source_code: str, language: str, filepath: str) -> List[MyBlock]:
        parser = get_parser(language)
        tree = parser.parse(source_code.encode("utf8"))
        return self._process_children(tree.root_node, source_code, filepath, None)
        

    def _process_children(self, node: Node, source_code: str, filepath: str, parent_block: Optional["MyBlock"]) -> List[MyBlock]:
        """Process all children of a node and return valid blocks."""
        blocks = []
        for child in node.children:
            if child_blocks := self._extract_blocks_hierarchical(child, source_code, filepath, parent_block):
                blocks.extend(child_blocks)
        return blocks

    def _extract_blocks_hierarchical(self, node: Node, source_code: str, filepath: str, 
                                   parent_block: Optional["MyBlock"] = None) -> Optional[MyBlock]:
        """Extract code blocks from the AST in a hierarchical manner."""
        blocks, current_block = [], None
        block_type = self._get_block_type(node)
        block_span = Chunk(node.start_byte, node.end_byte)
        code_content = source_code[block_span.start:block_span.end]
        if block_type:
            # Create block for current node
            file_name = filepath.split('/')[-1] if filepath else "unknown_file"
            block_name = self._get_block_name(node, block_type) or file_name
            full_name = self._get_full_block_name(block_name, parent_block)
            current_block = MyBlock(
                type=block_type,
                name=full_name,
                span=block_span,
                node=node,
                filepath=filepath,
                code_content=code_content,
                parent=parent_block
            )
            print ('Blocktype:', block_type, ', Name: ', full_name, ', Span: ', block_span, end="\n\n")
            self.block_map[full_name] = current_block
            if parent_block: parent_block.children.append(current_block)
            blocks.append(current_block)
        
        child_blocks = self._process_children(node, source_code, filepath, current_block or parent_block)
        return blocks + child_blocks

    @staticmethod
    def _get_block_type(node: Node) -> str:
        try:
            # Handle file-level nodes
            # if node.type in ["program", "module", "source_file"]:
            #     return "file"
            if node.type in ["function_definition","function_declaration","arrow_function","method_definition"]:
                return 'function'
            elif node.type in ["class_definition", "class_declaration"]:
                return 'class'
            elif node.type in ["jsx_element", "jsx_self_closing_element"]:
                return 'component'
            elif node.type == "impl_item":
                return 'impl'
            else:
                return None
        except Exception as e:
            print ("Error in CodeChunker, TreeSitterCodeParser _get_block_type()", e)
            return None


    @staticmethod
    def _get_block_name(node: Node, block_type: str) -> str:
        try:
            if block_type in ['function']:
                name_node = node.child_by_field_name('name')
            elif block_type == 'class':
                name_node = (
                    node.child_by_field_name('name') or          # Regular classes
                    node.child_by_field_name('id') or           # Some JS classes
                    next((                                      # Class expressions
                        child for child in node.children 
                        if child.type == "identifier"
                    ), None)
                )
            elif block_type == 'impl':
                name_node = node.child_by_field_name('trait') or node.child_by_field_name('type')
            elif block_type in ['component']:
                opening_element = node.child_by_field_name('opening_element')
                if opening_element:
                    name_node = opening_element.child_by_field_name('name')
                else:
                    return "unnamed_jsx"
            else:
                return "unnamed"

            return name_node.text.decode('utf-8') if name_node else "unnamed"
        except Exception as e:
            print ("Error in CodeChunker, TreeSitterCodeParser _get_block_name()", e)
            return "unnamed"

    def _get_full_block_name(self, block_name: str, parent_block: Optional[MyBlock]) -> str:
        if parent_block:
            return f"{parent_block.name}.{block_name}"
        else:
            return block_name




"""
Does topological sort and starts making abstracts
"""
class LLMBasedAbstractGenerator(AbstractGenerator):
    def __init__(self, llm_adapter: LLMAdapter):
        self.llm_adapter = llm_adapter
        self.function_summary_prompt = """
            Provide a concise summary of the following code block, including its purpose, parameters, and return value:
            {{CODE_CHUNK}}
            Located in the file: {{PATH_TO_FILE}}
        """

    def generate_abstracts(self, blocks: List[MyBlock]):
        sorted_blocks = self._topological_sort(blocks)
        for block in sorted_blocks:
            if block.type == "function":
                dependency_abstracts = {
                    dep.name: dep.abstract
                    for dep in block.dependencies
                    if dep.abstract
                }
                block.abstract = self._generate_abstract_prompt(block, dependency_abstracts)

    def _generate_abstract_prompt(self, block: MyBlock, dependency_abstracts: Dict[str, str]) -> str:
        system_prompt = "You are a helpful assistant that generates concise function/class abstracts without referencing the dependency function calls in the code block."
        user_prompt = self.function_summary_prompt.replace('{{PATH_TO_FILE}}', block.filepath)
        user_prompt = user_prompt.replace('{{CODE_CHUNK}}', block.code_content)

        if dependency_abstracts:
            user_prompt += "\n\nThis is the abstract of its dependency functions, \
            please use this information to generate a cohesive abstract as though the called functions were replaced by their actual code \
            , refrain from sayhing that the dependency functions were called \
            , instead give what the whole function would do:\n"
            for name, abstract in dependency_abstracts.items():
                user_prompt += f"- {name}: {abstract}\n"

        abstract = self.llm_adapter.chat_completion(user_prompt, system_prompt)
        return abstract

    def _topological_sort(self, blocks: List[MyBlock]) -> List[MyBlock]:
        visited = set()
        stack = []

        def visit(block):
            visited.add(block)
            for dependent in block.dependents:
                if dependent not in visited:
                    visit(dependent)
            stack.insert(0, block)

        for block in blocks:
            if block not in visited:
                visit(block)

        return stack


"""
class Neo4jGraphDatabase(GraphDatabase):
    def __init__(self, uri, user, password):
        from neo4j import GraphDatabase as Neo4jDriver
        self.driver = Neo4jDriver.driver(uri, auth=(user, password))

    def create_or_update_node(self, block: MyBlock):
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_node_tx, block)

    @staticmethod
    def _create_or_update_node_tx(tx, block: MyBlock):
        query = (
            "MERGE (n:CodeBlock {full_name: $full_name}) "
            "ON CREATE SET n.type = $type, n.name = $name, n.filepath = $filepath, n.start_line = $start_line, n.end_line = $end_line, n.code_content = $code_content, n.abstract = $abstract "
            "ON MATCH SET n.type = $type, n.name = $name, n.filepath = $filepath, n.start_line = $start_line, n.end_line = $end_line, n.code_content = $code_content, n.abstract = COALESCE($abstract, n.abstract)"
        )
        tx.run(query, full_name=block.name, type=block.type, name=block.name.split(".")[-1], filepath=block.filepath, start_line=block.span.start, end_line=block.span.end, code_content=block.code_content, abstract=block.abstract)

    def create_or_update_relationship(self, from_block: MyBlock, to_block: MyBlock, relationship_type: str):
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_relationship_tx, from_block, to_block, relationship_type)

    @staticmethod
    def _create_or_update_relationship_tx(tx, from_block: MyBlock, to_block: MyBlock, relationship_type: str):
        query = (
            "MATCH (a:CodeBlock {full_name: $from_full_name}) "
            "MATCH (b:CodeBlock {full_name: $to_full_name}) "
            "MERGE (a)-[r:" + relationship_type + "]->(b)"
        )
        tx.run(query, from_full_name=from_block.name, to_full_name=to_block.name)

    def close(self):
        self.driver.close()
"""

class CodeProcessor:
    def __init__(self, code_parser: CodeParser, 
            dependency_extractor: DependencyExtractor, 
            abstract_generator: AbstractGenerator,
            vector_store: VectorStore = None, 
            graph_database: GraphDatabase = None):
        self.code_parser = code_parser
        self.dependency_extractor = dependency_extractor
        self.abstract_generator = abstract_generator
        self.graph_database = graph_database
        self.block_map: Dict[str, MyBlock] = {}
        self.vector_store = vector_store
        self.embedding_generator = CodeEmbedding(use_llm=True)

    def process_codebase(self, codebases: List[Snippet]):
        all_blocks = []
        for snippet in codebases:
            source_code, filepath = snippet.content, snippet.file_path
            language = guess_language_all_methods(code=source_code, file_name=filepath)
            print (language, end=', ')
            blocks = self.code_parser.parse(source_code, language, filepath)
            all_blocks.extend(blocks)
            # Update the global block_map with the blocks from this file
            self.block_map.update(self.code_parser.block_map)

        
        self.dependency_extractor.extract_dependencies(all_blocks, self.block_map)  # Pass the global block_map here
        
        self.abstract_generator.generate_abstracts(all_blocks)
        self.print_graph()
        self.make_rag(0, all_blocks)
        

        # Store in Neo4j
        # for block in all_blocks:
        #     self.graph_database.create_or_update_node(block)
        #     for dep in block.dependencies:
        #         self.graph_database.create_or_update_relationship(block, dep, "DEPENDS_ON")

    def make_rag(self, repo_id, all_blocks: List[MyBlock]):
        for block in all_blocks:
            # --- Create embeddings
            my_text = block.filepath + block.abstract + block.code_content
            embedding = self.embedding_generator.generate_embeddings(my_text)
            print(f'embedding size ({len(embedding)})')
            # --- Store embeddings
            v = VectorNode(embedding=embedding, metadata={
                "repo_id": repo_id,
                "code_chunk": block.code_content,
                "file_path": block.filepath,
                "abstract": block.abstract,
            })
            self.vector_store.add_vectors([v])


    def print_graph(self):
        """Prints the graph structure of the codebase."""
        print ('----------- Graph ------------ ')
        def _print_block(block: MyBlock, indent_level=0):
            indent = "  " * indent_level
            print(f"\n{indent}Type: {block.type}, Name: {block.name}, File: {block.filepath}")
            if block.abstract:
                print(f"{indent}  Abstract: {block.abstract}")

            print(f"{indent}  Dependencies:")
            for dep in block.dependencies:
                print(f"{indent}    - {dep.name}")

            print(f"{indent}  Dependents:")
            for dep in block.dependents:
                print(f"{indent}    - {dep.name}")
            
            for child in block.children:
                _print_block(child, indent_level + 2)

        for block_name, block in self.block_map.items():
            # Check if the block is a top-level block (no parent)
            if block.parent is None:
                _print_block(block)

    



import os
class TestCodeProcessor(unittest.TestCase):
    def setUp(self):
        llm_adapter = LLMAdapter()
        self.code_parser = TreeSitterCodeParser()
        self.abstract_generator = LLMBasedAbstractGenerator(llm_adapter)


    def test_process_js_snippets(self):
        self.dependency_extractor = JavaScriptDependencyExtractor()
        self.code_processor = CodeProcessor(
            self.code_parser,
            self.dependency_extractor,
            self.abstract_generator
        )
        codebases = [
            Snippet(
                content="""
        class MyClass {
            function myFunction(a, b) {
                return a + b;
            }
            function myFunction2() {
                return this.myFunction(1, 2);
            }
        }
                """,
                file_path="file1.js"
            ),
            Snippet(
                content="""
        function anotherFunction() {
            console.log("Hello");
        }
        function anotherFunction2() {
            return anotherFunction()
        }
                """,
                file_path="file2.js"
            )
        ]
        
        # Process the codebase
        result = self.code_processor.process_codebase(codebases)
        
        # Add assertions based on expected behavior
        self.assertIsNotNone(result)  # Replace with more specific assertions
        
    def test_process_embedding_file(self):
        self.dependency_extractor = PythonDependencyExtractor()
        self.code_processor = CodeProcessor(
            self.code_parser,
            self.dependency_extractor,
            self.abstract_generator
        )
        # Get the path to embedding.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, 'code_chunker.py')
        
        # Read the embedding.py file
        with open(path, 'r') as f:
            content = f.read()
            
        codebases = [
            Snippet(
                content=content,
                file_path="code_chunker.py"
            )
        ]
        
        # Process the embedding file
        result = self.code_processor.process_codebase(codebases)
        
        # Add assertions
        self.assertIsNotNone(result)
        # Add more specific assertions based on what you expect from processing embedding.py

if __name__ == '__main__':
    unittest.main()
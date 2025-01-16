
# class Relation(enum.Enum):
#     HAS_FUNCTION = 'HAS_FUNCTION'
#     HAS_OBJECT_INITIALIZATION = 'HAS_OBJECT'



# class Node:
#     is_sdk_function: bool
#     children_nodes: List[Node]
#     parent_nodes: List[Node]


import textwrap
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Protocol

from tree_sitter import Node
from tree_sitter_languages import get_parser
from abc import ABC, abstractmethod


@dataclass
class Chunk:
    start: int
    end: int

def _get_line_number_from_char_index(index: int, source_code: str) -> int:
    """Placeholder for your actual line number calculation."""
    return source_code.encode("utf-8")[:index].count(b'\n') + 1

class LLMAdapter:
    def chat_completion(self, user_prompt, system_prompt):
        # Mock implementation - replace with your actual LLM interaction
        print("----- LLM Prompt -----")
        print(f"System: {system_prompt}")
        print(f"User: {user_prompt}")
        print("----- LLM Response -----")
        return "Mock LLM abstract response."
    
# --- Data Model ---

@dataclass
class MyBlock:
    type: str
    name: str
    span: Chunk
    node: Node
    filepath: str
    code_content: str
    children: List["MyBlock"] = None  # Will be initialized in post_init
    parent: Optional["MyBlock"] = None
    dependencies: List["MyBlock"] = None  # Will be initialized in post_init
    dependents: List["MyBlock"] = None  # Will be initialized in post_init
    abstract: Optional[str] = None

    def __post_init__(self):
        # Initialize lists properly
        self.children = [] if self.children is None else self.children
        self.dependencies = [] if self.dependencies is None else self.dependencies
        self.dependents = [] if self.dependents is None else self.dependents

    def __hash__(self):
        return hash(self.name + self.code_content)

    def __eq__(self, other):
        if not isinstance(other, MyBlock):
            return False
        if self.name == other.name:
            return self.code_content == other.code_content
        return False

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
        root_block = self._extract_blocks_hierarchical(tree.root_node, source_code, filepath)
        if root_block:
            top_level_blocks = root_block.children
            return top_level_blocks
        else:
            return []

    def _extract_blocks_hierarchical(self, node: Node, source_code: str, filepath: str, parent_block: Optional["MyBlock"] = None) -> Optional[MyBlock]:
        block_type = self._get_block_type(node)

        if block_type:
            block_name = self._get_block_name(node, block_type)
            block_span = Chunk(node.start_byte, node.end_byte)
            code_content = source_code[block_span.start:block_span.end]

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
            if parent_block:
                parent_block.children.append(current_block)

            self.block_map[full_name] = current_block

            current_block.children = [
                child_block
                for child in node.children
                if (child_block := self._extract_blocks_hierarchical(child, source_code, filepath, current_block)) is not None
            ]
            return current_block
        # Process container nodes
        elif node.children:
            container_block = MyBlock(
                type="container",
                name="container",
                span=Chunk(node.start_byte, node.end_byte),
                node=node,
                filepath=filepath,
                code_content=source_code[node.start_byte:node.end_byte],
                parent=parent_block
            )
            # Process children of container
            for child in node.children:
                child_block = self._extract_blocks_hierarchical(child, source_code, filepath, parent_block)
                if child_block:
                    container_block.children.append(child_block)
            
            return container_block if container_block.children else None
        else:
            return None

    @staticmethod
    def _get_block_type(node: Node) -> str:
        try:
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



class SimpleDependencyExtractor(DependencyExtractor):

    def extract_dependencies(self, blocks: List[MyBlock], block_map: Dict[str, MyBlock]):
        for block in blocks:
            self._find_dependencies(block, block_map)
            if block.children:
                self.extract_dependencies(block.children, block_map)  # recursive call

    def _find_dependencies(self, block: MyBlock, block_map: Dict[str, MyBlock]):
        for child in block.node.children:
            if child.type == 'call_expression':
                function_node = child.child_by_field_name('function')
                if function_node:
                    dependency_name = self._extract_dependency_name(function_node)
                    if dependency_name:
                        full_dependency_name = self._resolve_dependency_name(dependency_name, block, block_map)
                        if full_dependency_name in block_map:
                            dependency_block = block_map[full_dependency_name]
                            block.dependencies.append(dependency_block)
                            dependency_block.dependents.append(block)
            else:
                self._find_dependencies_recursive(child, block, block_map)

    def _find_dependencies_recursive(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        for child in node.children:
            if child.type == 'call_expression':
                function_node = child.child_by_field_name('function')
                if function_node:
                    dependency_name = self._extract_dependency_name(function_node)
                    if dependency_name:
                        full_dependency_name = self._resolve_dependency_name(dependency_name, block, block_map)
                        if full_dependency_name in block_map:
                            dependency_block = block_map[full_dependency_name]
                            block.dependencies.append(dependency_block)
                            dependency_block.dependents.append(block)
            else:
                self._find_dependencies_recursive(child, block, block_map)

    def _extract_dependency_name(self, node: Node) -> Optional[str]:
        if node.type == "identifier":
            return node.text.decode("utf-8")
        elif node.type == "member_expression":
            object_node = node.child_by_field_name("object")
            property_node = node.child_by_field_name("property")
            if object_node and property_node:
                object_name = self._extract_dependency_name(object_node)
                property_name = property_node.text.decode("utf-8")
                if object_name:
                    return f"{object_name}.{property_name}"
                else:
                    return property_name
        return None

    def _resolve_dependency_name(self, dependency_name: str, current_block: MyBlock, block_map: Dict[str, MyBlock]) -> str:
        parent = current_block.parent
        while parent:
            full_name_attempt = f"{parent.name}.{dependency_name}"
            if full_name_attempt in block_map:
                return full_name_attempt
            parent = parent.parent

        if dependency_name in block_map:
            return dependency_name

        return dependency_name



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

            Summary:
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
                block.abstract = self._generate_abstract(block, dependency_abstracts)

    def _generate_abstract(self, block: MyBlock, dependency_abstracts: Dict[str, str]) -> str:
        system_prompt = "You are a helpful assistant that generates concise function/class abstracts."
        user_prompt = self.function_summary_prompt.replace('{{PATH_TO_FILE}}', block.filepath)
        user_prompt = user_prompt.replace('{{CODE_CHUNK}}', block.code_content)

        if dependency_abstracts:
            user_prompt += "\n\nDependencies:\n"
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
    def __init__(self, code_parser: CodeParser, dependency_extractor: DependencyExtractor, abstract_generator: AbstractGenerator, graph_database: GraphDatabase = None):
        self.code_parser = code_parser
        self.dependency_extractor = dependency_extractor
        self.abstract_generator = abstract_generator
        self.graph_database = graph_database
        self.block_map: Dict[str, MyBlock] = {}

    def process_codebase(self, codebases: List[Tuple[str, str, str]]):
        all_blocks = []
        for source_code, language, filepath in codebases:
            blocks = self.code_parser.parse(source_code, language, filepath)
            all_blocks.extend(blocks)
            # Update the global block_map with the blocks from this file
            self.block_map.update(self.code_parser.block_map)

        
        self.dependency_extractor.extract_dependencies(all_blocks, self.block_map)  # Pass the global block_map here
        self.print_graph()

        # self.abstract_generator.generate_abstracts(all_blocks)

        # Store in Neo4j
        # for block in all_blocks:
        #     self.graph_database.create_or_update_node(block)
        #     for dep in block.dependencies:
        #         self.graph_database.create_or_update_relationship(block, dep, "DEPENDS_ON")


    def print_graph(self):
        """Prints the graph structure of the codebase."""

        def _print_block(block: MyBlock, indent_level=0):
            indent = "  " * indent_level
            print(f"{indent}Type: {block.type}, Name: {block.name}, File: {block.filepath}")
            if block.abstract:
                print(f"{indent}  Abstract: {block.abstract}")

            print(f"{indent}  Dependencies:")
            for dep in block.dependencies:
                print(f"{indent}    - {dep.name}")

            print(f"{indent}  Dependents:")
            for dep in block.dependents:
                print(f"{indent}    - {dep.name}")
            
            for child in block.children:
                _print_block(child, indent_level + 1)

        for block_name, block in self.block_map.items():
            # Check if the block is a top-level block (no parent)
            if block.parent is None:
                _print_block(block)

    



def main():
    # --- Configuration (Replace with your actual values) ---
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

    # --- Instantiate the components ---
    llm_adapter = LLMAdapter()  # Replace with your LLM adapter
    code_parser = TreeSitterCodeParser()
    dependency_extractor = SimpleDependencyExtractor()
    abstract_generator = LLMBasedAbstractGenerator(llm_adapter)
    # graph_database = Neo4jGraphDatabase(neo4j_uri, neo4j_user, neo4j_password)

    code_processor = CodeProcessor(code_parser, dependency_extractor, abstract_generator)

    # --- Process the codebase (Example with dummy code snippets) ---
    
    # Example usage with multiple codebases
    codebases = [
        (
            """
class MyClass {
    function myFunction(a, b) {
        return a + b;
    }
    function myFunction2() {
        return this.myFunction(1, 2);
    }
}
            """,
            "javascript",
            "file1.js",
        ),
        (
            """
function anotherFunction() {
    console.log("Hello");
}
function anotherFunction2() {
    return anotherFunction()
}
            """,
            "javascript",
            "file2.js",
        )
    ]
    code_processor.process_codebase(codebases)

    print("Codebase processed and abstracts generated. Data stored in Neo4j.")
    # graph_database.close()





if __name__ == "__main__":
    main()
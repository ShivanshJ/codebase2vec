# code_chunker.py
# Chunking code into smaller chunks for embedding

# We have to chunk and preserve context: which funnction is contained within a class


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from whats_that_code.election import guess_language_all_methods

from tree_sitter import Node
from tree_sitter_languages import get_parser
 



def chunk_code(source_code: str) -> List[str]:
    code_chunks = []
    language = guess_language_all_methods(source_code)
    for chunk in BlockAwareCodeSplitter.split_text(source_code, language):
        # continue
        print (chunk)
        code_chunk_str = chunk.extract_lines(source_code)
        print(code_chunk_str + "\n\n====================\n\n")
        code_chunks.append(code_chunk_str)
    return code_chunks



"""
As a brief helper data structure, we first implemented the following dataclass for representing a slice of a string:
"""
@dataclass
class Chunk:
    # Represents a slice of a string, in bytes (1 character = 1 byte)
    start: int = 0
    end: int = 0

    def __post_init__(self):
        if self.end is None:
            self.end = self.start

    def extract(self, s: str) -> str:
        # Grab the corresponding substring of string s by bytes
        return s[self.start : self.end]

    def extract_lines(self, s: str) -> str:
        # Grab the corresponding substring of string s by lines
        return "\n".join(s.splitlines()[self.start : self.end + 1 ])

    def __add__(self, other: Chunk | int) -> Chunk:
        # e.g. Chunk(1, 2) + Chunk(2, 4) = Chunk(1, 4) (concatenation)
        # There are no safety checks: Chunk(a, b) + Chunk(c, d) = Chunk(a, d)
        # and there are no requirements for b = c.

        if isinstance(other, int):
            return Chunk(self.start + other, self.end + other)
        elif isinstance(other, Chunk):
            return Chunk(self.start, other.end)
        else:
            raise NotImplementedError()

    def __len__(self) -> int:
        # i.e. Chunk(a, b) = b - a
        return self.end - self.start
    


    
def _get_line_number_from_char_index(index: int, source_code: str) -> int:
    """
    Get the line number from a given character index in the source code.
    It iterates through the lines of the source code, keeping track of the total characters
    processed, until it finds the line containing the specified index.

    Args:
        index (int): The character index to find the corresponding line number for.
        source_code (str): The full source code string.

    Returns:
        int: The line number (0-indexed) corresponding to the given character index.
    """
    total_chars = 0
    for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
        total_chars += len(line)
        if total_chars > index:
            return line_number - 1
    return line_number
    

# Inspired by SweepAI in: https://github.com/run-llama/llama_index/pull/7100/files
# and https://github.com/sweepai/sweep/blob/main/notebooks/chunking.ipynb
class TextChunker:
    @staticmethod
    def split_text(source_code: str, language: str) -> List[Chunk]:
        """Split incoming code and return chunks using the AST."""

        parser = get_parser(language)
        tree = parser.parse(source_code.encode("utf8"))
        chunks = TextChunker._chunk_node(tree.root_node)
        print (chunks, end="\n\n")

        # 2. Filling in the gaps
        # It sets the end of the previous chunk (prev.end) to the start of the current chunk
        for prev, curr in zip(chunks[:-1], chunks[1:]):
            prev.end = curr.start
        curr.start = tree.root_node.end_byte  #This ensures that the last chunk extends to the end of the parsed code
        print (chunks, end="\n\n")

        # 3. Combining small chunks with bigger ones
        chunks = TextChunker.__coalesce_chunks(chunks, source_code)
        print (chunks, end="\n\n")

        # 4. Changing line numbers
        line_chunks = [
            Chunk(_get_line_number_from_char_index(chunk.start, source_code), _get_line_number_from_char_index(chunk.end, source_code))
            for chunk in chunks
        ]
        print (line_chunks, end="\n\n")

        # 5. Eliminating empty chunks
        line_chunks = [chunk for chunk in line_chunks if len(chunk) > 0]
        print (line_chunks, end="\n\n")

        return line_chunks
     

            
    @staticmethod
    def _chunk_node(node: Node, MAX_CHARS: int = 600) -> List[Chunk]:
        # 1. Recursively form chunks based on the last post (https://docs.sweep.dev/blogs/chunking-2m-files)
        chunks: list[Chunk] = []
        current_chunk: Chunk = Chunk(node.start_byte, node.start_byte)

        for child in node.children:
            if child.end_byte - child.start_byte > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Chunk(child.end_byte, child.end_byte)
                chunks.extend(TextChunker._chunk_node(child, MAX_CHARS))
            elif child.end_byte - child.start_byte + len(current_chunk) > MAX_CHARS:
                chunks.append(current_chunk)
                current_chunk = Chunk(child.start_byte, child.end_byte)
            else:
                current_chunk += Chunk(child.start_byte, child.end_byte)

        chunks.append(current_chunk)
        return chunks
    

    @staticmethod
    def __coalesce_chunks(chunks: list[Chunk], source_code: str, coalesce: int = 50) -> list[Chunk]:
        """
        Coalesce small chunks into larger ones.

        Args:
            chunks (list[Chunk]): A list of Chunk objects representing the initial chunks.
            source_code (str): The original source code string.
            coalesce (int, optional): The minimum size for a chunk. Defaults to 50 characters.

        Returns:
            list[Chunk]: A new list of Chunk objects representing the coalesced chunks.

        The method works as follows:
        1. It iterates through the input chunks, combining them into a current_chunk.
        2. When the current_chunk exceeds the coalesce size and contains a newline, it's added to new_chunks.
        3. Any remaining content in current_chunk is added as the final chunk.

        This coalescing process helps to create more meaningful and sizeable code chunks for further processing.
        """
        new_chunks = []
        current_chunk = Chunk(0, 0)
        for chunk in chunks:
            current_chunk += chunk
            if len(current_chunk) > coalesce and "\n" in current_chunk.extract(source_code):
                new_chunks.append(current_chunk)
                current_chunk = Chunk(chunk.end, chunk.end)
        if len(current_chunk) > 0:
            new_chunks.append(current_chunk)
        return new_chunks
    

    


@dataclass
class MyBlock:
    type: str
    name: str
    span: Chunk

        

class BlockAwareCodeSplitter:
    """
    Split code into chunks while preserving the blocks (functions, classes, etc.) that contain them.
    """
    def __init__(self, overlap_lines: int = 2):
        self.overlap_lines = overlap_lines

    @staticmethod
    def split_text(source_code: str, language: str) -> List[Tuple[str, str]]:
        parser = get_parser(language)
        tree = parser.parse(source_code.encode("utf8"))
        
        blocks = BlockAwareCodeSplitter._extract_blocks(tree.root_node)
        
        line_chunks = []
        for b in blocks:
            chunk = b.span
            line_chunks.append(
                Chunk(_get_line_number_from_char_index(chunk.start, source_code), _get_line_number_from_char_index(chunk.end, source_code)) 
            )
        return line_chunks


    @staticmethod
    def _extract_blocks(node: Node) -> List[MyBlock]:
        blocks = []
        for child in node.children:
            block_type = BlockAwareCodeSplitter.__get_block_type(child)
            block_name = BlockAwareCodeSplitter._get_block_name(child, child.type)
            block_span = Chunk(child.start_byte, child.end_byte)
            if block_type:
                print ('Block', block_type, block_name, block_span, end="\n\n") 
                blocks.append(MyBlock(block_type, block_name, block_span))
            
            blocks.extend(BlockAwareCodeSplitter._extract_blocks(child))
        return blocks
    
    @staticmethod
    def __get_block_type(node: Node) -> str:
        # These are the types we are interested in
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
    
    @staticmethod
    def _get_block_name(node: Node, block_type: str) -> str:
        if block_type in ['function_definition', 'class_definition', 'method_definition']:
            name_node = node.child_by_field_name('name')
        elif block_type == 'impl_item':
            name_node = node.child_by_field_name('trait') or node.child_by_field_name('type')
        elif block_type in ['jsx_element', 'jsx_fragment']:
            opening_element = node.child_by_field_name('opening_element')
            if opening_element:
                name_node = opening_element.child_by_field_name('name')
            else:
                return "unnamed_jsx"
        else:
            return "unnamed"
        
        return name_node.text.decode('utf-8') if name_node else "unnamed"

    def _overlaps(self, func_span, chunk_span):
        return (func_span.start <= chunk_span.end and func_span.end >= chunk_span.start)
    




# Test code:
# import requests

# example_file = "https://raw.githubusercontent.com/sweepai/sweep/b267b613d4c706eaf959fe6789f11e9a856521d1/sweepai/handlers/on_check_suite.py"
# python_code = requests.get(example_file).text


# chunk_code(python_code, 'python')

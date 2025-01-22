from dataclasses import dataclass
from tree_sitter import Node
from typing import List, Optional

@dataclass
class Chunk:
    start: int
    end: int


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
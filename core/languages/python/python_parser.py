from typing import List, Dict, Optional, Set
from tree_sitter import Node
from core.schema import MyBlock

class PythonDependencyExtractor:
    def __init__(self):
        self.visited_nodes: Set[int] = set()
        
    def extract_dependencies(self, blocks: List[MyBlock], block_map: Dict[str, MyBlock]):
        """Extract dependencies for all blocks recursively with cycle detection."""
        self.visited_nodes.clear()
        for block in blocks:
            self._process_block(block, block_map)
            
    def _process_block(self, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Process a single block and its children."""
        self._find_dependencies(block.node, block, block_map)
        
        if block.children:
            for child in block.children:
                if id(child) not in self.visited_nodes:
                    self.visited_nodes.add(id(child))
                    self._process_block(child, block_map)

    def _find_dependencies(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Find dependencies in a node recursively with Python-specific pattern matching."""
        if not node:
            return
            
        if id(node) in self.visited_nodes:
            return
        self.visited_nodes.add(id(node))

        # Python-specific node types
        if node.type == 'call':
            self._handle_call(node, block, block_map)
        elif node.type == 'class_definition':
            self._handle_class_definition(node, block, block_map)
        elif node.type == 'import_statement':
            self._handle_import_statement(node, block, block_map)
        elif node.type == 'import_from_statement':
            self._handle_import_from_statement(node, block, block_map)
        elif node.type == 'decorator':
            self._handle_decorator(node, block, block_map)
        elif node.type == 'attribute':
            self._handle_attribute(node, block, block_map)
        elif node.type == 'type_annotation':
            self._handle_type_annotation(node, block, block_map)
            
        # Process children recursively
        for child in node.children:
            self._find_dependencies(child, block, block_map)

    def _handle_call(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle function calls including method calls."""
        function_node = node.child_by_field_name('function')
        if not function_node:
            return
            
        # Handle both direct calls and method calls
        dependency_name = self._extract_dependency_name(function_node)
        if dependency_name:
            self._add_dependency(block, dependency_name, block_map)
            
        # Handle arguments for potential dependencies
        arguments = node.child_by_field_name('arguments')
        if arguments:
            for arg in arguments.children:
                if arg.type == 'keyword_argument':
                    value = arg.child_by_field_name('value')
                    if value:
                        self._find_dependencies(value, block, block_map)
                else:
                    self._find_dependencies(arg, block, block_map)

    def _handle_class_definition(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle class definitions including inheritance and metaclasses."""
        # Handle base classes
        bases = node.child_by_field_name('bases')
        if bases:
            for base in bases.children:
                base_name = self._extract_dependency_name(base)
                if base_name:
                    self._add_dependency(block, base_name, block_map)
        
        # Handle metaclass
        keywords = node.child_by_field_name('keywords')
        if keywords:
            for keyword in keywords.children:
                if keyword.type == 'keyword_argument' and keyword.child_by_field_name('name').text.decode('utf-8') == 'metaclass':
                    value = keyword.child_by_field_name('value')
                    if value:
                        metaclass_name = self._extract_dependency_name(value)
                        if metaclass_name:
                            self._add_dependency(block, metaclass_name, block_map)

    def _handle_import_statement(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle import statements."""
        names = node.child_by_field_name('names')
        if names:
            for name_node in names.children:
                if name_node.type == 'dotted_name':
                    imported_name = name_node.text.decode('utf-8')
                    self._add_dependency(block, imported_name, block_map)
                elif name_node.type == 'aliased_import':
                    name = name_node.child_by_field_name('name')
                    if name:
                        imported_name = name.text.decode('utf-8')
                        self._add_dependency(block, imported_name, block_map)

    def _handle_import_from_statement(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle from ... import ... statements."""
        module = node.child_by_field_name('module')
        if module:
            module_name = module.text.decode('utf-8')
            names = node.child_by_field_name('names')
            if names:
                for name_node in names.children:
                    if name_node.type == 'aliased_import':
                        name = name_node.child_by_field_name('name')
                        if name:
                            imported_name = f"{module_name}.{name.text.decode('utf-8')}"
                            self._add_dependency(block, imported_name, block_map)
                    else:
                        imported_name = f"{module_name}.{name_node.text.decode('utf-8')}"
                        self._add_dependency(block, imported_name, block_map)

    def _handle_decorator(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle decorators."""
        decorator_name = self._extract_dependency_name(node)
        if decorator_name:
            self._add_dependency(block, decorator_name, block_map)

    def _handle_attribute(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle attribute access."""
        value = node.child_by_field_name('value')
        if value:
            dependency_name = self._extract_dependency_name(value)
            if dependency_name:
                self._add_dependency(block, dependency_name, block_map)

    def _handle_type_annotation(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle type annotations including generics."""
        annotation = node.child_by_field_name('annotation')
        if annotation:
            # Handle simple types
            if annotation.type == 'identifier':
                type_name = annotation.text.decode('utf-8')
                self._add_dependency(block, type_name, block_map)
            # Handle generic types
            elif annotation.type == 'subscript':
                value = annotation.child_by_field_name('value')
                if value:
                    type_name = self._extract_dependency_name(value)
                    if type_name:
                        self._add_dependency(block, type_name, block_map)
                # Handle type arguments
                slice_node = annotation.child_by_field_name('slice')
                if slice_node:
                    self._find_dependencies(slice_node, block, block_map)

    def _extract_dependency_name(self, node: Node) -> Optional[str]:
        """Extract dependency name with Python-specific patterns."""
        if not node:
            return None
        
        # For simple attribute access, just use the full text
        

        if node.type == "identifier":
            return node.text.decode("utf-8")

        elif node.type == "attribute" and not any(
            child.type in {"call", "parenthesized_expression", "conditional_expression"}
            for child in node.children
        ):
            return node.text.decode("utf-8")
            
        elif node.type == "attribute":
            value = node.child_by_field_name("value")
            attr = node.child_by_field_name("attribute")
            
            if value and attr:
                value_name = self._extract_dependency_name(value)
                attr_name = attr.text.decode("utf-8")
                
                if value_name:
                    return f"{value_name}.{attr_name}"
                return attr_name
                
        elif node.type == "dotted_name":
            return node.text.decode("utf-8")
            
        return None

    def _resolve_dependency_name(self, dependency_name: str, current_block: MyBlock, block_map: Dict[str, MyBlock]) -> str:
        """Resolve dependency name with Python-specific scope handling."""
        # Check if it's already a fully qualified name

        if dependency_name in block_map:
            return dependency_name
            
        # Try to resolve relative to current block's hierarchy
        parts = dependency_name.split('.')
        current = current_block
        
        while current:
            # Try full path from current scope
            full_name = f"{current.name}.{dependency_name}"
            if full_name in block_map:
                return full_name
                
            # Try partial paths for Python's relative imports
            for i in range(len(parts)):
                partial_name = '.'.join(parts[i:])
                full_name = f"{current.name}.{partial_name}"
                if full_name in block_map:
                    return full_name
                    
            current = current.parent
            
        return dependency_name

    def _add_dependency(self, block: MyBlock, dependency_name: str, block_map: Dict[str, MyBlock]):
        """Add dependency with duplicate checking."""
        full_name = self._resolve_dependency_name(dependency_name, block, block_map)
        if full_name in block_map:
            dependency_block = block_map[full_name]
            if dependency_block != block and dependency_block not in block.dependencies:
                block.dependencies.append(dependency_block)
                dependency_block.dependents.append(block)
from tree_sitter import Node
from typing import List, Dict, Optional, Set

from core.schema import MyBlock


class JavaScriptDependencyExtractor:

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
        """Find dependencies in a node recursively with enhanced pattern matching."""
        if not node:
            return
            
        # Avoid processing the same node multiple times
        if id(node) in self.visited_nodes:
            return
        self.visited_nodes.add(id(node))

        # Process different node types
        if node.type == 'call_expression':
            self._handle_call_expression(node, block, block_map)
        elif node.type == 'class_definition':
            self._handle_class_definition(node, block, block_map)
        elif node.type == 'import_statement':
            self._handle_import_statement(node, block, block_map)
        elif node.type == 'member_expression':
            self._handle_member_expression(node, block, block_map)
            
        # Process children recursively
        for child in node.children:
            self._find_dependencies(child, block, block_map)

    def _handle_call_expression(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle function call expressions."""
        function_node = node.child_by_field_name('function')
        if not function_node:
            return
            
        dependency_name = self._extract_dependency_name(function_node)
        if dependency_name:
            self._add_dependency(block, dependency_name, block_map)
            
        # Handle method arguments for potential dependencies
        arguments = node.child_by_field_name('arguments')
        if arguments:
            for arg in arguments.children:
                self._find_dependencies(arg, block, block_map)

    def _handle_class_definition(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle class definitions including inheritance."""
        base_classes = node.child_by_field_name('base_classes')
        if base_classes:
            for base in base_classes.children:
                base_name = self._extract_dependency_name(base)
                if base_name:
                    self._add_dependency(block, base_name, block_map)

    def _handle_import_statement(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle import statements."""
        imported_names = node.child_by_field_name('names')
        if imported_names:
            for name_node in imported_names.children:
                if name_node.type == 'dotted_name':
                    imported_name = name_node.text.decode('utf-8')
                    self._add_dependency(block, imported_name, block_map)

    def _handle_member_expression(self, node: Node, block: MyBlock, block_map: Dict[str, MyBlock]):
        """Handle member access expressions."""
        object_node = node.child_by_field_name('object')
        if object_node:
            dependency_name = self._extract_dependency_name(object_node)
            if dependency_name:
                self._add_dependency(block, dependency_name, block_map)

    def _extract_dependency_name(self, node: Node) -> Optional[str]:
        """Extract dependency name with enhanced handling of complex expressions."""
        if not node:
            return None
            
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
                return property_name
                
        elif node.type == "dotted_name":
            return node.text.decode("utf-8")
            
        return None

    def _resolve_dependency_name(self, dependency_name: str, current_block: MyBlock, block_map: Dict[str, MyBlock]) -> str:
        """Resolve dependency name with enhanced scope handling."""
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
                
            # Try partial paths
            for i in range(len(parts)):
                partial_name = '.'.join(parts[i:])
                full_name = f"{current.name}.{partial_name}"
                if full_name in block_map:
                    return full_name
                    
            current = current.parent
            
        # If no resolution found, return original name
        return dependency_name

    def _add_dependency(self, block: MyBlock, dependency_name: str, block_map: Dict[str, MyBlock]):
        """Add dependency with duplicate checking."""
        full_name = self._resolve_dependency_name(dependency_name, block, block_map)
        if full_name in block_map:
            dependency_block = block_map[full_name]
            if dependency_block != block and dependency_block not in block.dependencies:
                block.dependencies.append(dependency_block)
                dependency_block.dependents.append(block)
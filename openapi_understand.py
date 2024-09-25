from dataclasses import dataclass
import yaml
from yaml import YAMLError
import numpy as np
from embedding.embedding import CodeEmbedding
import os



@dataclass
class OpenAPIEmbedding:
    path: str
    method: str
    operation: dict
    embedding: list[float]



class OpenAPISpecHandler:
    def __init__(self, code_embedding_obj: CodeEmbedding):
        self.specs = {}
        self.code_embedding_obj = code_embedding_obj
        self.openapi_embeddings = []

    def _load_openapi_spec(self, spec_path):
        """Load the OpenAPI specification from a YAML file."""
        if self.specs.get(spec_path) is not None:
            return self.specs[spec_path]
        if not spec_path.endswith(('.yaml', '.yml')):
            print(f"Error: {spec_path} is not a YAML file.")
            return None
        try:
            with open(spec_path, 'r') as file:
                spec = yaml.safe_load(file)
                if not is_openapi_spec(spec):
                    print(f"Error: {spec_path} is not a valid OpenAPI specification.")
                    return None
                self.specs[spec_path] = spec
                return spec
        except YAMLError as e:
            print(f"Error parsing YAML: {e}")
        except IOError as e:
            print(f"Error reading file: {e}")
        return None

    def create_endpoint_embeddings(self, spec_path) -> list[OpenAPIEmbedding]:
        """Create embeddings for each endpoint in the OpenAPI spec."""
        spec = self._load_openapi_spec(spec_path)
        for path, path_item in spec['paths'].items():
            for method, operation in path_item.items():
                description = self._get_endpoint_description(path, method, operation)
                self.openapi_embeddings.append(OpenAPIEmbedding(
                    path=path,
                    method=method,
                    operation=operation,
                    embedding=self.code_embedding_obj.generate_embeddings(description)
                ))
        return self.openapi_embeddings

    def find_matching_endpoint_with_embeddings(self, user_query):
        """Find the best matching endpoint using embeddings."""
        if not self.openapi_embeddings:
            return None
        results = []
        user_embedding = self.code_embedding_obj.generate_embeddings(user_query)
        embeddings = [oe.embedding for oe in self.openapi_embeddings]
        top_idxs = self.code_embedding_obj.find_k_nearest_neighbors(user_embedding, embeddings)
        
        for idx in top_idxs:
            description = self._get_endpoint_description(self.openapi_embeddings[idx].path, self.openapi_embeddings[idx].method, self.openapi_embeddings[idx].operation)
            results.append(description)
        return results
    

    def _get_endpoint_description(self, path, method, operation):
        return f"{method.upper()} {path}: {operation.get('summary', '')} {operation.get('description', '')}"




def is_openapi_spec(spec):
    """
    Check if the given specification is a valid OpenAPI specification.
    
    :param spec: A dictionary containing the parsed YAML/JSON content
    :return: Boolean indicating whether it's a valid OpenAPI spec
    """
    # Check for required OpenAPI fields
    if not isinstance(spec, dict):
        return False
    
    # Check for OpenAPI version
    if 'openapi' not in spec:
        return False
    
    # Check for info object
    if 'info' not in spec or not isinstance(spec['info'], dict):
        return False
    
    # Check for title in info object
    if 'title' not in spec['info']:
        return False
    
    # Check for paths object
    if 'paths' not in spec or not isinstance(spec['paths'], dict):
        return False
    
    # If all checks pass, it's likely an OpenAPI spec
    return True




# Example usage
if __name__ == "__main__":
    spec_path = "./data/Qdrant OpenAPI Main.yaml"
    openapi_embeddings = OpenAPISpecHandler(code_embedding_obj=CodeEmbedding(use_sentence_transformer=True))
    
    openapi_embeddings.create_endpoint_embeddings(spec_path)
    res = openapi_embeddings.find_matching_endpoint_with_embeddings("delete vector")
    for x in res:
        print('Res: ', x)
    

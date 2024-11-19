import pickle
import os
import sys
from dataclasses import dataclass


@dataclass
class Snippet:
    content: str
    file_path: str


class SnippetDatabase:
    def __init__(self):
        self.db_folder = "database"
        self.db_file = os.path.join(self.db_folder, "snippets.pkl")
        self.all_snippets = {}

        os.makedirs(self.db_folder, exist_ok=True)
        
        if not os.path.exists(self.db_file):
            print('Creating empty database file')
            with open(self.db_file, 'wb') as f:
                pickle.dump({}, f)
        else:
            print('Loading existing database file')
            with open(self.db_file, 'rb') as f:
                self.all_snippets = pickle.load(f)
    

    def load_snippets(self, repo_id=None):
        """Load snippets from the pickle file."""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                all_snippets = pickle.load(f)
                repo_db = all_snippets.get(repo_id, {}) 
                return repo_db['snippets']
        return []
    
    def get_repo_dir_structure(self, repo_id: str) -> str:
        if os.path.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                all_snippets = pickle.load(f)
                repo_db = all_snippets.get(repo_id, {}) 
                return repo_db['dir_structure'] if 'dir_structure' in repo_db else ''
        return ''

    
    def make_repo_id(self, repo_input: str) -> str:
        if repo_input.startswith("http"):
            # It's a GitHub URL
            parts = repo_input.split("/")
            if len(parts) >= 2:
                return f"{parts[-2]}/{parts[-1]}"
        else:
            abs_path = os.path.abspath(repo_input)
            return os.path.basename(os.path.normpath(abs_path))
        return repo_input

    def save_snippet(self, repo_id: str, snippet: Snippet):
        """Save a snippet to the database."""
        if repo_id not in self.all_snippets:
            self.all_snippets[repo_id] = {}
            self.all_snippets[repo_id]['snippets'] = []
        self.all_snippets[repo_id]['snippets'].append(snippet)
        with open(self.db_file, 'wb') as f:
            pickle.dump(self.all_snippets, f)

    def save_repo_dir_structure(self, repo_id: str, dir_structure: str):
        if repo_id not in self.all_snippets:
            self.all_snippets[repo_id] = {}
        self.all_snippets[repo_id]['dir_structure'] = dir_structure

    def repo_exists(self, repo_id: str):
        """Check if the repository already exists in the database."""
        return repo_id in self.all_snippets and len(self.all_snippets[repo_id]) > 0
    


import unittest

class TestSnippetDatabase(unittest.TestCase):
    def setUp(self):
        self.db = SnippetDatabase()  # Assuming SnippetDatabase is the class name

    def test_make_repo_id_github_url(self):
        """Test the make_repo_id method."""
        # Test GitHub URL
        github_url = "https://github.com/username/repo-name"
        expected_id = "username/repo-name"
        self.assertEqual(self.db.make_repo_id(github_url), expected_id)



    def test_make_repo_id_local_path(self):
        # Test local directory path
        local_path = "./my-project"
        self.assertEqual(self.db.make_repo_id(local_path), "my-project")

        # Test edge case: URL with no repository name
        edge_case_url = "./"
        self.assertEqual(self.db.make_repo_id(edge_case_url), "codebase2vec")

    def test_make_repo_id_unrecognized(self):
        # Test unrecognized input
        unrecognized = "not_a_url_or_path"
        self.assertEqual(self.db.make_repo_id(unrecognized), unrecognized)

if __name__ == '__main__':
    unittest.main()





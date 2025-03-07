from dataclasses import dataclass
import requests
from database.snippet_database import SnippetDatabase, Snippet



db = SnippetDatabase()
ALLOWED_EXTENSIONS = [".py", ".tsx"]

def fetch_github_repo_contents(repo_url, subdirectory=''):
    """Fetch the contents of a GitHub repository."""
    api_url = f"https://api.github.com/repos/{repo_url}/contents/{subdirectory}"
    response = requests.get(api_url)
    # response.raise_for_status()  # Raise an error for bad responses
    return response.json()



def load_github_codebase(repo_url, subdirectory='') -> list[Snippet]:
    """Load codebase from GitHub repository."""

    snippets, repo_id = [], db.make_repo_id(repo_url)

    try:
        contents = fetch_github_repo_contents(repo_url, subdirectory)
        snippets = []
        
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        print(f"Repository contents for {repo_url}: {contents[0]}")
        pp.pprint(contents[0])

        if isinstance(contents, dict) and 'message' in contents:
            print(f"Error: {contents['message']}")
            return snippets
        if not isinstance(contents, list):
            print(f"Unexpected response format: {type(contents)}")
            return snippets

        for item in contents:
            if not isinstance(item, dict):
                print(f"Unexpected item format: {type(item)}")
                continue
            if item.get('type') == 'file' and any(item.get('name', '').endswith(ext) for ext in ALLOWED_EXTENSIONS):
                try:
                    file_response = requests.get(item['download_url'])
                    # file_response.raise_for_status()
                    content = file_response.text.strip()
                    if content:
                        snippet = Snippet(content=content, file_path=item.get('path', ''))
                        snippets.append(snippet)
                        db.save_snippet(repo_id,snippet)
                except requests.RequestException as e:
                    print(f"Error fetching file {item.get('name')}: {e}")
            elif item.get('type') == 'dir':
                # Recursively load directory contents if needed
                snippets.extend(load_github_codebase(f"{repo_url}", subdirectory=item.get('path', '')))

        return snippets
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
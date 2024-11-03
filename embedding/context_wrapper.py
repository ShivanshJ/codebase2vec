
from typing import List, Tuple, Dict
import textwrap
import ast

from embedding.llm_adapter import LLMAdapter


function_summary_prompt = """
<file_location>
{{PATH_TO_FILE}}
</file_location>

<target_code>
{{CODE_CHUNK}}
</target_code>

Analyze the code chunk within the larger file context and provide a concise docstring that includes:

1. The function call or class name
2. It's inputs if its a function. and if it's a class then the constructor if its a class along with the required parameters for initializing it.
3. It's return types, or how it modifies state.
4. The primary purpose of this code chunk, if its an entrypoint to the SDK or not.
Format the response as a brief, search-optimized summary focusing on technical relevance and relationships within the codebase. 
Use precise technical terminology that would be valuable for code search and understanding.
"""


prompt = """
Please provide an abstract for the following:
```
{{CODE_CHUNK}}
```
Include information about parameters, return types, and a brief description of its purpose.
"""



class Summarizer:

    @staticmethod
    def context_of_snippet(snippet_text, document_text):
        # generates function summary in context
        pass


    @staticmethod
    def generate_abstract_with_api(file_path: str, block_code: str) -> str:
        max_tokens = 4000  # Adjust this based on the model's capabilities
        if len(block_code) > max_tokens:
            block_code = textwrap.shorten(block_code, width=max_tokens, placeholder="...")

        system_prompt = "You are a helpful assistant that generates concise function/class abstracts."
        user_prompt = function_summary_prompt.replace('{{PATH_TO_FILE}}', file_path)
        user_prompt = user_prompt.replace('{{CODE_CHUNK}}', block_code)
        llm = LLMAdapter()
        return llm.chat_completion(user_prompt, system_prompt)


    
if __name__ == '__main__':
    llm = LLMAdapter()
    system_prompt = "you are an AI assistant that creates abstracts of functions/classes"
    block_code = "func1(a,b):\n return (a+b)^2"

    resp = Summarizer.generate_abstract_with_api('/path/', block_code)
    print (resp)
    

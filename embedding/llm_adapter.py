from litellm import completion


class LLMAdapter:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name


    def chat_completion(self, user_prompt, system_prompt):
        try:
            messages = [
                {"content": system_prompt, "role": "system"},
                {"content": user_prompt, "role": "user"},
            ]
            response = completion(model=self.model_name, messages=messages)
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"Error generating abstract: {str(e)}"


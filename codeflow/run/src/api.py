import requests
import openai
class ChatModelAPI:
    def __init__(self, api_url, api_key,model_name):
        """
        Initialize API connection
        :param api_url: API URL
        :param api_key: Optional API Key for authentication
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name


    def generate(self, messages, max_tokens=38912, temperature=0.6, top_p=0.95, top_k=20):
        client = openai.OpenAI(api_key=self.api_key,base_url=self.api_url)
        request_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": messages}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": 0,
        }
        # Keep top_k in request body for OpenAI-compatible backends that support it.
        if top_k is not None:
            request_kwargs["extra_body"] = {"top_k": int(top_k)}

        response = client.chat.completions.create(**request_kwargs)
    
        return response



import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


class VertexAIClient:
    def __init__(self, project, location, model):
        self.name = f"vertex/{model}"
        vertexai.init(project=project, location=location)
        self._generative_model = GenerativeModel(model)

    def ensure_ready(self):
        pass  # construction above already validates project/location/model; nothing lazy to check

    def query(self, prompt, temperature=0.0, max_tokens=16384):
        generation_config = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        response = self._generative_model.generate_content(prompt, generation_config=generation_config)
        return response.text

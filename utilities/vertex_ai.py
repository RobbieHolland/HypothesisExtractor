import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


class VertexAIClient:
    def __init__(self, project, location, model):
        self.name = f"vertex/{model}"
        vertexai.init(project=project, location=location)
        self._generative_model = GenerativeModel(model)

    def query(self, prompt, temperature=0.0, max_tokens=8192):
        generation_config = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        response = self._generative_model.generate_content(prompt, generation_config=generation_config)
        return response.text

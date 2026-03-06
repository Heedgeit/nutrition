from groq import Groq
import os

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))



class GroqReasoningModel:

    def __init__(self, groq_client, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.1):
        """
        Initialize the Groq reasoning model
        """
        self.groq = groq_client
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, results, query_text):
        """
        Generate reasoning-based answer using retrieved context
        """

        completion = self.groq.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a logical assistant. Use only the provided context retrieved from the database to answer. Do not use any other information. "
                        "Structure your output in Markdown with a 'Reasoning' and 'Final Answer' section."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context: {results}\n\nQuestion: {query_text}"
                }
            ],
            temperature=self.temperature
        )

        return completion.choices[0].message.content
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

with client.responses.stream(
    model="gpt-4o-mini",
    input="Tell me a joke"
) as stream:

    for event in stream:   # ✅ iterate directly
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

    final_response = stream.get_final_response()

import os
import time
import copy


os.environ["OPENAI_API_KEY"]= "YOUR-KEY"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

time_gap = {"gpt-4": 3, "gpt-3.5-turbo": 0.5}
if OPENAI_API_KEY != "":
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"OPENAI_API_KEY: ****{OPENAI_API_KEY[-4:]}")
else:
    print("Warning: OPENAI_API_KEY is not set")


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY != "":
    # import google.generativeai as palm
    import google.generativeai as genai
    import google.ai.generativelanguage as glm

    # palm.configure(api_key=GOOGLE_API_KEY)
    genai.configure(api_key=GOOGLE_API_KEY)
    print(f"GOOGLE_API_KEY: ****{GOOGLE_API_KEY[-4:]}")

CLAUDE2_API_KEY = os.getenv("CLAUDE2_API_KEY", "")
if CLAUDE2_API_KEY != "":
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

    anthropic = Anthropic(api_key=CLAUDE2_API_KEY, base_url="https://api.aiproxy.io")
    print(f"CLAUDE2_API_KEY: ****{CLAUDE2_API_KEY[-4:]}")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
if ANTHROPIC_API_KEY != "":
    from anthropic import Anthropic

    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print(f"ANTHROPIC_API_KEY: ****{ANTHROPIC_API_KEY[-4:]}")


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
if MISTRAL_API_KEY != "":
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage

    mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
    print(f"MISTRAL_API_KEY: ****{MISTRAL_API_KEY[-4:]}")


def gpt_response(message: list, model="gpt-4", temperature=0, max_tokens=500):
    time.sleep(time_gap.get(model, 3))
    try:
        res = client.chat.completions.create(model=model, messages=message, temperature=temperature, n=1,
                                             max_tokens=max_tokens)
        return res.choices[0].message.content
    except Exception as e:
        print(e)
        time.sleep(time_gap.get(model, 3) * 2)
        return gpt_response(message, model, temperature, max_tokens)


def cohere_response(message: list, model=None, temperature=0, max_tokens=500):
    msg = copy.deepcopy(message[:-1])
    new_msg = message[-1]["content"]
    for m in msg:
        m.update({"role": "CHATBOT" if m["role"] == "system" else "USER", "message": m.pop("content")})

    try:
        return co.chat(chat_history=msg, message=new_msg).text
    except Exception as e:
        print(e)
        time.sleep(1)
        return cohere_response(message)


def palm_response(message: list, model=None, temperature=0, max_tokens=500):
    msg = [{'author': '1' if m["role"] == "user" else '0', **m} for m in message]
    for m in msg:
        m.pop("role", None)
    try:
        res = palm.chat(messages=msg)
        return res.last
    except Exception as e:
        print(e)
        time.sleep(1)
        return palm_response(message, temperature=temperature)
    

def gemini_response(message: list, model="gemini-1.0-pro", temperature=0, max_tokens=500):
    msg = []
    for m in message[:-1]:
        role = "user" if m["role"] == "user" else "model"
        msg.append(glm.Content(parts=[glm.Part(text=m["content"])], role=role))
    genai_model = genai.GenerativeModel(model_name=model)
    try:
        chat = genai_model.start_chat()
        res = chat.send_message(message[-1]["content"])
        return res.text  
    except Exception as e:
        print(e)
        time.sleep(3)
        return gemini_response(message, model, temperature, max_tokens)


def claude_aiproxy_response(message, model=None, temperature=0, max_tokens=500):
    prompt = ""
    for m in message:
        prompt += AI_PROMPT if m["role"] in ["system", "assistant"] else HUMAN_PROMPT
        prompt += " " + m["content"]
    prompt += AI_PROMPT
    try:
        res = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            prompt=prompt,
        )
        return res.completion
    except Exception as e:
        print(e)
        time.sleep(1)
        return claude_aiproxy_response(message, model, temperature, max_tokens)


def claude_response(message, model="claude-3-sonnet-20240229", temperature=0, max_tokens=500):
    msg = []
    for m in message:
        role = m["role"] if m["role"] == "user" else "assistant"
        if msg and msg[-1]["role"] == role:
            msg[-1]["content"] += m["content"]
        else:
            msg.append({"role": role, "content": m["content"]})
    try:
        res = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=msg
        )
        return res.content[0].text
    except Exception as e:
        print(e)
        time.sleep(3)
        return claude_response(message, model, temperature, max_tokens)


def llama_response(message, model=None, temperature=0, max_tokens=500):
    try:
        chat_completion = llama_client.chat.completions.create(
            messages=message,
            model="meta-llama/Llama-2-70b-chat-hf",
            max_tokens=max_tokens
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        time.sleep(1)
        llama_response(message, model, temperature, max_tokens)


def mistral_response(message: list, model="mistral-large-latest", temperature=0, max_tokens=500):
    msg = [ChatMessage(role=m["role"], content=m["content"]) for m in message]
    try:
        res = mistral_client.chat(model=model, messages=msg)
        return res.choices[0].message.content
    except Exception as e:
        print(e)
        time.sleep(1)
        return mistral_response(message, model, temperature, max_tokens)


def gemma_response(message: list, model=None, temperature=0, max_tokens=500):
    import uot.model_gemma as gm
    try:
        gm.gemma_response(history=message, output_len=max_tokens)
    except Exception as e:
        print(e)
        time.sleep(1)
        return gemma_response(message, model, temperature, max_tokens)


def get_response_method(model):
    response_methods = {
        "gpt": gpt_response,
        "cohere": cohere_response,
        "palm": palm_response,
        "_claude": claude_aiproxy_response,
        "claude": claude_response,
        "llama": llama_response,
        "mistral": mistral_response,
        "gemma": gemma_response,
        "gemini": gemini_response,
    }
    return response_methods.get(model.split("-")[0], lambda _: NotImplementedError())

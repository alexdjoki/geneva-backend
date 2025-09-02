from flask import Blueprint, request, jsonify
from models.chat_history import ChatHistory
from models.product_history import ProductHistory
from models.devices import Devices
from app import db
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from together import Together
from urllib.parse import urlparse
from serpapi import GoogleSearch
import re
import anthropic
import google.generativeai as genai
import os
import requests
import asyncio
import json
from models.auth import AccessKey
from models.transactions import Transactions
from datetime import datetime, timedelta
import random
import time
from collections import OrderedDict
from itertools import chain


load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
serpapi_key = os.getenv('SERPAPI_KEY')
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
deepseek_client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
grok_client = OpenAI(api_key=os.getenv('GROK_API_KEY'), base_url="https://api.x.ai/v1")
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
bing_api_key = os.getenv('BING_API_KEY')
google_search_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
google_cx_id = os.getenv('GOOGLE_CX_ID')
brave_api_key = os.getenv('BRAVE_API_KEY')
rainforest_api_key = os.getenv('RAINFOREST_API_KEY')
together_client = Together()

openai_bp = Blueprint('openai', __name__)

def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

#Generate answer using GPT-4o (OpenAI)
def gpt4o_generate(history, prompt, question):
    messages = [
        {"role": "system", "content": prompt},
    ]

    for element in history:
        if element['type'] == 'question':
            messages.append({"role": "user", "content": element["text"]})
        if element['type'] == 'answer':
            messages.append({"role": "assistant", "content": element["text"]})

    messages.append({
        "role": "user",
        "content": question
    })

    return client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages
    ).choices[0].message.content

async def get_gpt4o_answer(history, prompt, question):
    print('start gpt', int(time.time()))
    try:
        answer = await asyncio.to_thread(gpt4o_generate, history, prompt, question)
        print(answer)
        print('end gpt', int(time.time()))
        return {"model": "GPT-4.1", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "GPT-4.1", "answer": "", "status": "failed"}

# Generate answer using Claude (Anthropic)
def claude_generate(history, question):
    messages = []
    for element in history:
        if element['type'] == 'question':
            messages.append({"role": "user", "content": element["text"]})
        elif element['type'] == 'answer':
            messages.append({"role": "assistant", "content": element["text"]})

    # Append the new user question
    messages.append({"role": "user", "content": question})

    full_response = ""

    # Use streaming API
    with anthropic_client.messages.stream(
        model="claude-3-5-sonnet-20240620",  # or "claude-opus-4-20250514"
        max_tokens=4096,   # set generously
        messages=messages
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                # Append incremental text
                full_response += event.delta
            elif event.type == "message_stop":
                break

    return full_response
async def get_claude_answer(history, prompt, question):
    print('start claude', int(time.time()))
    try:
        answer = await asyncio.to_thread(claude_generate, history, prompt, question)
        print(answer)
        print('end claud', int(time.time()))
        return {"model": "Claude Opus 4", "answer": answer, "status": "success"}
    except Exception as e:
        print(str(e))
        return {"model": "Claude Opus 4", "answer": "", "status": "failed"}

# Generate answer using Gemini
def gemini_generate(history, prompt, question):
    messages = []
    for element in history:
        if element['type'] == 'question':
            messages.append({"role": "user", "parts": [element["text"]]})
        if element['type'] == 'answer':
            messages.append({"role": "model", "parts": [element["text"]]})

    model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")
    chat = model.start_chat(history=messages)
    response = chat.send_message(question)
    return response.text

async def get_gemini_answer(history, prompt, question):
    print('start gemini', int(time.time()))
    try:
        answer = await asyncio.to_thread(gemini_generate, history, prompt, question)
        print('end gemini', int(time.time()))
        return {"model": "Gemini 2.5 Pro", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Gemini 2.5 Pro", "answer": "", "status": "failed"}

#Generate answer using deepseek
def deepseek_generate(history, prompt, question):
    try:
        messages = [
            {"role": "system", "content": prompt},
        ]

        for element in history:
            if element['type'] == 'question':
                messages.append({"role": "user", "content": element["text"]})
            if element['type'] == 'answer':
                messages.append({"role": "assistant", "content": element["text"]})

        messages.append({
            "role": "user",
            "content": question
        })

        return deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        ).choices[0].message.content
    except Exception as e:
        print(str(e))
        return ''

async def get_deepseek_answer(history, prompt, question):
    print('start deepseek', int(time.time()))
    try:
        answer = await asyncio.to_thread(deepseek_generate, history, prompt, question)
        print(answer)
        print('end deepseek', int(time.time()))
        return {"model": " Deepseek V3", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": " Deepseek V3", "answer": "", "status": "failed"}
#Generate answer using Grok 3
def grok_generate(history, prompt, question):
    messages = [
        {"role": "system", "content": prompt},
    ]

    for element in history:
        if element['type'] == 'question':
            messages.append({"role": "user", "content": element["text"]})
        if element['type'] == 'answer':
            messages.append({"role": "assistant", "content": element["text"]})

    messages.append({
        "role": "user",
        "content": question
    })

    return grok_client.chat.completions.create(
        model="grok-4-latest",
        messages=messages
    ).choices[0].message.content

async def get_grok_answer(history, prompt, question):
    print('start grok', int(time.time()))
    try:
        answer = await asyncio.to_thread(grok_generate, history, prompt, question)
        print(answer)
        print('end grok', int(time.time()))
        return {"model": "Grok 4", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Grok 4", "answer": str(e), "status": "failed"}
#Generate answer using mistral
def mistral_generate(history, prompt, question):
    payload = {
        "model": "mistral-large-2411",
        "messages": [
            {"role": "user", "content": question}
        ],
        "temperature": 0.7
    }

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)

    try:
        print("Parsed JSON:", response.json())
    except ValueError:
        print("Response is not valid JSON.")

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception("Request failed with status code:", response.status_code)

async def get_mistral_answer(history, prompt, question):
    print('start mistral')
    try:
        answer = await asyncio.to_thread(mistral_generate, history, prompt, question)
        print('end mistral')
        print(answer)
        return {"model": "Mistral Large", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Mistral Large", "answer": "", "status": "failed"}
#Generate answer using llama model of together.ai
def llama_generate(history, prompt, question):
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

async def get_llama_answer(history, prompt, question):
    print('start llama', int(time.time()))
    try:
        answer = await asyncio.to_thread(llama_generate, history, prompt, question)
        print(answer)
        print('end llama', int(time.time()))
        return {"model": "Llama 4", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Llama 4", "answer": str(e), "status": "failed"}

#Summarize the opinion from answers of each models
def summarize_opinion(responses, mode):
    successful_answers = [res for res in responses if res["status"] == "success"]
    if not successful_answers:
        return "All models failed to answer."
    if mode == "consensus":
        summary_prompt = """You will be given answers from different AI models. 
Provide a **Structured Comparison of AI Model Responses From Different Models** with the following format:

Structured Comparison of AI Model Responses

### Key Points of Agreement
- Clearly state what specific models (by name/version) agreed on. Example: "GPT-4.1 and Claude Opus 4 agree that X."
- If more than two models agree, say: "GPT-4.1, Claude Opus 4, and Gemini 2.5 Pro all agree that Y."

### Notable Differences
- Explicitly mention which model(s) gave a different perspective. Example: "Mistral Large emphasized..., while Grok 4 focused on..."

### Unique Insights By Model
Provide unique insights grouped under subheadings for each model.

**{Model Name Version}**:
- Summarize the unique insights only this model contributed.

---

### Final Observations
- before last section, please add divider line
**Consensus**:
- Summarize the strongest areas where models clearly aligned.

**Divergence**:
- Summarize the most significant disagreements or contrasting emphases.

**Skepticism vs Advocacy**:
- Highlight where models showed cautious reasoning (skepticism) vs confident promotion (advocacy).

Rules:
- Always include the section titles exactly as shown above.
- Always attribute points to the correct models by their name/version as provided in the input.
- Do not merge everything into a generic summary without attribution.
- Never leave a section empty; if nothing fits, write "None noted."
"""

    elif mode == "blaze":
        summary_prompt = """
> You will be given answers from several AI models. Summarize their collective reasoning as follows:
> 1) Begin with “The models agree that …” once, then list shared points.
> 2) Refer to individual models as “Model 1”, “Model 2”, etc. (do NOT use actual model names).
> 3) Use numbered sections or bullet points (no tables).
> 4) Do NOT mention unique insights or notable differences.
> 5) Do NOT use “both”; say “Model 1 and Model 2” or “all models”.
> 6) Ensure the response is never empty.
> 7) After the first sentence, don't repeat “The models agree that …”; just list the content.
"""

    print("-----------------SUMMARY PROMPT------------------", int(time.time()))
    print(summary_prompt)
    # Build input content for Llama
    content = summary_prompt + "\n\n"
    for idx, res in enumerate(successful_answers, start=1):
        # Preserve supplied model name for 'consensus' mode; for 'blaze' we still pass it in,
        # but the instructions tell Llama NOT to use them in the output.
        model_label = res.get('model', f"Model {idx}")
        ans = (res.get('answer') or "").strip()
        if ans:
            content += f"{model_label} said:\n{ans}\n\n"

    # Call Llama on Together
    llm = together_client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {"role": "system", "content": "You compare AI answers and produce clean, structured summaries."},
            {"role": "user", "content": content}
        ],
        temperature=0.2,
        max_tokens=4000,
    )
    print("-----------------SUMMARY ANSWERED------------------", int(time.time()))
    return (llm.choices[0].message.content or "").strip()
    # content = summary_prompt + "\n\n"
    # for res in successful_answers:
    #     if res['answer'] != "" :
    #         content += f"{res['model'].capitalize()} said: {res['answer']}\n\n"

    # response = deepseek_client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant who compares AI answers and summarizes consensus or differences."},
    #         {"role": "user", "content": content}
    #     ],
    # )
    # return response.choices[0].message.content.strip()

async def get_opinion(responses, mode):
    try:
        answer = await asyncio.to_thread(summarize_opinion, responses, mode)
        print("-----------------SUMMARY ANSWERED------------------", int(time.time()))
        print(answer)
        return answer
    except Exception as e:
        raise RuntimeError(f"Summarizing failed: {str(e)}")

#Pick besk answer from answers of AI mdoels
def pick_best_answer(responses):
    # # Here you can use heuristics or even send all to GPT for ranking
    # print("---------------PICK BEST ANSER---------------", int(time.time()))
    # valid_answers = [res for res in responses if res["status"] == "success"]
    # if len(valid_answers) < 2:
    #     return valid_answers[0]["answer"] if valid_answers else "No valid answers."
    # prompt = f"""
    # Multiple answers are provided for the same question, generated by different AI models.

    # Choose the answer that the user understands and is based on accurate explanation and evidence, detailed.

    # Please answer only with the selected answer. Do not include explanation, evidence, or additional explanation.
    # """

    # content = prompt + "\n\n"
    # for i, res in enumerate(valid_answers):
    #     content += f"Answer:\n{res['answer']}\n\n"

    # response = deepseek_client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": "You are a critical evaluator of AI-generated responses."},
    #         {"role": "user", "content": content}
    #     ],
    # )
    # print("---------------PICKED BEST ANSER---------------", int(time.time()))
    # return "**Consensus:**\n" + response.choices[0].message.content.strip()
    print("---------------PICK BEST ANSWER---------------", int(time.time()))
    valid_answers = [res for res in responses if res["status"] == "success" and (res.get("answer") or "").strip()]
    if len(valid_answers) == 0:
        return "No valid answers."
    if len(valid_answers) == 1:
        return "**Consensus:**\n" + valid_answers[0]["answer"]

    ranking_prompt = """
You are tasked with evaluating multiple answers produced by different AI models for the same user question.
Your role is to act as a strict evaluator who selects the SINGLE best response.

Follow these steps carefully:

1. Read and compare all provided answers thoroughly. 
   - Treat each answer as if it came from a candidate in an expert competition.
   - Consider accuracy, clarity, completeness, and practical usefulness.

2. Rank the answers according to the following criteria (in priority order):

   (a) **Factual Accuracy**
   - Verify that the answer is free of errors, hallucinations, or misleading claims.
   - Disqualify or penalize any answer that fabricates information or contradicts well-known facts.

   (b) **Clarity and Completeness**
   - Prefer answers that are easy to understand, logically structured, and self-contained.
   - The best answer should cover the likely follow-up questions without requiring additional clarification.

   (c) **Actionable Structure**
   - Reward answers that use steps, bullet points, or well-formatted sections.
   - For example: clear lists of pros/cons, step-by-step methods, or bullet-pointed explanations.

   (d) **Conciseness**
   - The best answer balances detail with brevity: no unnecessary filler, but also no missing essentials.
   - Avoid overly short summaries that leave gaps in understanding.

3. After applying these criteria, select the SINGLE best answer.
   - Do not merge or blend answers.
   - Do not try to rewrite or improve them.
   - Output the best one exactly as given.

**STRICT OUTPUT INSTRUCTIONS**
- Output ONLY the full text of the winning answer.
- Do NOT add commentary, rankings, explanations, or meta-text.
- Do NOT prepend or append anything like "The best answer is:" — just return the answer itself.
"""

    content = ranking_prompt + "\n\n"
    for i, res in enumerate(valid_answers, start=1):
        content += f"Answer {i}:\n{res['answer'].strip()}\n\n"

    # Call Llama on Together
    llm = together_client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {"role": "system", "content": "You are a strict evaluator that selects the best single answer verbatim."},
            {"role": "user", "content": content}
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    chosen = (llm.choices[0].message.content or "").strip()
    print("---------------PICKED BEST ANSWER---------------", int(time.time()))
    return "**Consensus:**\n" + chosen

async def get_best_answer(responses):
    try:
        answer = await asyncio.to_thread(pick_best_answer, responses)
        return answer
    except Exception as e:
        raise RuntimeError(f"Picking best answer failed: {str(e)}")
#Generate answer of each models using gathering 3/5/7
def generate_answers(mode, level, history, prompt, question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    print("-------------Prompt------------", level)
    print(prompt)
    print("-------------Question-------------", int(time.time()))
    print(question)
    if mode == 'blaze':
        tasks = [
            get_gpt4o_answer(history, prompt, question),
            get_llama_answer(history, prompt, question),
        ]
    else:
        if level == 'Easy':
            tasks = [
                get_llama_answer(history, prompt, question),
                get_deepseek_answer(history, prompt, question),
                get_grok_answer(history, prompt, question),
            ]
        if level == 'Medium':
            tasks = [
                get_gemini_answer(history, prompt, question),
                get_mistral_answer(history, prompt, question),
                get_llama_answer(history, prompt, question),
                get_deepseek_answer(history, prompt, question),
                get_grok_answer(history, prompt, question),
            ]
        if level == 'Complex':
            tasks = [
                get_gpt4o_answer(history, prompt, question),
                get_claude_answer(history, prompt, question),
                get_gemini_answer(history, prompt, question),
                get_mistral_answer(history, prompt, question),
                get_llama_answer(history, prompt, question),
                get_deepseek_answer(history, prompt, question),
                get_grok_answer(history, prompt, question),
            ]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    print("-------------END_Answers-------------", int(time.time()))
    return results

def _fallback_image_keywords_from_text(text: str):
    terms = extract_key_terms(text or "") or ["overview", "context", "details"]
    top_mods = ["overview", "real world", "context", "landscape", "scene",
                "wide", "example", "interface", "architecture", "in practice"]
    bottom_mods = ["close up", "diagram", "workflow", "comparison", "steps",
                   "how to", "setup", "configuration", "metrics", "results"]

    def expand(mods):
        out = []
        for m in mods:
            for t in terms:
                out.append(f"{t} {m}".strip())
        return out

    top_keywords = (expand(top_mods) or top_mods)[:30]
    bottom_keywords = (expand(bottom_mods) or bottom_mods)[:30]
    if len(top_keywords) < 10: top_keywords = (top_keywords * ((10 // max(1, len(top_keywords))) + 1))[:10]
    if len(bottom_keywords) < 10: bottom_keywords = (bottom_keywords * ((10 // max(1, len(bottom_keywords))) + 1))[:10]
    return top_keywords[:30], bottom_keywords[:30]

#Generate image keywords
def get_image_keywords_from_responses(responses):
    """
    Input: same as pick_best_answer(responses) -> list of {model, answer, status, ...}
    Output: (top_keywords, bottom_keywords)  # each >= 10 strings
    Uses Together Llama. Deterministic fallback if JSON invalid or short.
    """
    print("---------------IMAGE KEYWORDS---------------", int(time.time()))
    valid_answers = [res for res in responses if res.get("status") == "success" and (res.get("answer") or "").strip()]
    if not valid_answers:
        return _fallback_image_keywords_from_text("")

    parts = []
    for i, res in enumerate(valid_answers, start=1):
        parts.append(f"Answer {i}:\n{res['answer'].strip()}\n")
    bundle = "\n".join(parts)

    prompt = f"""
You will receive multiple ANSWERS to the same question.

Task:
1) Skim the answers and anchor on the strongest/clearest content.
2) Produce STRICT JSON ONLY in this exact shape:
   {{"top_keywords": ["..."], "bottom_keywords": ["..."]}}

Rules:
- Each array MUST contain AT LEAST 5 concise queries (3-6 words each).
- Queries must be tightly related to the anchored answer's concrete entities (names, models, orgs, places, versions, dates if helpful).
- "top_keywords" = hero/context image queries (wide/scene/overview).
- "bottom_keywords" = detail/diagram/comparison/workflow image queries.
- Avoid generic words like "image", "photo", "picture", "wallpaper".
- No explanations, no extra text — JSON only.

ANSWERS:
{bundle}
""".strip()

    try:
        resp = together_client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900,
        )
        raw = (resp.choices[0].message.content or "").strip()
        i, j = raw.find("{"), raw.rfind("}")
        if i == -1 or j == -1:
            raise ValueError("No JSON object found from Llama.")

        obj = json.loads(raw[i:j+1])
        top_keywords = [str(x).strip() for x in (obj.get("top_keywords") or []) if str(x).strip()]
        bottom_keywords = [str(x).strip() for x in (obj.get("bottom_keywords") or []) if str(x).strip()]

        # ensure >= 10 each; pad deterministically if needed
        if len(top_keywords) < 10 or len(bottom_keywords) < 10:
            seed_text = "\n\n".join([r["answer"] for r in valid_answers])
            fb_top, fb_bottom = _fallback_image_keywords_from_text(seed_text)
            while len(top_keywords) < 10:
                top_keywords.append(fb_top[len(top_keywords) % len(fb_top)])
            while len(bottom_keywords) < 10:
                bottom_keywords.append(fb_bottom[len(bottom_keywords) % len(fb_bottom)])

        print("---------------IMAGE KEYWORDS DONE-----------", int(time.time()))
        return top_keywords[:30], bottom_keywords[:30]

    except Exception as e:
        print(f"image keywords (llama) error: {e}")
        seed_text = "\n\n".join([r["answer"] for r in valid_answers]) if valid_answers else ""
        return _fallback_image_keywords_from_text(seed_text)

#Generate image keywords
async def get_keywords(responses):
    """
    Async wrapper so keywords are generated concurrently with get_best_answer and get_opinion.
    Returns: {"top_keywords": [...], "bottom_keywords": [...], "status": "success|failed"}
    """
    print("start keywords", int(time.time()))
    try:
        top_keywords, bottom_keywords = await asyncio.to_thread(get_image_keywords_from_responses, responses)
        print("end keywords", int(time.time()))
        print(top_keywords, bottom_keywords)
        return {"top_keywords": top_keywords, "bottom_keywords": bottom_keywords, "status": "success"}
    except Exception as e:
        print(f"keywords error: {e}")
        return {"top_keywords": [], "bottom_keywords": [], "status": "failed"}

#Pick best answer and summarize the opinion from answers of each models
def analyze_result(results, mode):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("\n------------ANALYZE RESULT------------\n", int(time.time()))
    print(results)
    summarize = loop.run_until_complete(asyncio.gather(
        get_best_answer(results),
        get_opinion(results, mode),
        get_keywords(results)  
    ))
    best_answer, opinion, keywords = summarize
    loop.close()
    print("\n------------END ANALYZE RESULT------------\n", int(time.time()))
    return best_answer, opinion, keywords

#Generate answer of user asked(findal_answer: main answer, status_report: AI models success report, opinion: summarized opinion)
def get_answer(mode, level, history, prompt, question):
    results = generate_answers(mode, level, history, prompt, question)
    status_report = [
        {key: value for key, value in result.items() if key != 'answer'}
        for result in results
    ]
    best_answer, opinion, keywords = analyze_result(results, mode)

    if mode == 'consensus':
        final_answer_with_images = insert_images(best_answer, question, keywords)
    if mode == 'blaze':
        final_answer_with_images = best_answer

    return {
        "final_answer": final_answer_with_images,
        "status_report": status_report,
        "opinion": opinion
    }

def format_top_image(img_data):
    """Format top image HTML with full width"""
    return f"""
    <div class="image-container-field" onclick="window.open('{img_data['thumbnail']}')">
        <img src="{img_data['thumbnail']}" 
             alt="{img_data['title']}" 
             class="full-width-image responsive-img">
        <div class="image-caption">{img_data['title']}</div>
    </div>
    """

def format_bottom_image(img_data):
    """Format bottom image HTML with centered large display"""
    return f"""
    <div style="text-align: center; margin: 10px 0;">
        <div class="bottom-image-container" onclick="window.open('{img_data['thumbnail']}')">
            <img src="{img_data['thumbnail']}" 
                 alt="{img_data['title']}" 
                 style="max-width: 90%; height: auto; margin: 0 auto; display: block; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            <div class="image-caption" style="text-align: center; margin-top: 10px; font-size: 0.9em;">{img_data['title']}</div>
        </div>
    </div>
    """

def insert_images(answer_text, query, keywords):
    """
    Insert only 2 relevant images into the answer text:
    - 1 big image at the top (from top_keywords candidates)
    - 1 big centered image at the bottom (from bottom_keywords candidates)
    **Rules**
    - Always insert correct images related to the content
    - 2 images must be different
    """
    paragraphs = [p.strip() for p in re.split(r'(?<=\n\n)(?=\S)', answer_text or '') if p.strip()]
    if not paragraphs:
        paragraphs = [answer_text or ""]

    # Safely read keyword arrays
    kw = keywords or {}
    top_kw = kw.get('top_keywords') or []
    bottom_kw = kw.get('bottom_keywords') or []

    # Build candidate queries (use a few best items)
    # Prepend the original query for extra context
    top_queries = [f"{query} {k}".strip() for k in top_kw][:5]
    bottom_queries = [f"{query} {k}".strip() for k in bottom_kw][:5]

    if not top_queries and not bottom_queries:
        # No keywords; just return original text
        return "\n\n".join(paragraphs)

    # Fetch images concurrently and pick the first valid from each group
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        top_tasks = [fetch_image(q) for q in top_queries] if top_queries else []
        bottom_tasks = [fetch_image(q) for q in bottom_queries] if bottom_queries else []

        top_results, bottom_results = loop.run_until_complete(asyncio.gather(
            asyncio.gather(*top_tasks) if top_tasks else asyncio.sleep(0, result=[]),
            asyncio.gather(*bottom_tasks) if bottom_tasks else asyncio.sleep(0, result=[])
        ))
    finally:
        loop.close()

    def first_image(cands):
        for c in cands:
            if isinstance(c, dict) and c:
                return c
        return None

    top_img = first_image(top_results)
    bottom_img = first_image(bottom_results)

    # Avoid using the same image twice (compare by URL/thumbnail)
    if top_img and bottom_img:
        top_url = top_img.get("thumbnail") or top_img.get("url")
        bot_url = bottom_img.get("thumbnail") or bottom_img.get("url")
        if top_url and bot_url and top_url == bot_url:
            # Try to find a different bottom image
            bottom_img = first_image([c for c in bottom_results if c and (c.get("thumbnail") or c.get("url")) != top_url])

    # Build the final content
    results = []
    if top_img:
        results.append(format_top_image(top_img))

    results.extend(paragraphs)

    if bottom_img:
        results.append(format_bottom_image(bottom_img))

    return "\n\n".join(results)

async def fetch_image(search_query):
    try:
        params = {
            "engine": "google_images",
            "q": search_query,
            "api_key": serpapi_key,
            "num": 3,  # Get multiple results to find best quality
            "safe": "active",
            "hl": "en",
            "img_size": "large",  # Only get large images
            "img_type": "photo"   # Only get photos (not clipart etc)
        }
        
        response = requests.get('https://serpapi.com/search', params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("images_results"):
                # Find the highest resolution image available
                best_image = max(
                    data["images_results"],
                    key=lambda x: x.get("original_width", 0) * x.get("original_height", 0),
                    default=None
                )
                if best_image:
                    return {
                        "url": best_image.get("original"),
                        "thumbnail": best_image.get("original"),  # Use original for both
                        "title": best_image.get("title", search_query),
                        "source": best_image.get("source", "")
                    }
    except Exception as e:
        print(f"High quality image fetch error: {e}")
    return None

#Get news using brave API
def fetch_brave_news(query):
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key
    }

    params = {
        "q": query,
        "count": 10,  # number of results (max: 20 for free tier),
        # If Brave supports a freshness param in your tier, include it; examples: 'pd','pw','pm','py'
        # Comment out if your plan doesn't accept it.
        "freshness": "py"
    }

    response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
    data = response.json()
    results = data.get("web", {}).get("results", [])

    return [{
        "title": r.get("title", ""),
        "snippet": r.get("description", ""),
        "source": r.get("source") or urlparse(r.get("url", "")).netloc,
        "url": r.get("url", "")
    } for r in results]

#Get news from google search
def fetch_google_pse_news(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_search_api_key,
        "cx": google_cx_id,
        "q": f"site:reuters.com OR site:bloomberg.com OR site:wsj.com OR site:cnbc.com {query}",
        "num": 10,
        # Prefer recent results; 'dateRestrict' restricts freshness (y1 = last 1 year; m3 = last 3 months)
        "dateRestrict": "y1",
        # 'sort=date' requests time ordering where supported by the CSE
        "sort": "date"
    }
    response = requests.get(url, params=params)
    results = response.json().get('items', [])
    return [{
        "title": r.get("title", ""),
        "snippet": r.get("snippet", ""),
        "source": r.get("displayLink", ""),
        "url": r.get("link", ""),
    } for r in results]

#Set the priority of searching websites(Search Reuters, Bloomberg, WSJ, CNBC first and search other sites)
def prioritize_sources(articles):
    trusted_sources = ["Reuters", "Bloomberg", "WSJ", "CNBC"]
    prioritized = [a for a in articles if any(ts.lower() in a['source'].lower() for ts in trusted_sources)]
    others = [a for a in articles if a not in prioritized]
    return prioritized + others

#Combine all news search results in one text and compress
def compress_articles(articles, max_tokens=700):
    combined_text = ""
    for article in articles:
        combined_text += f"{article['source']}: {article['snippet']} "
    # Dummy compression (replace with real summarizer later)
    compressed = combined_text[:4000]  # Rough cut, to replace with smart compression
    return compressed

#Get news using google search and brave API(query, sources_used, compressed_summary: news searched result)
def retrieve_news(query):
    # Fetch from two APIs
    google_articles = fetch_google_pse_news(query)
    brave_articles = fetch_brave_news(query)

    # Merge + deduplicate
    all_articles = google_articles + brave_articles
    print("------All Articles---------------")
    print(int(time.time()))
    print(all_articles)
    seen_urls = set()
    unique_articles = []
    for a in all_articles:
        if a['url'] not in seen_urls:
            unique_articles.append(a)
            seen_urls.add(a['url'])

    # Prioritize trusted sources
    sorted_articles = prioritize_sources(unique_articles)

    # Compress for model
    compressed_summary = compress_articles(sorted_articles)

    return {
        "query": query,
        "sources_used": list({a['source'] for a in sorted_articles[:4]}),
        "compressed_summary": compressed_summary
    }

#Analyze the news by AI models using gathering. 3/5/7
def generate_news(mode, level, history, prompt, question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    print("----------------generate_new--------------", level, int(time.time()))
    if mode == 'blaze':
        tasks = [
            get_gpt4o_answer(history, prompt, question),
            get_llama_answer(history, prompt, question),
        ]
    else:
        if level == 'Easy':
            tasks = [
                get_llama_answer(history, prompt, question),
                get_deepseek_answer(history, prompt, question),
                get_mistral_answer(history, prompt, question),
            ]
        if level == 'Medium':
            tasks = [
                get_gemini_answer(history, prompt, question),
                get_mistral_answer(history, prompt, question),
                get_llama_answer(history, prompt, question),
                get_deepseek_answer(history, prompt, question),
                get_gpt4o_answer(history, prompt, question),
            ]
        if level == 'Complex':
            tasks = [
                get_gpt4o_answer(history, prompt, question),
                get_claude_answer(history, prompt, question),
                get_gemini_answer(history, prompt, question),
                get_mistral_answer(history, prompt, question),
                get_llama_answer(history, prompt, question),
                get_deepseek_answer(history, prompt, question),
                get_grok_answer(history, prompt, question),
            ]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    print("----------------end_generate_new--------------", int(time.time()))
    loop.close()

    return results

#Get the news using google search and brave API. After that Analyze the searched results using AI models
def get_news(mode, level, query):
    result = retrieve_news(query)
    print('---------------start analyzing news----------------', int(time.time()))
    prompt = "Summarize the provided news search results clearly and objectively for a human reader."

    question = f"""
    Summarize the topic: "{query}"

    Use only the search results provided below. Do not add information from other sources. The goal is to create a clear, easy-to-read news summary that emphasizes the latest developments.

    **Focus first on the most recent and time-sensitive information**, then include relevant background or supporting context as needed.

    Guidelines:
    - Start with the most recent events or updates (especially those from the last 30 days). Please answer based on newest data and news.
    - Highlight specific facts: dates, names, companies, places, decisions, or statistics
    - Summarize only what is reported in the provided search results
    - Avoid speculation, repetition, or vague commentary
    - If multiple viewpoints or updates exist, summarize the differences concisely

    Use a clear, professional tone. Present the summary in bullet points or short paragraphs for quick readability.

    Search Results:
    {result['compressed_summary']}
    """

    results = generate_news(mode, level, [], prompt, question)
    print("-------News for Generated from news-----------", int(time.time()))
    print(results)

    status_report = [
        {key: value for key, value in result.items() if key != 'answer'}
        for result in results
    ]
    best_answer, opinion, keywords = analyze_result(results, mode)

    if mode == 'consensus':
        final_answer = insert_images(best_answer, query, keywords)
    if mode == 'blaze':
        final_answer = best_answer
    return {
        "final_answer": final_answer,
        "status_report": status_report,
        "opinion": opinion
    }

#Test route
@openai_bp.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.get_json()
    question = data.get("question")
    return retrieve_news(question)

#level: question complexity, last_year: user want recent_data or not, used_in_context: user asked question is related with previous question or not.
#updated_question: if used_in_context is yes, generated updated_question(ex. question: explain more about above product. updated_question: explain more about PS5 )
#product: user wanted products list
def judge_system(mode, question, history = []):
    print("------------start judge system------------", int(time.time()))
    history_text = ""
    for item in history:
        text = item.get('text', '')  # This won't raise KeyError
        prefix = "User" if item["type"] == "question" else "Assistant"
        history_text += f"{prefix}: {text}\n"

    if mode == 'blaze':
        expected_output = """
            {
                "level": "blaze",
                "last_year": "Yes",
                "used_in_context": true or false,
                "updated_question": "Rewritten question if used_in_context is true, otherwise empty string",
                "product": []
            }
        """
        judge_prompt = f"""
        You are an AI assistant responsible for classifying user queries.

        1. **Difficulty Assessment**: Categorize the complexity of the question as one of the following: "Easy", "Medium", or "Complex".
        
        2. Determine if the user's query requires information newer than your knowledge cutoff date. Use these rules:

            1) Respond ONLY with "Yes" if:
                - The question asks about events/developments after your cutoff
                - Uses terms like "current", "latest", "recent", "now", "this year"
                - Concerns rapidly-changing topics (tech, medicine, news, events)
                - Dealing with phenomena(events, tech, news, medicine) that may occur in the past or present

            2) For ambiguous cases where time isn't specified but the topic evolves:
                - Default to "Yes" for safety
                
            3) Respond ONLY with "No" if:
                - No time-sensitive information is requested
                - The subject matter is static 

            Never explain your reasoning - only output "Yes" or "No".

        3. **used_in_context**: Determine if the current question depends on or references the history. Say `true` if it's a follow-up or refers to anything previously discussed. Say `false` if it's a completely new topic.

        4. **updated_question**:
            - If `used_in_context` is `true`, rewrite the question so that it is fully self-contained, including the necessary context from the conversation history.
            - If `used_in_context` is `false`, return an empty string.
        5. No product search at all.
            - Just return empty array.
            - Product search is not part of the task.
        6. The generation should be very fast and concise.
        Return your answer strictly in the following JSON format: {expected_output}

        Conversation history:
        {history_text}

        Current question:
        {question}

        """
    if mode == 'consensus':
        expected_output = """
            {
                "level": "Easy"
                "last_year": "Yes"
                "used_in_context": true or false,
                "updated_question": "Rewritten question if used_in_context is true, otherwise empty string"
                "product": ["Product A", "Product B"] or []
            }
        """
        judge_prompt = f"""
        You are an AI assistant responsible for classifying user queries.

        1. **Difficulty Assessment**: Categorize the complexity of the question as one of the following: "Easy", "Medium", or "Complex".
        
        2. Determine if the user's query requires information newer than your knowledge cutoff date. Use these rules:

            1) Respond ONLY with "Yes" if:
                - The question asks about events/developments after your cutoff
                - Uses terms like "current", "latest", "recent", "now", "this year"
                - Concerns rapidly-changing topics (tech, medicine, news, events)
                - Dealing with phenomena(events, tech, news, medicine) that may occur in the past or present

            2) For ambiguous cases where time isn't specified but the topic evolves:
                - Default to "Yes" for safety
                
            3) Respond ONLY with "No" if:
                - No time-sensitive information is requested
                - The subject matter is static 

            Never explain your reasoning - only output "Yes" or "No".

        3. **used_in_context**: Determine if the current question depends on or references the history. Say `true` if it's a follow-up or refers to anything previously discussed. Say `false` if it's a completely new topic.

        4. **updated_question**:
            - If `used_in_context` is `true`, rewrite the question so that it is fully self-contained, including the necessary context from the conversation history.
            - If `used_in_context` is `false`, return an empty string.

        5. **Product Intent Detection**:  

            You MUST extract all explicitly mentioned buyable consumer products and close paraphrases (brand + model/family), including ALL sides of any comparison, “worth buying X if you have Y”, or “X vs Y” constructions.
            
            Rules:
            - Include every product explicitly named or strongly implied as a concrete retail item.
            - Include all sides of comparisons or conditional questions (e.g., “Is X worth it if I have Y?” → extract both X and Y).
            - For different types of products (e.g., “X vs Y” → extract both X and Y).
            - Keep names as they appear (brand + model), but be consistent and concise (e.g., “PlayStation 5”, “Nintendo Switch 2”).
            - Do NOT include vague categories (e.g., “a console”, “a controller”) unless a specific retail product is named.
            - Deduplicate. Be case-insensitive.

            Return an array. If none, return [].

            Normalization/Aliases (non-exhaustive examples the model should recognize as the same products):
            - PlayStation 5 ≡ PS5
            - Nintendo Switch 2 ≡ Switch 2, Nintendo Switch (2nd gen)

        Return your answer strictly in the following JSON format: {expected_output}

        Conversation history:
        {history_text}

        Current question:
        {question}

        """

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": question}
    ]

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.1
    )

    judge_output = response.choices[0].message.content
    index_open_brace = judge_output.find('{')
    index_open_bracket = judge_output.find('[')
    first_pos = min(p for p in [index_open_brace, index_open_bracket] if p != -1)

    index_close_brace = judge_output.rfind('}')
    index_close_bracket = judge_output.rfind(']')
    last_pos = max(p for p in [index_close_brace, index_close_bracket] if p != -1)

    judge_output = judge_output[first_pos:last_pos + 1]
    judge_output = json.loads(judge_output)
    print("--------------end_judge_system--------------", int(time.time()))
    print(judge_output)
    return judge_output

# Test route
@openai_bp.route('/judge', methods=['POST'])
def judge():
    data = request.get_json()
    question = data.get('question')
    return judge_system(question)

def extract_key_terms(text):
    """Enhanced keyword extraction with noun phrase detection"""
    # Remove special characters and split
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = re.findall(r'\b\w{3,}\b', clean_text.lower())
    
    # Custom stopwords list
    stopwords = {
        'the', 'a', 'an', 'in', 'on', 'at', 'and', 'or', 
        'of', 'to', 'is', 'are', 'was', 'were', 'for', 'with'
    }
    
    # Filter and count
    filtered = [w for w in words if w not in stopwords]
    freq = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1
    
    # Get top 3 most frequent meaningful words
    return sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:3]

@openai_bp.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    chat_id = data.get("chat_id")
    user_id = data.get("user_id")
    history = data.get("history")
    question = data.get("question")
    mode = data.get("mode")
    # valid = check_valid(user_id)
    #
    # if valid == False:
    #     return jsonify({"level": "subscribe"})
    # if not question:
    #     return jsonify({"error": "Question is required"}), 400

    # return jsonify({"level": "subscribe"})
    prompt = f"""
You are a knowledgeable and objective AI assistant.  
Your primary task is to generate answers that are clear, accurate, and helpful in natural language.  
Every response should feel complete, structured, and directly useful to the user.  

=========================
GENERAL RULES  
=========================
- Always provide the latest and most up-to-date information available at the time of answering.  
- Use clear formatting: headings, subheadings, bullet points, numbered steps, and tables where appropriate.  
- Always provide **actionable guidance or decisions**, not just summaries.  
- Do not use filler phrases like “as of my last update.”  
- Keep answers concise but **fully complete** — no missing details.  
- Ensure all facts are **100% correct**.  
- Always include the **latest events, changes, or updates** relevant to the question.  
- If something is uncertain or context-dependent, explain the possible cases and give practical recommendations.  
- Always observe and analyze all relevant issues from the perspective of the current moment, ensuring that the information reflects the most up-to-date context available at the time of answering.
- If possible, provide final conclusions or recommendations based on the information provided.
=========================
COMPARISON QUESTIONS  
=========================
When comparing products, technologies, or concepts:  
1. Identify all items clearly, and add any important missing options.  
2. For each item, explain:  
   - What it is / Key features  
   - Strengths (advantages)  
   - Limitations (trade-offs)  
   - Best-fit use cases (who should use it and why)  
3. Highlight the **meaningful differences** between items (performance, ecosystem, cost, usability, compliance, learning curve, etc.).  
4. Provide a short **decision guide** that explains when to choose which option.  
5. If helpful, include a **comparison table** for clarity.  

=========================
PRODUCT RECOMMENDATIONS  
=========================
- Always include **exact models or versions**, standout features, typical price range, and any known caveats.  
- If the prompt includes the pattern:  
  `Recommend Products: ---PRODUCTS---`  
  Then follow these rules:  
  - Treat each listed product as **Rank 1** in its own category.  
  - Explain why it deserves Rank 1 (unique strengths, ideal user).  
  - Suggest **Rank 2 and Rank 3 alternatives** for different budgets or needs.  
- Provide practical buying advice and considerations (e.g., warranty, ecosystem, compatibility).  
- Always provide correct products the user intends.

=========================
HOW-TO & TROUBLESHOOTING  
=========================
When explaining processes or fixing problems:  
- Start with a short **overview** of the goal.  
- List **prerequisites** (tools, versions, permissions, environment).  
- Provide a **step-by-step numbered guide**.  
- Add a way to **verify success** at the end of the steps.  
- Mention **common pitfalls or mistakes** and how to fix them.  
- End with **optional improvements or next steps** the user can take.  

=========================
DEFINITIONS & CONCEPTS  
=========================
When defining or explaining a concept:  
- Give a **one-sentence definition**.  
- Explain **why it matters / its practical relevance**.  
- Provide **one simple, relatable example**.  
- Mention **related terms** and explain how they differ.  

=========================
NEWS & TIME-SENSITIVE TOPICS  
=========================
When the topic involves recent events or fast-changing information:  
- Provide the **latest confirmed updates with exact dates** (e.g., “On 2 September 2025…”).  
- Explain clearly **what changed** and **why it matters**.  
- Summarize the **impact or implications** of these updates.  
- End with a short **“What's next / possible outcomes”** section.  

=========================
FULL REPORT MODE  
=========================
If the user explicitly requests a full/academic report, always include:  
1. **Introduction** - context, background, objectives.  
2. **Methods** - data sources, assumptions, evaluation criteria, analysis steps.  
3. **Results** - findings (tables or figures if useful).  
4. **Discussion** - interpretation, trade-offs, limitations, and alternative views.  
5. **Conclusion** - key takeaways and recommended actions.  
6. **References** - concise, relevant sources (include only if report mode or explicitly required).  

=========================
STYLE & TONE  
=========================
- Neutral, professional, and **user-friendly**.  
- Use plain, everyday English — avoid unnecessary jargon.  
- Structure text with short paragraphs, clear bullets, and compact tables.  
- Always focus on clarity and usability over complexity.  

=========================
CHECKLIST BEFORE FINALIZING  
=========================
- Is the answer up-to-date with the latest events and details?  
- Are all facts accurate and double-checked?  
- Is the response concise but **fully complete**?  
- Does it provide actionable guidance?  
- Is it clear, well-structured, and easy to read?  
- Does it completely cover the user's intent?  

    """

    print(prompt)
    print('start prompt', int(time.time()))
    judge_output = judge_system(mode, question, history)
    if judge_output['used_in_context'] == False:
        history = []
    else:
        question = judge_output['updated_question']
    print("Judge_output", len(judge_output['product']))
    print(int(time.time()))
    print(judge_output)

    if len(judge_output['product']) > 0:
        if mode == 'consensus':
            random.shuffle(judge_output['product'])
            judge_output['level'], result = analyze_product(mode, judge_output['level'], judge_output['last_year'], history, prompt, question)
    elif judge_output['last_year'] == 'Yes':
        print("getting news", int(time.time()))
        result = get_news(mode, judge_output['level'], question)
        print("got news", int(time.time()))
    else:
        result = get_answer(mode, judge_output['level'], history, prompt, question)

    token = Devices.query.filter_by(device_id=user_id).first()
    
    if token:
        user_id = token.email
    
    new_history = ChatHistory(user_id = user_id, answer = result["final_answer"], status_report = json.dumps(result["status_report"]), opinion = result["opinion"], chat_id = chat_id, question = question, mode = mode, level = judge_output['level'], created_at = datetime.now(), updated_at = datetime.now())
    db.session.add(new_history)
    db.session.commit()
    return jsonify({"question": question, "mode": mode, "answer": result["final_answer"], "level": judge_output['level'], "status_report": result["status_report"], "opinion": result["opinion"]})

async def generate_answer(mode, level, history, prompt, query):
    try:
        answer = await asyncio.to_thread(get_answer, mode, level, history, prompt, query)
        return answer
    except Exception as e:
        raise RuntimeError(f"Generating answer failed: {str(e)}")

def check_valid(user_id):
    if not user_id:
        return jsonify({"detail": "Missing device_id"}), 400

    valid = AccessKey.query.filter_by(device_id=user_id).first()

    # Debug: Check if 'valid' exists and has 'count' attribute
    if not valid:
        print(f"No AccessKey found for device_id: {user_id}")
        return False

    if not hasattr(valid, 'count'):
        print(f"AccessKey for {user_id} has no 'count' attribute")
        return False

    now = datetime.utcnow()
    count = valid.count
    if valid.valid_date > now:  # Check date validity
        return True
    elif count > 0:  # Check remaining uses
        valid.count = count - 1
        try:
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            return False
    return False

async def search_news(mode, level, query):
    try:
        answer = await asyncio.to_thread(get_news, mode, level, query)
        return answer
    except Exception as e:
        raise RuntimeError(f"Searching news failed: {str(e)}")

async def get_product(query):
    try:
        answer, is_specific_model = await asyncio.to_thread(search_product, query)
        return answer
    except Exception as e:
        raise RuntimeError(f"Searching product failed: {str(e)}")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _product_key(p: dict):
    """
    Prefer stable ID; else fallback to normalized title/name.
    """
    if isinstance(p, dict):
        if p.get("id"):
            return ("id", _norm(p["id"]))
        title = p.get("title") or p.get("name") or p.get("product")
        if title:
            return ("title", _norm(title))
    # last resort: stable hash of the dict
    return ("blob", _norm(json.dumps(p, sort_keys=True, ensure_ascii=False)))

def _merge(a: dict, b: dict) -> dict:
    """
    Shallow merge: keep existing non-empty values in `a`,
    fill missing/empty fields from `b`.
    """
    out = dict(a)
    for k, v in b.items():
        if k not in out or out[k] in (None, "", [], {}):
            out[k] = v
    return out

def combine_product_results(product_results):
    """
    product_results is typically a list where each item is either:
      - a list[dict] (preferred), or
      - a single dict, or
      - None
    Returns: list[dict] flattened, deduped, original order preserved.
    """
    # 1) Flatten
    flat = []
    for chunk in product_results:
        if not chunk:
            continue
        if isinstance(chunk, list):
            flat.extend([x for x in chunk if x])
        elif isinstance(chunk, dict):
            flat.append(chunk)
        else:
            # ignore unexpected shapes
            continue

    # 2) Deduplicate + stable order, merging fields when duplicates appear
    seen = OrderedDict()
    for p in flat:
        key = _product_key(p)
        if key in seen:
            seen[key] = _merge(seen[key], p)
        else:
            seen[key] = p

    return list(seen.values())

#Generate answer of user asked question and search products mentioned in user asked question.
#If last_year: Yes, get news and analyze news with AI models 3/5/7. After that combine answer and searched products.
#IF last_year: No, generate answer with AI models 3/5/7. After that combine answer and searched products.
def compare_product(mode, level, last_year, history, prompt, query, products):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Kick off product fetches
    product_tasks = [get_product(product) for product in products]
    product_results = loop.run_until_complete(asyncio.gather(*product_tasks))

    print("--------Product tasks----------")
    # Combine ALL product results here
    product_data = combine_product_results(product_results)

    # shuffle product_data
    random.shuffle(product_data)

    # Inject combined products into the prompt (instead of raw product_results)
    prompt = prompt.replace("---PRODUCTS---", json.dumps(product_data, ensure_ascii=False))
    print("----------Adjust Prompt-----------")
    print(prompt)

    # Run analysis
    if last_year.lower() == 'yes':
        analyze = loop.run_until_complete(asyncio.gather(
            search_news(mode, level, query),
        ))
    else:
        analyze = loop.run_until_complete(asyncio.gather(
            generate_answer(mode, level, history, prompt, query),
        ))
    result = analyze[0]
    loop.close()

    answer = result["final_answer"]

    print("------------PRODUCT_RESULTS (combined)------------")
    # print(product_data)

    result["final_answer"] = json.dumps({
        "answer": answer,
        "products": product_data
    }, ensure_ascii=False)

    return result
#This function will called if user asked question with products. Generate the answer of user asked question.
#If user just want only looking products search products and return.
#If user want product's analyzed data, compared data, review and etc, Combine analyzed data with AI answers and products list.

def analyze_product(mode, level, last_year, history, prompt, query):
    analyze_prompt = f"""
    You are an AI assistant that analyzes product-related user queries.

    Perform the following steps:

    1. **Looking Only**: Determine whether the user is just looking for or exploring a product casually (e.g., searching, browsing, discovering). 
    - Respond with `"Yes"` if the intent is simply to look or search.
    - Respond with `"No"` if the user is seeking specific information beyond just looking.

    2. **User Intent**: If the answer to (1) is "No", classify the user's actual intent into one of the following:
    - "cost" (asking about price or affordability)
    - "review" (asking for feedback or evaluations)
    - "compare" (comparing two or more products)
    - "buy" (asking where or how to purchase)
    - "unknown" (if unclear)

    3. Product Identification: Extract and list all product names explicitly or implicitly referenced in the user query. If a product name is incomplete, ambiguous, or does not exactly match a known product, infer and return the most relevant and popular full product name from the product catalog that closely aligns with the original term. If no valid product reference is detected, return an empty list.

    Return your response in the following strict JSON format:
    {{
    "just_looking": "Yes" or "No",
    "intent": "search" | "cost" | "review" | "compare" | "buy" | "unknown",
    "products": ["Product A", "Product B"]
    }}

    Message: "{query}"
    """

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": analyze_prompt}
        ],
        temperature=0.1
    )

    response = response.choices[0].message.content
    print("---------analyze product------------")
    print(response)
    index_open_brace = response.find('{')
    index_open_bracket = response.find('[')
    first_pos = min(p for p in [index_open_brace, index_open_bracket] if p != -1)

    index_close_brace = response.rfind('}')
    index_close_bracket = response.rfind(']')
    last_pos = max(p for p in [index_close_brace, index_close_bracket] if p != -1)

    response = response[first_pos:last_pos + 1]
    response = json.loads(response)

    compare_prompt = f"""
    You are a helpful assistant that compares two products based on a user's question.

    A user asked:

    "{query}"

    Please do the following:

    1. Determine if the question involves comparing two specific products.
    2. Don't include table format data.
    3. If yes, extract the product names and generate a clear comparison in **Markdown format**, with the following structure:

    ### 🧾 Product Comparison: [Product A] vs [Product B]

    **Similarities**
    - Point 1
    - Point 2

    **Differences**
    - Point 1
    - Point 2

    **Pros and Cons**

    **Product A**
    ✅ Pros:
    - ...
    ❌ Cons:
    - ...

    **Product B**
    ✅ Pros:
    - ...
    ❌ Cons:
    - ...

    **🔍 Recommendation**
    > Your final advice based on typical user needs.

    If the input does not contain a comparison request, respond with:
    > **Not a product comparison question.**
    """

    # if response['just_looking'] == 'Yes':
    #     products, is_specific_model = search_product(query)
    #     result = {
    #         "final_answer": json.dumps(products),
    #         "status_report": [],
    #         "opinion": '',
    #     }
    #     return "specific_product" if is_specific_model.lower() == "yes" else "general_product", result
    # else:
    result = {}
    if response['intent'] == 'compare':
        prompt = compare_prompt
    result = compare_product(mode, level, last_year, history, prompt, query, response['products'])
    return "compare_product", result

#Analyzed the user asked product and get the exact information of product
def extract_info(query):
    prompt = f"""
    You are a product categorization assistant. Your job is to extract the most relevant Amazon category and category_id for a given product search query.

    Also identify whether the query refers to a specific product model (e.g., "Anker Soundcore Liberty 4 NC") or a general product category (e.g., "men's shoes", "laptops").

    Query: "{query}"

    Return your answer in the following JSON format:
    {{
    "search_term": "<cleaned version of query>",
    "category": "<best-matching Amazon category name>",
    "category_id": "<corresponding category ID>",
    "color": "<color mentioned in query>",
    "size": "<size mentioned in query>",
    "min_price": "<min price user wants, if specified>",
    "max_price": "<max price user wants, if specified>",
    "is_specific_model": "<Yes or No>"
    }}
    Be as precise as possible. If it's an exact model name, prioritize specific categories (e.g., Electronics > Headphones > In-Ear Headphones). If you're unsure of the category_id, still return the best guess for category.
    """
    messages = [
        {"role": "system", "content": prompt},
    ]
    messages.append({
        "role": "user",
        "content": query
    })
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )

    result = response.choices[0].message.content
    index_open_brace = result.find('{')
    index_open_bracket = result.find('[')
    first_pos = min(p for p in [index_open_brace, index_open_bracket] if p != -1)

    index_close_brace = result.rfind('}')
    index_close_bracket = result.rfind(']')
    last_pos = max(p for p in [index_close_brace, index_close_bracket] if p != -1)

    result = result[first_pos:last_pos + 1]
    result = json.loads(result)

    return result

async def fetch_html(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.text()
            else:
                print(f"Failed to fetch {url} with status {response.status}")
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

#Search product from google shopping using SerpAPI
def search_product(query):
    info = extract_info(query)
    search_term = info['search_term']
    print("------------Extract Info for product---------")
    print(info)

    if info['color']:
        search_term += f" {info['color']}"
    if info['size']:
        search_term += f" size {info['size']}"
    print("----------------Search Term--------------")
    print(search_term)
    params = {
        "engine": "google_shopping",
        "q": search_term,
        "api_key": serpapi_key,
        "sorted": "newest"
    }

    search = GoogleSearch(params)
    print("-----------SEARCH for GOOGLE---------")
    print(search)
    results = search.get_dict()

    results = results.get("shopping_results", [])
    print("---------Search Shopping Results---------")
    print(results[1:5])
    min_price = to_float(info.get('min_price'))
    max_price = to_float(info.get('max_price'))

    # Use results_sorted for further processing
    products = [
        {
            "price": r.get("price"),
            "image": r.get("thumbnail"),
            "url": r.get("product_link"),
            "title": r.get("title"),
            "thumbnails":  r.get("thumbnails")
        }
        for r in results
        if isinstance(r.get("extracted_price", None), (int, float))
           and (
                   (min_price is None or r["extracted_price"] >= min_price) and
                   (max_price is None or r["extracted_price"] <= max_price)
           )
    ]

    return products, info['is_specific_model']

#main route that get the user wanted products
@openai_bp.route('/product', methods=['POST'])
def product():
    data = request.get_json()
    query = data.get('query')
    device_id = data.get("user_id")
    query_type = data.get("type")

    products, is_specific_model = search_product(query)
    product_type = "specific" if is_specific_model.lower() == "yes" else "general"
    device_info = Devices.query.filter_by(device_id = device_id).first()
    if query_type == 'search':
        new_history = ProductHistory(user_id = device_info.email, search = query, products = json.dumps(products), product_type = product_type, created_at = datetime.now(), updated_at = datetime.now())
        db.session.add(new_history)
        db.session.commit()

        return jsonify({'products': products, 'id': new_history.id, 'product_type': product_type})

    return jsonify({'products': products, 'id': 0, 'product_type': product_type})

# Test route
@openai_bp.route('/search-product', methods=['POST'])
def search_products():
    data = request.get_json()
    query = data.get('question')

    info = extract_info(query)
    search_term = info['search_term']
    print(info)
    if not search_term:
        search_term = info['category_id']
    if info.get('color'):
        search_term += f""", color is {info.get('color')}"""
    if info.get('size'):
        search_term += f""", size is {info.get('size')}"""
    search_params = {
        "api_key": rainforest_api_key,
        "type": "search",
        "amazon_domain": "amazon.com",
        "search_term": search_term,
        "category_id": info['category_id'],
        "sort_by": "featured"
    }

    response = requests.get('https://api.rainforestapi.com/request', params=search_params)
    results = response.json()
    results = results.get('search_results', [])

    min_price = to_float(info.get('min_price'))
    max_price = to_float(info.get('max_price'))

    products = [
        {
            "image": r.get("image", ""),
            "price": r.get("price", {}).get("raw", ""),
            "url": r.get("link", ""),
            "value": r.get("price", {}).get("value", "")
        }
        for r in results
        if isinstance(r.get("price", {}).get("value", None), (int, float))
           and (
                   (min_price is None or r["price"]["value"] >= min_price) and
                   (max_price is None or r["price"]["value"] <= max_price)
           )
    ]

    if info['is_specific_model'].lower() == "yes":
        products = products[:7]

    return products, info['is_specific_model']

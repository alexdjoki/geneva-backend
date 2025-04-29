from flask import Blueprint, request, jsonify
from models.chat_history import ChatHistory
from app import db
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from together import Together
import anthropic
import google.generativeai as genai
import os
import requests
import asyncio
import json

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
deepseek_client = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")
grok_client = OpenAI(api_key=os.getenv('GROK_API_KEY'), base_url="https://api.x.ai/v1")
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
bing_api_key = os.getenv('BING_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cx_id = os.getenv('GOOGLE_CX_ID')
together_client = Together()

openai_bp = Blueprint('openai', __name__)

# GPT-4o (OpenAI)
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
        model="gpt-4-turbo",
        messages=messages
    ).choices[0].message.content

async def get_gpt4o_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(gpt4o_generate, history, prompt, question)
        return {"model": "GPT-4", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "GPT-4", "answer": str(e), "status": "failed"}

# Claude (Anthropic)
def claude_generate(history, prompt, question):
    messages = []
    for element in history:
        if element['type'] == 'question':
            messages.append({"role": "user", "content": element["text"]})
        if element['type'] == 'answer':
            messages.append({"role": "assistant", "content": element["text"]})
    
    messages.append({
        "role": "user",
        "content": question
    })
    return anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=messages,
        system=prompt
    ).content[0].text

async def get_claude_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(claude_generate, history, prompt, question)
        return {"model": "Claude", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Claude", "answer": str(e), "status": "failed"}

# Gemini
def gemini_generate(history, prompt, question):
    # prompt = flatten_messages(messages)
    messages = []
    for element in history:
        if element['type'] == 'question':
            messages.append({"role": "user", "parts": [element["text"]]})
        if element['type'] == 'answer':
            messages.append({"role": "model", "parts": [element["text"]]})

    model = genai.GenerativeModel("gemini-1.5-flash-001")
    chat = model.start_chat(history=messages)
    response = chat.send_message(question)
    return response.text

async def get_gemini_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(gemini_generate, history, prompt, question)
        return {"model": "Gemini", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Gemini", "answer": str(e), "status": "failed"}

def deepseek_generate(history, prompt, question):
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

async def get_deepseek_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(deepseek_generate, history, prompt, question)
        return {"model": "DeepSeek", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "DeepSeek", "answer": str(e), "status": "failed"}

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
        model="grok-3-latest",
        messages=messages
    ).choices[0].message.content

async def get_grok_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(grok_generate, history, prompt, question)
        return {"model": "Grok", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Grok", "answer": str(e), "status": "failed"}

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

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return response.text

async def get_mistral_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(mistral_generate, history, prompt, question)
        return {"model": "Mistral", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Mistral", "answer": str(e), "status": "failed"}
        
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

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return response.text

async def get_mistral_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(mistral_generate, history, prompt, question)
        return {"model": "Mistral", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Mistral", "answer": str(e), "status": "failed"}

def llama_generate(history, prompt, question):
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

async def get_llama_answer(history, prompt, question):
    try:
        answer = await asyncio.to_thread(llama_generate, history, prompt, question)
        return {"model": "Llama 4", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Llama 4", "answer": str(e), "status": "failed"}

def summarize_opinion(responses):
    successful_answers = [res for res in responses if res["status"] == "success"]
    if not successful_answers:
        return "All models failed to answer."

    summary_prompt = "Given the answers from different AI models, summarize the key points of agreement or notable differences among them in a concise paragraph."

    content = summary_prompt + "\n\n"
    for res in successful_answers:
        content += f"{res['model'].capitalize()} said: {res['answer']}\n\n"

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who compares AI answers and summarizes consensus or differences."},
            {"role": "user", "content": content}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

async def get_opinion(responses):
    try:
        answer = await asyncio.to_thread(summarize_opinion, responses)
        return answer
    except Exception as e:
        raise RuntimeError(f"Summarizing failed: {str(e)}")

def pick_best_answer(responses):
    # Here you can use heuristics or even send all to GPT for ranking
    valid_answers = [res for res in responses if res["status"] == "success"]
    if len(valid_answers) < 2:
        return valid_answers[0]["answer"] if valid_answers else "No valid answers."

    prompt = "You're given answers from three AI models to the same question. Evaluate which answer is the most helpful, accurate, and complete. Then output only the best answer."

    content = prompt + "\n\n"
    for i, res in enumerate(valid_answers):
        content += f"{res['model'].capitalize()} Answer:\n{res['answer']}\n\n"

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a critical evaluator of AI-generated responses."},
            {"role": "user", "content": content}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

async def get_best_answer(responses):
    try:
        answer = await asyncio.to_thread(pick_best_answer, responses)
        return answer
    except Exception as e:
        raise RuntimeError(f"Picking best answer failed: {str(e)}")

def generate_answers(level, history, prompt, question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
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
    
    return results

def anaylze_result(results):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    summarize = loop.run_until_complete(asyncio.gather(
        get_best_answer(results),
        get_opinion(results)
    ))
    best_answer, opinion = summarize
    loop.close()

    return best_answer, opinion

def get_answer(level, history, prompt, question):
    results = generate_answers(level, history, prompt, question)
    status_report = [
        {key: value for key, value in result.items() if key != 'answer'}
        for result in results
    ]
    best_answer, opinion = anaylze_result(results)

    return {
        "final_answer": best_answer,
        "status_report": status_report,
        "opinion": opinion
    }

def fetch_bing_news(query):
    url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "count": 10}
    response = requests.get(url, headers=headers, params=params)
    results = response.json().get('value', [])
    print('bing')
    print(results)
    return [{
        "title": r.get("name", ""),
        "snippet": r.get("description", ""),
        "source": r.get("provider", [{}])[0].get("name", ""),
        "url": r.get("url", "")
    } for r in results]

def fetch_google_pse_news(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_api_key,
        "cx": google_cx_id,
        "q": f"site:reuters.com OR site:bloomberg.com {query}",
        "num": 10
    }
    response = requests.get(url, params=params)
    results = response.json().get('items', [])
    print('google_pse')
    print(results)
    return [{
        "title": r.get("title", ""),
        "snippet": r.get("snippet", ""),
        "source": r.get("displayLink", ""),
        "url": r.get("link", "")
    } for r in results]

def prioritize_sources(articles):
    trusted_sources = ["Reuters", "Bloomberg", "WSJ", "CNBC"]
    prioritized = [a for a in articles if any(ts.lower() in a['source'].lower() for ts in trusted_sources)]
    others = [a for a in articles if a not in prioritized]
    return prioritized + others

def compress_articles(articles, max_tokens=700):
    combined_text = ""
    for article in articles:
        combined_text += f"{article['source']}: {article['snippet']} "
    # Dummy compression (replace with real summarizer later)
    compressed = combined_text[:4000]  # Rough cut, to replace with smart compression
    return compressed

def retrieve_news(query):
    # Fetch from two APIs
    bing_articles = fetch_bing_news(query)
    google_articles = fetch_google_pse_news(query)

    # Merge + deduplicate
    all_articles = bing_articles + google_articles
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
        "sources_used": list({a['source'] for a in sorted_articles[:2]}),
        "compressed_summary": compressed_summary
    }

def retrieve():
    data = request.get_json()
    question = data.get("question")
    return fetch_google_pse_news(question)

def judge_system(question):
    expected_output = """
        {
            "level": "Easy"
            "last_year": "Yes"
        }
    """
    judge_prompt = f"""
    Classify the difficulty of the following question as Easy, Medium, or Complex. And also check that user want data post-2025 or pre-2025.
    I want output like this format: {expected_output}
    Question: {question}"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",  # or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates the difficulty of questions."},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=1
    )
    judge_output = response.choices[0].message.content
    judge_output = json.loads(judge_output)
    return judge_output

@openai_bp.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    chat_id = data.get("chat_id")
    user_id = data.get("user_id")
    history = data.get("history")
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question is required"}), 400

    prompt = f"""
    You are a knowledgeable AI assistant. Your task is to generate a clear, accurate, and helpful answer based solely on your understanding of the topic.
    Please carefully read the question below and provide a detailed response using natural language.
    """
    expected_output = """
        {
            'level': 'Easy'
            'last_year': 'Yes'
        }
    """
    judge_prompt = f"""
    Classify the difficulty of the following question as Easy, Medium, or Complex. And also check that user want data post-2025 or pre-2025.
    I want output like this format:
    Output: {expected_output}
    Question: {question}
    """
    try:
        judge_output = judge_system(question)
        response = get_answer(judge_output['level'], history, prompt, question)
        new_history = ChatHistory(user_id = user_id, answer = response["final_answer"], status_report = json.dumps(response["status_report"]), opinion = response["opinion"], chat_id = chat_id, question = question, level = judge_output['level'], created_at = datetime.now(), updated_at = datetime.now())
        db.session.add(new_history)
        db.session.commit()
        return jsonify({"question": question, "answer": response["final_answer"], "level": judge_output['level'], "status_report": response["status_report"], "opinion": response["opinion"]})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
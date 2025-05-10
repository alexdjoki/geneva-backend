from flask import Blueprint, request, jsonify
from models.chat_history import ChatHistory
from models.product_history import ProductHistory
from app import db
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from together import Together
from urllib.parse import urlparse
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
google_search_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
google_cx_id = os.getenv('GOOGLE_CX_ID')
brave_api_key = os.getenv('BRAVE_API_KEY')
rainforest_api_key = os.getenv('RAINFOREST_API_KEY')
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
    print('start gpt')
    try:
        answer = await asyncio.to_thread(gpt4o_generate, history, prompt, question)
        print('end gpt')
        return {"model": "GPT-4.1", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "GPT-4.1", "answer": str(e), "status": "failed"}

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

    full_response = ""
    stop_reason = None

    while True:
        response = anthropic_client.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=1024,
            messages=messages
        )

        chunk_text = response.content[0].text
        stop_reason = response.stop_reason

        full_response += chunk_text

        if stop_reason != "max_tokens":
            break

        # Append current response to context and ask it to continue
        messages.append({"role": "assistant", "content": chunk_text})
        messages.append({"role": "user", "content": "Please continue."})

    return full_response

async def get_claude_answer(history, prompt, question):
    print('start claude')
    try:
        answer = await asyncio.to_thread(claude_generate, history, prompt, question)
        print('end claude')
        return {"model": "Claude Sonnet 3.7", "answer": answer, "status": "success"}
    except Exception as e:
        print(str(e))
        return {"model": "Claude Sonnet 3.7", "answer": str(e), "status": "failed"}

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
    print('start gemini')
    try:
        answer = await asyncio.to_thread(gemini_generate, history, prompt, question)
        print('end gemini')
        return {"model": "Gemini 2.5 Pro", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Gemini 2.5 Pro", "answer": str(e), "status": "failed"}

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
    print('start deepseek')
    try:
        answer = await asyncio.to_thread(deepseek_generate, history, prompt, question)
        print('end deepseek')
        return {"model": "DeepSeek V3", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "DeepSeek V3", "answer": str(e), "status": "failed"}

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
    print('start grok')
    try:
        answer = await asyncio.to_thread(grok_generate, history, prompt, question)
        print('end grok')
        return {"model": "Grok 3 Fast", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Grok 3 Fast", "answer": str(e), "status": "failed"}

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
    print('start mistral')
    try:
        answer = await asyncio.to_thread(mistral_generate, history, prompt, question)
        print('end mistral')
        return {"model": "Mistral Large", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Mistral Large", "answer": str(e), "status": "failed"}

def llama_generate(history, prompt, question):
    response = together_client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

async def get_llama_answer(history, prompt, question):
    print('start llama')
    try:
        answer = await asyncio.to_thread(llama_generate, history, prompt, question)
        print('end llama')
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

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who compares AI answers and summarizes consensus or differences."},
            {"role": "user", "content": content}
        ],
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

    prompt = f"""You're given answers from AI models to the same question. Analyze the answers then give only the best answer without any explain(info) of answer."""

    content = prompt + "\n\n"
    for i, res in enumerate(valid_answers):
        content += f"Answer:\n{res['answer']}\n\n"

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a critical evaluator of AI-generated responses."},
            {"role": "user", "content": content}
        ],
    )

    return "Consensus:\n" + response.choices[0].message.content.strip()

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

def fetch_brave_news(query):
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": brave_api_key
    }

    params = {
        "q": query,
        "count": 10  # number of results (max: 20 for free tier)
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

def fetch_bing_news(query):
    url = "https://api.bing.microsoft.com/v7.0/news/search"
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": query, "count": 10}
    response = requests.get(url, headers=headers, params=params)
    results = response.json().get('value', [])
    return [{
        "title": r.get("name", ""),
        "snippet": r.get("description", ""),
        "source": r.get("provider", [{}])[0].get("name", ""),
        "url": r.get("url", "")
    } for r in results]

def fetch_google_pse_news(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": google_search_api_key,
        "cx": google_cx_id,
        "q": f"site:reuters.com OR site:bloomberg.com OR site:wsj.com OR site:cnbc.com {query}",
        "num": 10
    }
    response = requests.get(url, params=params)
    results = response.json().get('items', [])
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
    brave_articles = fetch_brave_news(query)
    google_articles = fetch_google_pse_news(query)

    # Merge + deduplicate
    all_articles = google_articles + brave_articles
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

def generate_news(level, history, prompt, question):
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

def get_news(level, query):
    result = retrieve_news(query)
    print('start analyzing news')
    prompt = "You are a helpful research assistant that summarizes news search results."
    question = f"""
    Summarize the following search results into a brief report. Highlight the most relevant and recent information for the topic from the Search Results: "{query}"

    Search Results:
    {result['compressed_summary']}
    """

    results = generate_news(level, [], prompt, question)
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
    
@openai_bp.route('/retrieve', methods=['POST'])
def retrieve():
    data = request.get_json()
    question = data.get("question")
    return retrieve_news(question)

def judge_system(question):
    expected_output = """
        {
            "level": "Easy"
            "last_year": "Yes"
            "product": "Men's shoes"
        }
    """
    judge_prompt = f"""
    Classify the difficulty of the following question as Easy, Medium, or Complex. And determine if the user is asking for information about events or data from this year (i.e., {datetime.now().year}). Respond with "Yes" or "No".
    And also check if user is looking for product.
    I want output like this format: {expected_output}"""

    messages = [
        {"role": "system", "content": judge_prompt},
    ]
    messages.append({
        "role": "user",
        "content": question
    })

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
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
    return judge_output

@openai_bp.route('/judge', methods=['POST'])

def judge():
    data = request.get_json()
    question = data.get('question')
    return judge_system(question)

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
    try:
        judge_output = judge_system(question)
        result = {}
        products = []
        if judge_output['product'] != 'No':
            products = search_product(question)
            judge_output['level'] = 'product'
            result = {
                "final_answer": json.dumps(products),
                "status_report": [],
                "opinion": '',
            }
        elif judge_output['last_year'] == 'Yes':
            result = get_news(judge_output['level'], question)
        else:
            result = get_answer(judge_output['level'], history, prompt, question)
        new_history = ChatHistory(user_id = user_id, answer = result["final_answer"], status_report = json.dumps(result["status_report"]), opinion = result["opinion"], chat_id = chat_id, question = question, level = judge_output['level'], created_at = datetime.now(), updated_at = datetime.now())
        db.session.add(new_history)
        db.session.commit()
        return jsonify({"question": question, "answer": result["final_answer"], "level": judge_output['level'], "status_report": result["status_report"], "opinion": result["opinion"]})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

def extract_info(query):
    prompt = f"""
    Extract the following information from the user's query:

    - category_id (must need)
    - search_term (general item or keyword being looked for. must need)
    - color
    - size
    - min_price
    - max_price

    Return it as JSON.
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

def search_product(query):
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
        "sort_by": "price_low_to_high"
    }
    
    if info.get('min_price') is not None :
        search_params["min_price"] = info['min_price']
    if info.get('max_price') is not None :
        search_params["max_price"] = info['max_price']

    response = requests.get('https://api.rainforestapi.com/request', params=search_params)
    results = response.json()
    results = results.get('search_results', [])

    products = [{
        "image": r.get("image", ""),
        "price": r.get("price", {}).get("raw", ""),
        "url": r.get("link", "")
    } for r in results]

    return products

@openai_bp.route('/product', methods=['POST'])

def product():
    data = request.get_json()
    query = data.get('query')
    user_id = data.get("user_id")
    query_type = data.get("type")
    
    products = search_product(query)

    if query_type == 'search':
        new_history = ProductHistory(user_id = user_id, search = query, products = json.dumps(products), created_at = datetime.now(), updated_at = datetime.now())
        db.session.add(new_history)
        db.session.commit()

        return jsonify({'products': products, 'id': new_history.id})
    
    return jsonify({'products': products, 'id': 0})
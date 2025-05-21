from flask import Blueprint, request, jsonify
from models.chat_history import ChatHistory
from models.product_history import ProductHistory
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

    model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")
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
        "model": "mistral-large-latest",
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

    summary_prompt = """Given the answers from different AI models, provide a structured comparison including:

    Key points of agreement

    Notable differences

    Any unique insights provided by individual models
    
    Present the output in a clear and organized format using bullet points or numbered sections without any tables."""

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

def analyze_result(results):
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
    best_answer, opinion = analyze_result(results)

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
    best_answer, opinion = analyze_result(results)

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
            "product": ["Product A", "Product B"] or []
        }
    """
    judge_prompt = f"""
    You are an AI assistant responsible for classifying user queries.

    1. **Difficulty Assessment**: Categorize the complexity of the question as one of the following: "Easy", "Medium", or "Complex".
    
    2. Determine if the user's query requires information newer than your knowledge cutoff date. Use these rules:

        1) Respond ONLY with "No" if:
            - The question is about facts existing before or during your knowledge period
            - No time-sensitive information is requested
            - The subject matter is historical or static

        2) Respond ONLY with "Yes" if:
            - The question asks about events/developments after your cutoff
            - Uses terms like "current", "latest", "recent", "now", "this year"
            - Concerns rapidly-changing topics (tech, medicine, news)

        3) For ambiguous cases where time isn't specified but the topic evolves:
            - Default to "Yes" for safety

        Never explain your reasoning - only output "Yes" or "No".

    3. **Product Intent Detection**:  

    Determine whether the user is inquiring about a **buyable consumer product** such as clothing, electronics, tools, or household goods.  
    - If yes, extract the product names mentioned in the query and return them in a list.  
    - If no products are found, return an empty list `[]`.

    Return your answer strictly in the following JSON format: {expected_output}
    """

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

    If the question involves comparing multiple products, technologies, or concepts, make sure to:
    - Identify all items being compared.
    - Explain the key features, strengths, and weaknesses of each.
    - Highlight important differences and when one might be preferred over another.
    - Use bullet points or structured formatting if it improves clarity.

    Be objective and informative.
    """
    try:
        judge_output = judge_system(question)
        result = {}
        products = []
        print(judge_output)
        if len(judge_output['product']) > 0:
            judge_output['level'], result = analyze_product(judge_output['level'], history, prompt, question)
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

async def generate_answer(level, history, prompt, query):
    try:
        answer = await asyncio.to_thread(get_answer, level, history, prompt, query)
        return answer
    except Exception as e:
        raise RuntimeError(f"Generating answer failed: {str(e)}")

async def get_product(query):
    try:
        answer, is_specific_model = await asyncio.to_thread(search_product, query)
        return answer
    except Exception as e:
        raise RuntimeError(f"Searching product failed: {str(e)}")

def compare_product(level, history, prompt, query, products):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    product_tasks = [get_product(product) for product in products]

    # Run generate_answer and all product fetches concurrently
    analyze = loop.run_until_complete(asyncio.gather(
        generate_answer(level, history, prompt, query),
        *product_tasks
    ))
    result = analyze[0]
    product_results = analyze[1:]
    loop.close()

    # Flatten product_results (each is a list)
    product_data = [item[0] for item in product_results]

    # Set correct titles
    for i, product in enumerate(product_data):
        product["title"] = products[i]

    answer = result["final_answer"]
    result["final_answer"] = json.dumps({
        "answer": answer,
        "products": product_data
    })

    return result

def analyze_product(level, history, prompt, query):
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
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": analyze_prompt}
        ],
        temperature=1
    )

    
    response = response.choices[0].message.content
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

    if response['just_looking'] == 'Yes':
        products, is_specific_model = search_product(query)
        result = {
            "final_answer": json.dumps(products),
            "status_report": [],
            "opinion": '',
        }
        return "specific_product" if is_specific_model.lower() == "yes" else "general_product", result
    else:
        result = {}
        if response['intent'] == 'compare':
            prompt = compare_prompt
        result = compare_product(level, history, prompt, query, response['products'])
        return "compare_product", result

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

def search_product(query):
    info = extract_info(query)
    search_term = info['search_term']
    print(info)

    params = {
        "engine": "google_shopping",
        "q": search_term,
        "api_key": serpapi_key,
    }
    
    print(params)
    search = GoogleSearch(params)
    results = search.get_dict()

    results = results.get("shopping_results", [])

    products = [
        {
            "price": r.get("price"),
            "image": r.get("thumbnail"),
            "url": r.get("product_link")
        }
        for r in results
        if isinstance(r.get("extracted_price", None), (int, float))
    ]
    return products, info['is_specific_model']

@openai_bp.route('/product', methods=['POST'])

def product():
    data = request.get_json()
    query = data.get('query')
    user_id = data.get("user_id")
    query_type = data.get("type")
    
    products, is_specific_model = search_product(query)
    product_type = "specific" if is_specific_model.lower() == "yes" else "general"

    if query_type == 'search':
        new_history = ProductHistory(user_id = user_id, search = query, products = json.dumps(products), product_type = product_type, created_at = datetime.now(), updated_at = datetime.now())
        db.session.add(new_history)
        db.session.commit()

        return jsonify({'products': products, 'id': new_history.id, 'product_type': product_type})
    
    return jsonify({'products': products, 'id': 0, 'product_type': product_type})

@openai_bp.route('/search-product', methods=['POST'])

def search_products():
    data = request.get_json()
    query = data.get('question')
    
    info = extract_info(query)
    search_term = info['search_term']
    print(info)

    params = {
        "engine": "google_shopping",
        "q": search_term,
        "api_key": serpapi_key
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()

    results = results.get("shopping_results", [])

    products = [
        {
            "price": r.get("price"),
            "image": r.get("thumbnail"),
            "url": r.get("product_link")
        }
        for r in results
        if isinstance(r.get("extracted_price", None), (int, float))
    ]
    return results
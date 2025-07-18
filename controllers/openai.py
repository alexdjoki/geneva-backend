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
    print('start gpt')
    try:
        answer = await asyncio.to_thread(gpt4o_generate, history, prompt, question)
        print('end gpt')
        print(answer)
        return {"model": "GPT-4.1", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "GPT-4.1", "answer": "", "status": "failed"}

# Generate answer using Claude (Anthropic)
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
            model='claude-opus-4-20250514',
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
        print(answer)
        return {"model": "Claude Opus 4", "answer": answer, "status": "success"}
    except Exception as e:
        print(str(e))
        return {"model": "Claude Opus 4", "answer": "", "status": "failed"}

# Generate answer using Gemini
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
    print('start deepseek')
    try:
        answer = await asyncio.to_thread(deepseek_generate, history, prompt, question)
        print('end deepseek')
        print(answer)
        return {"model": "DeepSeek 4", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "DeepSeek 4", "answer": "", "status": "failed"}
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
    print('start grok')
    try:
        answer = await asyncio.to_thread(grok_generate, history, prompt, question)
        print('end grok')
        print(answer)
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
    print('start llama')
    try:
        answer = await asyncio.to_thread(llama_generate, history, prompt, question)
        print('end llama')
        print(answer)
        return {"model": "Llama 4", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Llama 4", "answer": str(e), "status": "failed"}

#Summarize the opinion from answers of each models
def summarize_opinion(responses):
    successful_answers = [res for res in responses if res["status"] == "success"]
    if not successful_answers:
        return "All models failed to answer."

    summary_prompt = """Given the answers from different AI models, provide a structured comparison including:

    Key points of agreement

    Notable differences

    Any unique insights provided by individual models
    
    Present the output in a clear and organized format using bullet points or numbered sections without any tables. The answer can't be empty.
    
    And You have to include the model name and version I provided when you answer .
    """
    
    content = summary_prompt + "\n\n"
    for res in successful_answers:
        if res['answer'] != "" :
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

#Pick besk answer from answers of AI mdoels
def pick_best_answer(responses):
    # Here you can use heuristics or even send all to GPT for ranking
    valid_answers = [res for res in responses if res["status"] == "success"]
    if len(valid_answers) < 2:
        return valid_answers[0]["answer"] if valid_answers else "No valid answers."

    prompt = f"""
    Multiple answers are provided for the same question, generated by different AI models.

    Choose the answer that the user understands and is based on accurate explanation and evidence, detailed.

    Please answer only with the selected answer. Do not include explanation, evidence, or additional explanation.
    """

    content = prompt + "\n\n"
    for i, res in enumerate(valid_answers):
        content += f"Answer:\n{res['answer']}\n\n"

    print("---------------PICK BEST ANSER---------------")
    print(content)

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a critical evaluator of AI-generated responses."},
            {"role": "user", "content": content}
        ],
    )

    return "**Consensus:**\n" + response.choices[0].message.content.strip()

async def get_best_answer(responses):
    try:
        answer = await asyncio.to_thread(pick_best_answer, responses)
        return answer
    except Exception as e:
        raise RuntimeError(f"Picking best answer failed: {str(e)}")
#Generate answer of each models using gathering 3/5/7
def generate_answers(level, history, prompt, question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    print("-------------Prompt------------")
    print(prompt)
    print("-------------Question-------------")
    print(question)
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

#Pick best answer and summarize the opinion from answers of each models
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

#Generate answer of user asked(findal_answer: main answer, status_report: AI models success report, opinion: summarized opinion)
def get_answer(level, history, prompt, question):
    results = generate_answers(level, history, prompt, question)
    status_report = [
        {key: value for key, value in result.items() if key != 'answer'}
        for result in results
    ]
    best_answer, opinion = analyze_result(results)
    final_answer_with_images = insert_images(best_answer, question)

    return {
        "final_answer": final_answer_with_images,
        "status_report": status_report,
        "opinion": opinion
    }

#Get news using brave API
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

#Get news from google search
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
    print("All Articles")
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

#Get the news using google search and brave API. After that Analyze the searched results using AI models
def get_news(level, query):
    result = retrieve_news(query)
    print("Result for retrieved news")
    print(result)
    print('start analyzing news')
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

    results = generate_news(level, [], prompt, question)
    print("News for Generated from news")
    print(results)
    status_report = [
        {key: value for key, value in result.items() if key != 'answer'}
        for result in results
    ]
    best_answer, opinion = analyze_result(results)
    final_answer_with_images = insert_images(best_answer, query)
    return {
        "final_answer": final_answer_with_images,
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
def judge_system(question, history = []):
    history_text = ""
    for item in history:
        text = item.get('text', '')  # This won't raise KeyError
        prefix = "User" if item["type"] == "question" else "Assistant"
        history_text += f"{prefix}: {text}\n"

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

    3. **used_in_context**: Determine if the current question depends on or references the history. Say `true` if it’s a follow-up or refers to anything previously discussed. Say `false` if it’s a completely new topic.

    4. **updated_question**:
        - If `used_in_context` is `true`, rewrite the question so that it is fully self-contained, including the necessary context from the conversation history.
        - If `used_in_context` is `false`, return an empty string.

    5. **Product Intent Detection**:  

    Determine whether the user is inquiring about a **buyable consumer product** such as clothing, electronics, tools, or household goods.  
    - If yes, extract the product names mentioned in the query and return them in a list.  
    - If no products are found, return an empty list `[]`.

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
    return judge_output

# Test route
@openai_bp.route('/judge', methods=['POST'])

def judge():
    data = request.get_json()
    question = data.get('question')
    return judge_system(question)

def get_contextual_image(paragraph, search_query):
    """Fetches one highly relevant image for the specific paragraph"""
    try:
        params = {
            "engine": "google_images",
            "q": search_query,
            "api_key": serpapi_key,
            "num": 1,  # Only need one best match
            "safe": "active"
        }
        result = GoogleSearch(params).get_dict()
        images = result.get("images_results", [])
        
        if images:
            return {
                "thumbnail": images[0].get("thumbnail"),
                "title": images[0].get("title", search_query)
            }
    except Exception as e:
        print(f"Image search error for '{search_query}': {e}")
    return None

def extract_key_terms(text):
    """Improved keyword extraction focusing on nouns and named entities"""
    words = re.findall(r'\w+', text.lower())
    stopwords = {'the','a','an','and','or','but','is','are','was','were','for','on','in','of','to','with'}
    
    # Filter stopwords and short words
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Simple frequency analysis (can be enhanced with NLP)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top 3 most frequent meaningful words
    return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:3]
def find_best_image_match(paragraph, images, used_indices):
    """
    Finds the most relevant image for the given paragraph using semantic matching
    """
    if not images:
        return None
        
    # First try to find an exact match in image titles
    paragraph_lower = paragraph.lower()
    for i, img in enumerate(images):
        if i not in used_indices and img['title'].lower() in paragraph_lower:
            return i
    
    # Then try keyword matching
    keywords = extract_keywords(paragraph)
    best_score = 0
    best_idx = None
    
    for i, img in enumerate(images):
        if i in used_indices:
            continue
            
        img_keywords = extract_keywords(img['title'])
        score = len(set(keywords) & set(img_keywords))
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    # Fallback: use the first unused image if no good match found
    if best_idx is None:
        for i in range(len(images)):
            if i not in used_indices:
                return i
                
    return best_idx
def insert_images(answer_text, query):
    """
    Insert relevant images into the answer text with dynamic distribution:
    - 1 image for very short answers (<150 words)
    - 2-3 images for medium answers (150-400 words)
    - 4-7 images for long articles (>400 words)
    - Always includes a top image
    """
    paragraphs = [p.strip() for p in re.split(r'(?<=\n\n)(?=\S)', answer_text) if p.strip()]
    if not paragraphs:
        return answer_text

    # Calculate word count and determine image count
    word_count = sum(len(re.findall(r'\w+', p)) for p in paragraphs)
    
    if word_count < 150:
        num_images = 1
    elif word_count < 400:
        num_images = random.randint(2, 3)  # 2-3 images for medium content
    else:
        # 4-7 images for long content, scaling with length
        num_images = min(7, max(4, word_count // 150))
    
    # Ensure we don't have more images than paragraphs
    num_images = min(num_images, len(paragraphs) + 1)  # +1 for top image
    
    # Generate positions (always include top position 0)
    positions = [0]  # Top image position
    
    # Calculate remaining positions for even distribution
    if num_images > 1:
        step = max(1, len(paragraphs) // (num_images - 1))
        positions.extend(min(i * step, len(paragraphs) - 1) for i in range(1, num_images))
    
    # Remove duplicates and sort
    positions = sorted(list(set(positions)))
    
    # Prepare diverse image queries
    base_queries = [
        f"{query}",
        f"{query} product",
        f"{query} details",
        f"{query} example",
        f"{query} photo",
        f"{query} illustration",
        f"{query} diagram"
    ]
    
    # Generate context-specific queries for remaining positions
    image_queries = base_queries[:num_images]
    for i in range(len(base_queries), num_images):
        if i < len(paragraphs):
            keywords = extract_key_terms(paragraphs[positions[i]])[:2]
            image_queries.append(f"{query} {' '.join(keywords)}")
        else:
            image_queries.append(random.choice(base_queries))

    # Fetch images concurrently
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        tasks = [fetch_image(q) for q in image_queries[:num_images]]
        images = loop.run_until_complete(asyncio.gather(*tasks))
    finally:
        loop.close()

    # Build the final content with images
    results = []
    img_idx = 0
    
    # Insert top image if available
    if images and img_idx < len(images) and images[img_idx]:
        results.append(format_image(images[img_idx]))
        img_idx += 1
    
    # Insert content with remaining images
    for i, para in enumerate(paragraphs):
        results.append(para)
        
        # Insert image at designated positions
        if i+1 in positions[1:] and img_idx < len(images) and images[img_idx]:
            results.append(format_image(images[img_idx]))
            img_idx += 1

    return "\n\n".join(results)

def format_image(img_data):
    """Format image HTML with responsive classes"""
    return f"""
    <div class="image-container-field">
        <img src="{img_data['thumbnail']}" 
             alt="{img_data['title']}" 
             class="rounded-shadow-image responsive-img">
        <div class="image-caption">{img_data['title']}</div>
    </div>
    """

async def fetch_image(search_query):
    """Fetch image with timeout and error handling"""
    def _fetch():
        try:
            params = {
                "engine": "google_images",
                "q": search_query,
                "api_key": serpapi_key,
                "num": 1,
                "safe": "active",
                "hl": "en"
            }
            response = requests.get('https://serpapi.com/search', 
                                 params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("images_results"):
                    return {
                        "thumbnail": data["images_results"][0].get("thumbnail"),
                        "title": data["images_results"][0].get("title", search_query)
                    }
        except Exception as e:
            print(f"Image fetch error for '{search_query}': {str(e)}")
        return None

    return await asyncio.get_event_loop().run_in_executor(None, _fetch)

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

    # valid = check_valid(user_id)
    #
    # if valid == False:
    #     return jsonify({"level": "subscribe"})
    # if not question:
    #     return jsonify({"error": "Question is required"}), 400

    # return jsonify({"level": "subscribe"})
    prompt = f"""

    You are a knowledgeable and objective AI assistant. Your task is to generate a clear, accurate, and helpful answer based solely on your understanding of the topic.
    Please carefully read the question below and provide a detailed, well-structured response in natural language.
    If the question involves comparing multiple products, technologies, or concepts, ensure that you:
    - Identify all items being compared.
    - Explain the key features, advantages, and limitations of each.
    - Highlight meaningful differences and suggest when one may be more suitable than another.
    - Use bullet points, tables, or clear formatting to improve readability.
    - If there are products, you have to provide the details about the best products in answer.
    - Recommend Products: '''---PRODUCTS---''' If there is the data in Recommend Products, search the recommend the products, and give and explain as Rank 1 for each product.
    
    Avoid including disclaimers such as "as of my last update." Focus on delivering useful, confident information without referencing time limitations.
    And about answers and news, you have not to mention source that you got the new data.
    You have to provide accurate and detailed answers while satisfying the conditions above.
    """

    print(prompt)
    judge_output = judge_system(question, history)
    if judge_output['used_in_context'] == False:
        history = []
    else:
        question = judge_output['updated_question']
    print("Judge_output")
    print(judge_output)

    if len(judge_output['product']) > 0:
        judge_output['level'], result = analyze_product(judge_output['level'], judge_output['last_year'], history, prompt, question)
    elif judge_output['last_year'] == 'Yes':
        result = get_news(judge_output['level'], question)
    else:
        result = get_answer(judge_output['level'], history, prompt, question)
    token = Devices.query.filter_by(device_id=user_id).first()
    if token:
        user_id = token.email
    new_history = ChatHistory(user_id = user_id, answer = result["final_answer"], status_report = json.dumps(result["status_report"]), opinion = result["opinion"], chat_id = chat_id, question = question, level = judge_output['level'], created_at = datetime.now(), updated_at = datetime.now())
    db.session.add(new_history)
    db.session.commit()
    return jsonify({"question": question, "answer": result["final_answer"], "level": judge_output['level'], "status_report": result["status_report"], "opinion": result["opinion"]})



async def generate_answer(level, history, prompt, query):
    try:
        answer = await asyncio.to_thread(get_answer, level, history, prompt, query)
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

async def search_news(level, query):
    try:
        answer = await asyncio.to_thread(get_news, level, query)
        return answer
    except Exception as e:
        raise RuntimeError(f"Searching news failed: {str(e)}")

async def get_product(query):
    try:
        answer, is_specific_model = await asyncio.to_thread(search_product, query)
        return answer
    except Exception as e:
        raise RuntimeError(f"Searching product failed: {str(e)}")
#Generate answer of user asked question and search products mentioned in user asked question.
#If last_year: Yes, get news and analyze news with AI models 3/5/7. After that combine answer and searched products.
#IF last_year: No, generate answer with AI models 3/5/7. After that combine answer and searched products.
def compare_product(level, last_year, history, prompt, query, products):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    product_tasks = [get_product(product) for product in products]

    product_results = loop.run_until_complete(asyncio.gather(*product_tasks))

    print("--------Product tasks----------")
    # print(product_results)
    prompt = prompt.replace("---PRODUCTS---", json.dumps(product_results))
    print("----------Adjust Prompt-----------")
    print(prompt)
    # Run generate_answer and all product fetches concurrently
    if last_year.lower() == 'yes':
        analyze = loop.run_until_complete(asyncio.gather(
            search_news(level, query),
        ))
    else:
        analyze = loop.run_until_complete(asyncio.gather(
            generate_answer(level, history, prompt, query),
        ))
    result = analyze[0]
    loop.close()

    # Flatten product_results (each is a list)
    print("------------PRODUCT_RESULTS------------")
    # print(product_results)
    product_data = product_results[0]
    print("---------Fianl Product ---data")
    # print(product_data)
    # Set correct titles
    # for i, product in enumerate(product_data):
    #     product["title"] = products[i]

    answer = result["final_answer"]
    print("----------Products----------")
    # print(product_data)
    result["final_answer"] = json.dumps({
        "answer": answer,
        "products": product_data
    })

    return result

#This function will called if user asked question with products. Generate the answer of user asked question.
#If user just want only looking products search products and return.
#If user want product's analyzed data, compared data, review and etc, Combine analyzed data with AI answers and products list.

def analyze_product(level, last_year, history, prompt, query):
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
    result = compare_product(level, last_year, history, prompt, query, response['products'])
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
    print(results)
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

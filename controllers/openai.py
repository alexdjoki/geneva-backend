from flask import Blueprint, request, jsonify
from models.chat_history import ChatHistory
from app import db
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
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
        model="gpt-4o",
        messages=messages
    ).choices[0].message.content

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
        model="claude-3-haiku-20240307",
        max_tokens=300,
        messages=messages,
        system=prompt
    ).content[0].text

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
    print(chat)
    response = chat.send_message(question)
    return response.text

async def get_gpt4o_answer(history, prompt, question):
    try:
        answer = gpt4o_generate(history, prompt, question)
        return {"model": "GPT-4", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "GPT-4", "answer": str(e), "status": "failed"}

async def get_claude_answer(history, prompt, question):
    try:
        answer = claude_generate(history, prompt, question)
        return {"model": "Claude", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Claude", "answer": str(e), "status": "failed"}

async def get_gemini_answer(history, prompt, question):
    try:
        answer = gemini_generate(history, prompt, question)
        return {"model": "Gemini", "answer": answer, "status": "success"}
    except Exception as e:
        return {"model": "Gemini", "answer": str(e), "status": "failed"}

def summarize_opinion(responses):
    successful_answers = [res for res in responses if res["status"] == "success"]
    if not successful_answers:
        return "All models failed to answer."

    summary_prompt = "Given the answers from different AI models, summarize the key points of agreement or notable differences among them in a concise paragraph."

    content = summary_prompt + "\n\n"
    for res in successful_answers:
        content += f"{res['model'].capitalize()} said: {res['answer']}\n\n"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who compares AI answers and summarizes consensus or differences."},
            {"role": "user", "content": content}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

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
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a critical evaluator of AI-generated responses."},
            {"role": "user", "content": content}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

def generate_easy(history, prompt, question):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    tasks = [
        get_gpt4o_answer(history, prompt, question),
        get_claude_answer(history, prompt, question),
        get_gemini_answer(history, prompt, question),
    ]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()

    status_report = [
        {key: value for key, value in result.items() if key != 'answer'} 
        for result in results
    ]
    # Extract results
    best_answer = pick_best_answer(results)
    opinion = summarize_opinion(results)

    return {
        "final_answer": best_answer,
        "status_report": status_report,
        "opinion": opinion
    }

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

    judge_prompt = f"""
    Classify the difficulty of the following question as Easy, Medium, or Complex.
    I want only one word of (Easy, Medium, Complex)
    Question: {question}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4"
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates the difficulty of questions."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.1
        )

        level = response.choices[0].message.content
        response = {}
        if level == 'Easy':
            response = generate_easy(history, prompt, question)
        elif level == 'Medium':
            response = generate_easy(history, prompt, question)
        elif level == 'Complex':
            response = generate_easy(history, prompt, question)
        new_history = ChatHistory(user_id = user_id, answer = response["final_answer"], status_report = json.dumps(response["status_report"]), opinion = response["opinion"], chat_id = chat_id, question = question, level = level, created_at = datetime.now(), updated_at = datetime.now())
        db.session.add(new_history)
        db.session.commit()
        return jsonify({"question": question, "answer": response["final_answer"], "level": level, "status_report": response["status_report"], "opinion": response["opinion"]})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500
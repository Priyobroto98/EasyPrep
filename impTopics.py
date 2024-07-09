import os
from flask import Flask, render_template, session
from googleapiclient.discovery import build
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Google Generative AI Configuration
google_api_key = os.getenv("GOOGLE_API_KEY")
generation_config = {"temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 4000}

model = GoogleGenerativeAI(model="gemini-pro", generation_config=generation_config)

prompt_template = PromptTemplate(
    input_variables=['text'],
    template="you are an expert in most important topic extraction from a large text input . suggest top 10 most important topics from that text that can be useful for the user.\ntext:{text}\n"
)

chain = LLMChain(llm=model, prompt=prompt_template, verbose=True)

# YouTube API Configuration
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

def get_youtube_video_link(topic):
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)

    request = youtube.search().list(
        part='snippet',
        q=topic,
        type='video',
        maxResults=1
    )
    response = request.execute()

    video_id = response['items'][0]['id']['videoId']
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    return video_url

def vdo_id(results_text):
    topics = [line.strip() for line in results_text.strip().split('\n')]

    video_info = []
    for topic in topics:
        topic_name = ' '.join(topic.split('.')[1:]).strip()
        video_link = get_youtube_video_link(topic_name)
        video_info.append((topic_name, video_link))

    return video_info

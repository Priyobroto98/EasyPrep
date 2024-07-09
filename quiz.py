from dotenv import load_dotenv
import os
# Load environment variables from the .env file 
load_dotenv()

import spacy.cli
# spacy.cli.download("en_core_web_sm")

from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
import PyPDF2
from PyPDF2 import PdfReader,PdfWriter  # Import PdfReader

#global session -----------------------------------------------------------------------------------------

global_session = {'text': ''}

import pke
from nltk.corpus import stopwords
import string
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn
import requests
import re
import random

def getImportantWords(text): 
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text, language='en')
    
    pos = {'PROPN'}
    stops = list(string.punctuation)
    stops += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stops += stopwords.words('english')
    
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting()
    
    result = []
    ex = extractor.get_n_best(n=25)
    for each in ex:
        result.append(each[0])
    return result

def splitTextToSents(art):
    s = [sent_tokenize(art)]
    s = [y for x in s for y in x]
    s = [sent.strip() for sent in s if len(sent) > 15]
    return s

def mapSents(impWords, sents):
    processor = KeywordProcessor()
    keySents = {}
    for word in impWords:
        keySents[word] = []
        processor.add_keyword(word)
    for sent in sents:
        found = processor.extract_keywords(sent)
        for each in found:
            keySents[each].append(sent)
    for key in keySents.keys():
        temp = keySents[key]
        temp = sorted(temp, key=len, reverse=True)
        keySents[key] = temp
    return keySents

def getWordSense(sent, word):
    word = word.lower()
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    synsets = wn.synsets(word, 'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

def getDistractors(syn, word):
    dists = []
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return dists
    for each in hypernym[0].hyponyms():
        name = each.lemmas()[0].name()
        if name == actword:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in dists:
            dists.append(name)
    return dists

def getDistractors2(word):
    word = word.lower()
    actword = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    dists = []
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"
    obj = requests.get(url).json()
    for edge in obj['edges']:
        link = edge['end']['term']
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in dists and actword.lower() not in word2.lower():
                dists.append(word2)
    return dists

def generate_and_transform_mcq(text, num_questions):
    # Step 1: Get important words
    impWords = getImportantWords(text)

    # Step 2: Split the text into sentences
    sents = splitTextToSents(text)

    # Step 3: Map sentences to keywords
    mappedSents = mapSents(impWords, sents)

    # Step 4: Get distractors for the keywords
    mappedDists = {}
    for each in mappedSents:
        wordsense = getWordSense(mappedSents[each][0], each)
        if wordsense:
            dists = getDistractors(wordsense, each)
            if len(dists) == 0:
                dists = getDistractors2(each)
            if len(dists) != 0:
                mappedDists[each] = dists
        else:
            dists = getDistractors2(each)
            if len(dists) > 0:
                mappedDists[each] = dists

    # Step 5: Generate MCQs
    mcqs = []
    keywords = list(mappedDists.keys())
    random.shuffle(keywords)
    keywords = keywords[:num_questions]

    for each in keywords:
        sent = mappedSents[each][0]
        p = re.compile(each, re.IGNORECASE)
        op = p.sub("________", sent)
        
        options = [each.capitalize()] + mappedDists[each]
        options = options[:4]
        opts = ['a', 'b', 'c', 'd']
        random.shuffle(options)
        
        mcq = {
            "question": op,
            "options": {opts[i]: options[i] for i in range(len(options))},
            "correct": next(opt for opt, ans in zip(opts, options) if ans == each.capitalize())
        }
        mcqs.append(mcq)

    # Step 6: Transform output
    transformed = []
    for item in mcqs:
        question = item['question']
        options_dict = item['options']
        options = list(options_dict.values())
        correct = item['correct']
        transformed.append((question, options, correct.upper()))

    return transformed

def process_pdf(file):
    # Initialize an empty string to store the extracted text
    text = ""
    # Create a PyPDF2 PdfReader object
    pdf_reader = PdfReader(file)
    # Loop through each page of the PDF
    for page_num in range(len(pdf_reader.pages)):
        # Extract text from the current page
        page_text = pdf_reader.pages[page_num].extract_text()
        # Append the extracted text to the overall text
        text += page_text
    return text


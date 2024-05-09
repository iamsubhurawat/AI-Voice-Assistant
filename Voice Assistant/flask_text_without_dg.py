import os
import uuid
import time
import cohere  
import requests
import json
import requests

from flask                    import Flask, request, render_template, send_file, session, jsonify
from deepgram                 import Deepgram
from langchain_groq           import ChatGroq
from langchain_core.prompts   import ChatPromptTemplate
from langchain_community.embeddings            import HuggingFaceEmbeddings
from langchain_community.vectorstores          import FAISS
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains                          import RetrievalQA
from faster_whisper import WhisperModel
from TTS.api import TTS

cohere_api_key = 'bcQbgPdqJ1S9xNPNI4bKJMSoklBzZzFlYA8waA9H'             #"UVTeHolPAuoQplZYimMg0wJLHw762yL4ROL9OAE8"
deepgram_api_key = "00674d8b76928876dd6423bf209ad815a1e9e485"
groq_api_key = "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"

data_url = 'http://192.168.30.106/magento2/magento/pub/rest/V1/semantic/search'

whisper_model_size = "large-v3"
whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

embeddings    = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

app = Flask(__name__)

app.config["SECRET_KEY"] = 'T/oEntZZNHq2anK7AOTNAmIvbpkyYZKyqV5zlkARwYOaDuBXnS7xJbivhm8uwAYRXBUHj4uVMEnuJ4oTQtKl/A=='

@app.route('/favicon.ico')
def favicon():
    return render_template('index_text.html')

@app.route('/')
def index():
    # if 'session_id' not in session:
    #     session['session_id'] = str(uuid.uuid4())
    return render_template('index_text.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        audio_file = request.files['audio']
        t1 = time.time()
        t_s = round(t1,2)
        audio_file.save(f'Speech to text audio/user{t_s}.wav')
        if audio_file:
            global transcribed_text
            global llm_response
            # transcribed_text = speech_to_text(f'Speech to text audio/user{t_s}.wav')
            transcribed_text = speech_to_text(f'Speech to text audio/user{t_s}.wav')
            documents = get_data(transcribed_text)
            retriever = vectordb(documents)
            llm_response = response_from_llm(transcribed_text,retriever)
            if text_to_speech(llm_response,f'Text to speech audio/ai.wav') == 1:
                audio_file_path = f'Text to speech audio/ai.wav'
                t2 = time.time()
                print(round(t2-t1,2))
                return send_file(audio_file_path, mimetype='audio/wav')
            
@app.route('/llm_response', methods=['GET'])
def get_llm_response():
    return jsonify({"user_query":transcribed_text,"llm_response": llm_response})

def speech_to_text(audio):
    segments, info = whisper_model.transcribe(audio)
    for segment in segments:
        transcribed_text = segment.text

    return transcribed_text

def get_data(user_query):
    # # With Semantic
    # payload = {"query": user_query}
    # response = requests.post(data_url, json=payload)
    # data = json.loads(response.text)    
    # docs = data[0]['data']['metadatas'][0]

    # # GROQ
    # documents = []
    # for d in docs:
    #     st_doc = str(d)
    #     st_doc = st_doc.replace('{','')
    #     st_doc = st_doc.replace('}','')
    #     st_doc = st_doc.replace("'",'')
    #     doc =  Document(page_content=st_doc, metadata={"source": "local"})
    #     documents.append(doc)

    # # COHERE
    # documents = []
    # for doc in docs:
    #     product_desc = ""
    #     for key in doc.keys():
    #         product_desc = product_desc + f"{key}:{doc[key]}" + "\n"
    #     product_name  = doc['name']
    #     d = {"title":product_name, "snippet": product_desc}
    #     documents.append(d)

    documents = []
    for i in range(5):
        product_desc = "Reebok"
        product_name  = f"Shoes{i}"
        d = {"title":product_name, "snippet": product_desc}
        documents.append(d)

    return documents

def vectordb(documents):

    # # GROQ
    # db            = FAISS.from_documents(documents, embeddings)
    # retriever     = db.as_retriever(search_kwargs={"k": 5})
    # return retriever
    
    # COHERE
    return documents

def response_from_llm(user_query,retriever):

    # # GROQ
    # llm = ChatGroq(temperature=0.2,model_name= "llama3-70b-8192",groq_api_key=groq_api_key)
    # chain = RetrievalQA.from_llm(llm=llm,retriever=retriever)
    # result = chain.invoke({"query": user_query})
    # response = result['result'] 
    # print("AI: " + response)

    # COHERE
    llm = cohere.Client(api_key=cohere_api_key)
    result = llm.chat(model="command-r-plus",message=user_query,documents=retriever)
    response = result.text
    print("AI: " + response)

    return response

def text_to_speech(text,path):
    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {
        "Authorization": f"Token {deepgram_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print("Success in TTS!")
        return 1
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == '__main__':
    app.run(host='192.168.15.246',port=5001,debug=True,ssl_context='adhoc')

# pip install pyopenssl --- for running webpage on https://


import logging
import sys
import torch
########INTEL########
#import intel_extension_for_pytorch as ipex
#import modin.pandas as pd
#####################
from pprint import pprint
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index import LangchainEmbedding
from llama_index.prompts.prompts import SimpleInputPrompt
from transformers import LlamaTokenizer, LlamaForCausalLM

from pathlib import Path
from llama_index import download_loader

PDFReader = download_loader("PDFReader")
loader = PDFReader()

system_prompt = "You are a data extractor. Extract the exact data from given document. If no information is found, please reply 'No information found'"
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

context_window = 4096
temperature = 0.0
tokenizer_name = 'meta-llama/Llama-2-7b-chat-hf'
model_name = 'huggyllama/llama-7b'
'''
if torch.xpu.is_available():
    seed = 88
    random.seed(seed)
    torch.xpu.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)
'''
def select_device(preferred_device=None):
    """
    Selects the best available XPU device or the preferred device if specified.

    Args:
        preferred_device (str, optional): Preferred device string (e.g., "cpu", "xpu", "xpu:0", "xpu:1", etc.). If None, a random available XPU device will be selected or CPU if no XPU devices are available.

    Returns:
        torch.device: The selected device object.
    """
    try:
        if preferred_device and preferred_device.startswith("cpu"):
            print("Using CPU.")
            return torch.device("cpu")
        if preferred_device and preferred_device.startswith("xpu"):
            if preferred_device == "xpu" or (
                ":" in preferred_device
                and int(preferred_device.split(":")[1]) >= torch.xpu.device_count()
            ):
                preferred_device = (
                    None  # Handle as if no preferred device was specified
                )
            else:
                device = torch.device(preferred_device)
                if device.type == "xpu" and device.index < torch.xpu.device_count():
                    vram_used = torch.xpu.memory_allocated(device) / (
                        1024**2
                    )  # In MB
                    print(
                        f"Using preferred device: {device}, VRAM used: {vram_used:.2f} MB"
                    )
                    return device

        if torch.xpu.is_available():
            device_id = random.choice(
                range(torch.xpu.device_count())
            )  # Select a random available XPU device
            device = torch.device(f"xpu:{device_id}")
            vram_used = torch.xpu.memory_allocated(device) / (1024**2)  # In MB
            print(f"Selected device: {device}, VRAM used: {vram_used:.2f} MB")
            return device
    except Exception as e:
        print(f"An error occurred while selecting the device: {e}")
    print("No XPU devices available or preferred device not found. Using CPU.")
    return torch.device("cpu")

selected_device = select_device("xpu")

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
model_llm = (
    LlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    .to(selected_device)
    .eval()
)

# model_llm = ipex.optimize(model_llm, weights_prepack=False)
'''
if hasattr(ipex, "optimize_transformers"):
    print("Optimizing transformer")
    try:
        ipex.optimize_transformers(model_llm, dtype=torch.bfloat16)
    except:
        ipex.optimize(model_llm, dtype=torch.bfloat16)
else:
    print("Simple Optimization")
    ipex.optimize(model_llm, dtype=torch.bfloat16)
'''
llm = HuggingFaceLLM(
    context_window=context_window,
    max_new_tokens=256,
    generate_kwargs={"temperature":temperature, "do_sample": False},
    system_prompt= system_prompt,
    query_wrapper_prompt = query_wrapper_prompt,
    tokenizer=tokenizer,
    model=model_llm,
    device_map='auto',
    model_kwargs={"use_auth_token": True}
)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

vector_store_data = {}

def create_vector_stores():
    print("Creating vector stores....")

#    pdfs_list = ["83561-1992___jonew__judis__19509.pdf","8356-2015___supremecourt__2015__8356__8356_2015_Judgement_12-Oct-2017.pdf","9985-2001___jonew__judis__18029.pdf","8359-2004___jonew__judis__26717.pdf","8358-1997___jonew__judis__20264.pdf","8352-2000___jonew__judis__33988.pdf","8352-2008___jonew__judis__44860.pdf","8352-2016___jonew__judis__43552.pdf","8353-1997___jonew__judis__19100.pdf","8354-2006___jonew__judis__31471.pdf","9923-2017___supremecourt__2017__9923__9923_2017_Judgement_23-Feb-2018.pdf","indian-penal-code.pdf"]
    pdfs_list = ["8352-2016___jonew__judis__43552.pdf","8358-1997___jonew__judis__20264.pdf","8359-2004___jonew__judis__26717.pdf","8352-2000___jonew__judis__33988.pdf"] 
    all_docs_store=[]
    for pdf_name in pdfs_list:
        documents = loader.load_data(file=Path('./LegalDocuments/'+pdf_name))
        all_docs_store.extend(documents)
        vector_store_data[pdf_name] = VectorStoreIndex.from_documents(documents, service_context=service_context)
    vector_store_data[""] = VectorStoreIndex.from_documents(all_docs_store, service_context=service_context)
    print("Vector Stores created")

create_vector_stores()

import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

response_dict = {}

def response_creator(gen, query):
    global response_dict
    print("response creator started")
    for text in gen.response_gen:
        if len(text) != 0:
            response_dict[query] = response_dict[query] + str(text)
            print(response_dict[query])
    print("thread completed")
    response_dict[query] = response_dict[query] + '{##eoa##}'

@app.route('/chat', methods=['POST'])
def get():
    json_data = request.json
    query = json_data["query"]
    context = json_data["context"]
    thread_start = json_data["init"]
    if thread_start:
        index = vector_store_data[context]
        query_engine = index.as_query_engine(streaming=True)
        # streaming_response=query_engine.query(query)
        # json_data['response'] = response.response
        
        response_dict[context] = ""
        response_gen = query_engine.query(query)
        generation_kwargs = dict(
            gen = response_gen,
            query = context,
        )
        thread = Thread(target=response_creator, kwargs=generation_kwargs)
        thread.start()
        json_data['response'] = "thread started"
    else:
        json_data['response']=response_dict[context];
    
    return jsonify(json_data)

app.run(host='0.0.0.0', port=5000)

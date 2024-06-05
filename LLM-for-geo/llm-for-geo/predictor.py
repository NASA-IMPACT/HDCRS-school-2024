import boto3
import datetime
import gc
import json
import os
import torch
import time

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from peft import get_peft_model, LoraConfig
from pydantic import BaseModel

from safetensors import safe_open
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM


BUCKET_NAME = 'workhsop-llama-weights'
LORA_FILENAME = 'lora_tensors.safetensors'

torch.set_default_device("cuda")

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

PROMPT = """Today's date is May 30, 2024. you are given a string with information of an event with time and location. you should extract the information from the given text in json format as shown:
{{"datetime": "", "location": ""}} 

here's the text: {text}
json:
"""


class Item(BaseModel):
    query: str


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def download_llama():
    session = assumed_role_session()
    my_bucket = session.resource('s3').Bucket(BUCKET_NAME)
    if not(os.path.exists('llama')):
        os.mkdir('llama')

    # download file into current directory
    for s3_object in my_bucket.objects.filter(Prefix='llama/').all():
        # Need to split s3_object.key into path and file name, else it will give error file not found.
        path, filename = os.path.split(s3_object.key)
        if filename:
            my_bucket.download_file(s3_object.key, f"llama/{filename}")
    

def prepare_base():
    download_llama()
    return LlamaForCausalLM.from_pretrained(
        "llama",
        use_cache=True,
        quantization_config=BNB_CONFIG
    )


def assumed_role_session():
    # Assume the "notebookAccessRole" role we created using AWS CDK.
    client = boto3.client("sts")
    return boto3.session.Session()


def download_lora():
    session = assumed_role_session()
    bucket_name = os.environ.get('BUCKET_NAME', BUCKET_NAME)
    lora_filename = os.environ.get('LORA_FILENAME', LORA_FILENAME) 
    s3_connection = session.client('s3')
    if not(os.path.exists('lora')):
        os.mkdir('lora')
    print(bucket_name, lora_filename)
    s3_connection.download_file(bucket_name, lora_filename, f"lora/{lora_filename}")


def prepare_lora():
    model = prepare_base()
    download_lora()
    tokenizer = AutoTokenizer.from_pretrained('llama')
    llama_peft_sd = model.state_dict()

    tensors = {}
    lora_filename = os.getenv('LORA_FILENAME', LORA_FILENAME)
    with safe_open(f"lora/{lora_filename}", framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    for k in llama_peft_sd:
        if 'lora' in k:
            llama_peft_sd[k] = tensors[k]

    model.load_state_dict(llama_peft_sd)
    peft_config = LoraConfig(
                task_type="CAUSAL_LM", 
                inference_mode=True,
                r=64,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
            )
    return get_peft_model(model, peft_config), tokenizer
    

MODEL, TOKENIZER = prepare_lora()

@app.get("/ping", status_code=200)
def health():
    return {}

@app.post("/invocations")
def infer(item: Item):
    inputs = TOKENIZER(PROMPT.format(text=item.query), return_tensors="pt", return_attention_mask=False)
    outputs = MODEL.generate(**inputs, max_length=150)
    text = TOKENIZER.batch_decode(outputs)[0]
    response = text.split("json:")[-1]
    print(response[:response.find("}") +1])
    return JSONResponse({"models": response[:response.find("}") + 1]})


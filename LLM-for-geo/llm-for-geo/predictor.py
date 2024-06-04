import boto3
import datetime
import gc
import json
import os
import torch
import time

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


torch.set_default_device("cuda")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def assumed_role_session():
    # Assume the "notebookAccessRole" role we created using AWS CDK.
    client = boto3.client("sts")
    return boto3.session.Session()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


@app.get("/ping", status_code=200)
def health():
    return {}


@app.get("/invocations")
def infer(content):
    print(content)
    response = jsonable_encoder(["testing", "heelo", "world"])
    return JSONResponse({"models": response})

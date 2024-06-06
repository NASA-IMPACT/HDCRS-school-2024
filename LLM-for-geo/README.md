# LLM for Geo
This part of the summer school covers using Large Language models for a Geospatial Usecase.

# LLM Task
The main task is a Entity extraction usecase where an Input string is converted into structured format.
This structured format will be consumed 

# LLM Variations Hands-on
The Task is approached in 3 different ways, each with their pros and cons:

1. Encoder Model NER (Jupyter Notebook)
2. Decoder Model Prompt Engineering (Jupyter Notebook)
3. Decoder Model Finetuning using LORA (SLURM)

Finally, Agentic workflows are discussed briefly, with a demo.

# LLM Deployment and Inference
This part covers how the LLM models are deployed on AWS environment and is used for inference


## Getting Started: AWS

1. Login to AWS account.

2. Go To Sagemaker > create Jupyterlab instance

3. Open the terminal, and clone the repo
`git clone https://github.com/NASA-IMPACT/HDCRS-school-2024.git`

4. navigate to downloaded repo and follow encoder-ner.ipynb


## Getting Started: JSC

1. Clone the repo within your environment.

If you have cloned already:

cd $PROJECT/<user>/HDCRS-school-2024

`git pull origin main`

else, clone the repo:
`git clone https://github.com/NASA-IMPACT/HDCRS-school-2024.git`

2. Create a Jupyter Notebook instance within JSC

  goto https://judoor.fz-juelich.de/
  Login
  Click on jupyter-jsc under Connected Services
  system > Jureca, Partition > LoginNode

3. navigate to downloaded repo and follow encoder-ner.ipynb

4. navigate to downloaded repo and follow decoder-prompt-quant.ipynb for decoder based Extraction.

5. Steps for Decoder Finetuning:

From a terminal: (with pwd as LLM-for-geo)
    1. Activate your HDCRS environment

    ```module --force purge
    ml Stages/2024
    ml CUDA
    git clone https://github.com/meta-llama/llama-recipes.git
    cd llama-recipes
    pip install -U pip setuptools
    pip install -e .

    2. modify decoder-ft.sh file -> 
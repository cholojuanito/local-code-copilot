import argparse

import torch 
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import Response
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define stopping behavior for python-esque generated code
stop_tokens = ['def', 'class', 'Instruction', 'Output']
stop_token_ids = [589, 823, 9597, 2301]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = stop_token_ids
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
# connect to vector db
connections.connect('default', host='localhost', port='19530')
collection_name = 'codepilot_python'
collection = Collection(collection_name)

embedder = SentenceTransformer("krlvi/sentence-t5-base-nlpl-code_search_net")
embedder = embedder.to(device)

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Deci/DeciCoder-1b')
tokenizer.add_special_tokens({'additional_special_tokens': stop_tokens}, replace_additional_special_tokens=False)

model = AutoModelForCausalLM.from_pretrained('Deci/DeciCoder-1b', torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.to(device)

# util for tokenization
def token_length(text):
    tokens = tokenizer([text], return_tensors="pt")
    return tokens["input_ids"].shape[1]

def search_similar_prompts(prompt):
    search_embedding = embedder.encode(prompt)
    search_params = {
        'metric_type': 'L2',
        'params': {'nprobe': 10}
    }
    return collection.search(
        [search_embedding],
        'embeddings',
        search_params,
        limit=5,
        output_fields=['metadata']
    )

def create_few_shot_prompt(results, user_prompt):
     # create few-shot prompt examples
    examples = [(f"Instruction: {hit.entity.metadata['instruction']}\n"
                f"Output: {hit.entity.metadata['output']}\n\n") for hits in results for hit in hits]
    
    prompt_instruction = (
        'You are an expert software engineer who specializes in Python. '
        'Write python code to fulfill the request from the user.\n\n'
    )
    prompt_user = f'Instruction: {user_prompt}\nOutput: '
    max_tokens = 2048
    token_count = token_length(prompt_instruction+prompt_user)

    prompt_examples = ''
    # add few shot prompt examples but do not go over the max token embeddings
    for example in examples:
        token_count += token_length(example)
        if token_count < max_tokens:
            prompt_examples += example
        else:
            break

    return f'{prompt_instruction}{prompt_examples}{prompt_user}'

# handle loading vector db in and out of memory
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load on startup
    collection.load()
    yield
    # release collection on shutdown
    collection.release()

# the API/server
app = FastAPI(lifespan=lifespan)

@app.post('/generate')
async def generate(request: Request) -> Response:
    """Generate LLM Response

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    """

    request_dict = await request.json()
    prompt = request_dict.pop('prompt')

    # RAG
    # results = search_similar_prompts(prompt)
    # full_prompt = create_few_shot_prompt(results, prompt)
    full_prompt = prompt

    # generate response
    inputs = tokenizer(full_prompt, return_tensors='pt').to(device)
    response_tokens = model.generate(
        inputs['input_ids'],
        max_new_tokens=1024,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(response_tokens[0][input_length:], skip_special_tokens=True)

    return response

if __name__ == '__main__':
    # start server
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=None)
    parser.add_argument('--port', type=int, default=8000)

    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level='debug')
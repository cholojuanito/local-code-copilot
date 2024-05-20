from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

from transformers import AutoTokenizer
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm
from uuid import uuid4

# connect to Milvus
connections.connect('default', host='localhost', port='19530')

class PythonCodeIngestor:
    def __init__(self, collection, python_code=None, embedder=None, tokenizer=None, text_spliter=None, batch_limit=100) -> None:
        self.collection = collection
        self.python_code = python_code or load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
        self.embedder = embedder or SentenceTransformer('krlvi/sentence-t5-base-nlpl-code_search_net')
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('Deci/Decicoder-1b')
        self.text_splitter = text_spliter or RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' ', ''], chunk_size=400, chunk_overlap=20, length_function=self.token_length)
        self.batch_limit = batch_limit

    def token_length(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def get_metadata(self, page: dict) -> None:
        return {
            'instruction': page['instruction'],
            'input': page['input'],
            'output': page['output']
        }
    
    def spilt_texts_and_metadatas(self, page: dict) -> None:
        basic_metadata = self.get_metadata(page)
        prompts = self.text_splitter.split_text(page['prompt'])
        metadatas = [{'chunk': j, 'prompt': prompt, **basic_metadata} for j, prompt in enumerate(prompts)]

        return prompts, metadatas
    
    def upload_batch(self, texts, metadatas) -> None:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeddings = self.embedder.encode(texts)
        self.collection.insert([ids, embeddings, metadatas])

    def batch_upload(self) -> None:
        batch_texts = []
        batch_metadatas = []

        for page in tqdm(self.python_code):
            texts, metadatas = self.spilt_texts_and_metadatas(page)

            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            if len(batch_texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)

        self.collection.flush()


if __name__ == '__main__':
    collection_name = 'codepilot_python'
    dim = 768 # from the selected embedding model

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(
            name='ids',
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=36,
        ),
        FieldSchema(
            name='embeddings',
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
        ),
        FieldSchema(
            name='metadata', 
            dtype=DataType.JSON
        )
    ]

    coll_schema = CollectionSchema(fields, f'{collection_name} is a collection of python code prompts')

    print(f'Create collection {collection_name}')
    collection = Collection(collection_name, coll_schema)

    # connect to collection
    collection = Collection(collection_name)
    # show zero entries
    print(collection.num_entities)

    # ingest data and show stats after
    python_code_ingestor = PythonCodeIngestor(collection)
    python_code_ingestor.batch_upload()
    print(collection.num_entities)

    # build a search index
    search_idx_params = {
        'index_type': 'IVF_FLAT',
        'metric_type': 'L2',
        'params': {'nlist': 128} # number of clusters
    }

    collection.create_index('embeddings', search_idx_params)
        
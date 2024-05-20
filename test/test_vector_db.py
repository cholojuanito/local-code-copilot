import pytest
from sentence_transformers import SentenceTransformer
from pymilvus import connections, utility, Collection


# connect to Milvus
connections.connect('default', host='localhost', port='19530')

@pytest.fixture
def embedder():
    return SentenceTransformer('krlvi/sentence-t5-base-nlpl-code_search_net')

@pytest.fixture
def idx2search():
    return 'embeddings'

@pytest.fixture
def collection_name():
    return 'codepilot_python'

def test_db_search(collection_name, embedder, idx2search):
    # connect to collection
    collection = Collection(collection_name)

    collection.load()

    query = (
        'Construct a neural network model in Python to classify '
        'the MNIST data set correctly.'
    )
    search_embedding = embedder.encode(query)
    search_params = {
        'metric_type': 'L2',
        'params': {'nprobe': 10} # number of clusters to search
    }

    results = collection.search([search_embedding], idx2search, 
                                search_params, limit=3, output_fields=['metadata'])
    
    # assert len(results) > 0
    # assert len(results) == 3
    for hits in results:
        for hit in hits:
            print(hit.distance)
            print(hit.entity.metadata['instruction'])
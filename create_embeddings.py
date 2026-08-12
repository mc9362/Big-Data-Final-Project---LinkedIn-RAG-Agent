import chromadb
from chromadb.utils.embedding_functions.chroma_cloud_qwen_embedding_function import ChromaCloudQwenEmbeddingFunction, ChromaCloudQwenEmbeddingModel
import pandas as pd
import os

EMBEDDING_MODEL = ChromaCloudQwenEmbeddingModel.QWEN3_EMBEDDING_0p6B
COLLECTION_NAME = "job_postings"
CHUNKS_PATH = "Data/documents.parquet"     
EMBED_DIMENSION = 768        
TOP_K = 5

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")


def get_chunks(path):
	"""Get chunks from parquet file, already processed"""
	df = pd.read_parquet(path)

	document_list = df["doc"].tolist()
	metadata_list = df["metadata"].tolist()


	# find unique values for the metadata fields to identify potential filters
	work_types = sorted(list({metadata_dict['work_type'] for metadata_dict in metadata_list}))
	exp_levels = sorted(list({metadata_dict['experience_level'] for metadata_dict in metadata_list}))
	titles = sorted(list({metadata_dict['title'] for metadata_dict in metadata_list}))

	return document_list, metadata_list, work_types, exp_levels, titles


def create_embeddings (document_list, metadata_list):
	"""Add all chunks and metadata to ChromaDB database"""
	chroma_client = chromadb.CloudClient(
		api_key = CHROMA_API_KEY,
		tenant = CHROMA_TENANT,
		database = CHROMA_DATABASE
	)
	embedding_fn = ChromaCloudQwenEmbeddingFunction(
		model = EMBEDDING_MODEL, # go with default
		task = "text_retrieval"
	)

	# Delete collection  
	try:
	    chroma_client.delete_collection(name=COLLECTION_NAME)
	except:
	    pass

	# create new collection 
	collection = chroma_client.create_collection(
	    name=COLLECTION_NAME,
	    metadata={"description": "LinkedIn Job Descriptions", "hnsw:space": "cosine"},
	    embedding_function = embedding_fn
	)

	batch_size = 100
	length = len(document_list)

	for i in range(0, length, batch_size):
		end = min(length, i+batch_size)
		collection.add(
			ids = [f"{j}" for j in range(i, end)],
			documents = document_list[i:end],
			metadatas = metadata_list[i:end]
		)
		print(f"Processed {end}/{length} records")

	return collection



def retrieve_chunks(collection, query, metadata, top_k = TOP_K):
	"""Given query, retrieve the top_k most relevant chunks and their sources"""

	# add the user input for metadata
	filters = ["work_type", "experience_level", "title"]
	where = []
	for i in range(len(metadata)):
		if len(metadata[i]) > 0 : # filter for each option that was chosen
			where.append({filters[i]: {"$in":metadata[i]}})

	# need to add and if more than 1 filter
	if len(where) >1:
		where = {"$and":where}
	elif len(where) ==1:
		where = where[0]

	# no filter vs filter
	if where:
		results = collection.query(
			query_texts = [query],
			n_results = top_k,
			where = where
		)
	else:
		results = collection.query(
			query_texts = [query],
			n_results = top_k
		)
	print(where)

	# extract chunk and source (job_id)
	final_results = [] 
	for i in range(len(results["documents"][0])):
		final_results.append({
			"chunk_text" : results["documents"][0][i],
			"job_id" : results["metadatas"][0][i]["job_id"]
		})

	return final_results

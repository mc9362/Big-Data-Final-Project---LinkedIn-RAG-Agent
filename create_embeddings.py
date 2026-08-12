# referenced: https://docs.trychroma.com/cloud/schema/sparse-vector-search

from chromadb import Schema, SparseVectorIndexConfig, K, CloudClient, Knn, Rrf, Search
from chromadb.utils.embedding_functions import ChromaCloudSpladeEmbeddingFunction
import pandas as pd
import os

COLLECTION_NAME = "job_postings"
CHUNKS_PATH = "Data/documents.parquet"     
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
	chroma_client = CloudClient(
		api_key = CHROMA_API_KEY,
		tenant = CHROMA_TENANT,
		database = CHROMA_DATABASE
	)
	schema = Schema()
	sparse_ef = ChromaCloudSpladeEmbeddingFunction() # default keyword embedding model
	schema.create_index(
	    config=SparseVectorIndexConfig(
	        source_key=K.DOCUMENT,
	        embedding_function=sparse_ef
	    ),
	    key="sparse_embedding"
	)

	# Delete collection  
	try:
	    chroma_client.delete_collection(name=COLLECTION_NAME)
	except:
	    pass

	# create new collection 
	collection = chroma_client.create_collection(
	    name=COLLECTION_NAME,
	    metadata={"description": "LinkedIn Job Descriptions"},
	    schema = schema
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

	# use RRF for hybrid search
	hybrid_rank = Rrf(
	    ranks=[
	        Knn(query=query, return_rank=True), # semantic search, default all-MiniLM-L6-v2
	        Knn(query=query, key="sparse_embedding", return_rank=True) # keyword search
	    ],
	    weights=[0.7, 0.3],  # 70% semantic, 30% keyword
	    k=60 # default
	)



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
	else: 
		where = None
	print(where)

	search_args = {
		"rank": hybrid_rank,
		"limit": top_k,
		"select": ["#document", "#metadata"]
	}

	if where: 
		search_args['where'] = where

	results = collection.search(Search(**search_args))



	# extract chunk and source (job_id)
	final_results = [] 
	for i in range(len(results["documents"][0])):
		final_results.append({
			"chunk_text" : results["documents"][0][i],
			"job_id" : results["metadatas"][0][i]["job_id"]
		})

	return final_results

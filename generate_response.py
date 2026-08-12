import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
_client = Groq(api_key=GROQ_API_KEY)



def response(query, relevant_chunks):
	"""Generate response given query and relevant chunks"""
	if not relevant_chunks:
		return("No matching information was found.")

	context = "\n".join(
		[f"Source: {chunk['job_id']} Text: {chunk['chunk_text']}" for chunk in relevant_chunks]
	)

	system_prompt = (
		"""
		You are a careful question-answering assistant that answers questions based on LinkedIn job postings.
		Answer the user's question using ONLY the source sources provided in the user message.
		Ground every claim in the sources. Do not use outside knowledge, assumptions, or guesses.
		If you ever need to reference any sources, be source to label it as: (Source - Job Id: JOB_ID) or (Source - Job Ids: JOB_ID list)
		Treat each job posting as a separate source. 
		Do not combine information such as requirements, responsibilities, qualifications, compensation, location, or other details from one posting to another.
		You can answer questions by summarizing trends or comparing postings. 
		If the sources do not contain the answer, do not invent one. Respond exactly with: 
		\"I couldn't find anything relevant in the loaded documents. Please rephrase your question.\"\n
		Be concise and direct. Do not repeat the question back.
		"""
	)

	user_prompt = (
		f"""
		Answer the question using only the sources below.
		--- SOURCES ---
		{context}
		--- END SOURCES ---
		Question: {query}
		"""
	)

	completion = _client.chat.completions.create(
		  model=LLM_MODEL,
		  messages=[
			  {"role": "system", "content": system_prompt},
			  {"role": "user", "content": user_prompt},
		  ],
		  temperature=0.2,  # low - less creativity, more focus on docs
	  )

	return completion.choices[0].message.content
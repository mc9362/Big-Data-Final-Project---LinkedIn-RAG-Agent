import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from create_embeddings import get_chunks, create_embeddings, retrieve_chunks, CHUNKS_PATH
from generate_response import response


document_list, metadata_list, work_types, exp_levels, titles= get_chunks(CHUNKS_PATH)

print("Creating embeddings...")
collection = create_embeddings(document_list, metadata_list)
print("Finished creating embeddings. Retrieving relevant chunks...")

def answer_query(query, drop_work_type, drop_exp_level, drop_titles):
	query = query.strip()
	if not query:
		return "Please enter a question", ""

	metadata = [drop_work_type, drop_exp_level, drop_titles]
	chunks = retrieve_chunks(collection, query, metadata)

	final_response = response(query, chunks)
	sources = [f"https://www.linkedin.com/jobs/view/{chunk['job_id']}" for chunk in chunks]

	return final_response, sources 


with gr.Blocks() as demo:
	# inputs
    inp = gr.Textbox(label="Your question")
    btn = gr.Button("Ask")
    drop_work_type = gr.Dropdown(
    	choices = work_types,
    	multiselect = True, 
    	label = "Choose employment type"
    )
    drop_exp_level = gr.Dropdown(
    	choices = exp_levels,
    	multiselect = True, 
    	label = "Choose experience level"
    )
    drop_titles = gr.Dropdown(
        choices = titles,
        multiselect = True, 
        label = "Choose job title"
    )

    # outputs
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Sources", lines=10)

    # submit input
    btn.click(answer_query, inputs=[inp, drop_work_type, drop_exp_level, drop_titles], outputs=[answer, sources])
    inp.submit(answer_query, inputs=[inp, drop_work_type, drop_exp_level, drop_titles], outputs=[answer, sources])

demo.launch()


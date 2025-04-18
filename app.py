import os
import uuid
from datetime import datetime
import gradio as gr
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sklearn.metrics.pairwise import cosine_similarity
import re

# Suppress Hugging Face tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

# Global list to track uploaded files
UPLOADED_FILES = []


class RAGEvaluator:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
        self.collection_name = "rag_evaluation"
        self.voting_collection = "voting_results"
        self.embedding_models = [
            {"model": "text-embedding-3-small", "dim": 1536, "type": "openai"},
            {"model": "text-embedding-3-large", "dim": 1536, "type": "openai"},
        ]
        self.chunk_sizes = [300, 500, 800]
        self.similarity_thresholds = [0.3, 0.5]  # Lowered to make retrieval less restrictive
        self.retrieval_limits = [3, 5, 7]
        self._ensure_collections()

    def _ensure_collections(self):
        """Ensure Qdrant collections exist with 1536 dimensions, delete if dimension mismatch."""
        for collection in [self.collection_name, self.voting_collection]:
            if self.qdrant_client.collection_exists(collection_name=collection):
                # Check collection configuration
                collection_info = self.qdrant_client.get_collection(collection_name=collection)
                vector_size = collection_info.config.params.vectors.size
                if vector_size != 1536:
                    # Delete and recreate if dimension is incorrect
                    self.qdrant_client.delete_collection(collection_name=collection)
                    self.qdrant_client.create_collection(
                        collection_name=collection,
                        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
                    )
            else:
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
                )

    def _get_embeddings(self, model_name, model_type):
        """Initialize OpenAI embeddings with 1536 dimensions."""
        if model_type == "openai":
            dimensions = 1536 if model_name == "text-embedding-3-large" else None
            return OpenAIEmbeddings(
                model=model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                dimensions=dimensions
            )
        raise ValueError(f"Unsupported model type: {model_type}")

    def _extract_text(self, file_path):
        """Extract text from PDF using PyMuPDF."""
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text("text") or ""
        return text

    def _chunk_text(self, text, max_chunk_size, similarity_threshold, embedding_model, model_type):
        """Semantic chunking with specified parameters."""
        embeddings = self._get_embeddings(embedding_model, model_type)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_embeddings = embeddings.embed_documents(sentences)
        chunks, chunk, chunk_embedding = [], [], []
        chunk_token_length = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            if not chunk:
                chunk.append(sentence)
                chunk_embedding.append(embedding)
                chunk_token_length = len(sentence)
                continue

            similarity = cosine_similarity([embedding], [chunk_embedding[-1]])[0][0]
            if chunk_token_length + len(sentence) > max_chunk_size:
                if chunk_token_length >= 100:
                    chunks.append(" ".join(chunk))
                chunk = [sentence]
                chunk_embedding = [embedding]
                chunk_token_length = len(sentence)
            elif similarity > similarity_threshold:
                chunk.append(sentence)
                chunk_embedding.append(embedding)
                chunk_token_length += len(sentence)
            else:
                if chunk_token_length >= 100:
                    chunks.append(" ".join(chunk))
                chunk = [sentence]
                chunk_embedding = [embedding]
                chunk_token_length = len(sentence)

        if chunk and (chunk_token_length >= 100 or len(chunks) == 0):
            chunks.append(" ".join(chunk))
        return list(dict.fromkeys(chunks))

    def process_and_store_pdf(self, file_obj, pipeline_configs):
        """Process PDF and store with different pipeline configurations."""
        global UPLOADED_FILES
        text = self._extract_text(file_obj.name)
        filename = os.path.basename(file_obj.name)
        if filename not in UPLOADED_FILES:
            UPLOADED_FILES.append(filename)
        batch_size = 50

        for config in pipeline_configs:
            chunks = self._chunk_text(
                text,
                config["chunk_size"],
                config["similarity_threshold"],
                config["embedding_model"],
                config["model_type"]
            )
            embeddings = self._get_embeddings(config["embedding_model"], config["model_type"]).embed_documents(chunks)

            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source_file": filename,
                            "pipeline_config": {
                                "embedding_model": config["embedding_model"],
                                "chunk_size": config["chunk_size"],
                                "similarity_threshold": str(config["similarity_threshold"]),
                                "retrieval_limit": config["retrieval_limit"],
                                "config_id": config["config_id"]
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                )

                if len(points) >= batch_size:
                    try:
                        self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                        points = []
                    except Exception as e:
                        print(f"Error upserting points: {str(e)}")
                        raise

            if points:
                try:
                    self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                except Exception as e:
                    print(f"Error upserting remaining points: {str(e)}")
                    raise

        return filename

    def query_pipeline(self, query, config):
        """Query the vector database with specific pipeline configuration."""
        embeddings = self._get_embeddings(config["embedding_model"], config["model_type"])
        query_embedding = embeddings.embed_query(query)
        query_response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=config["retrieval_limit"],
            with_payload=True,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="pipeline_config.embedding_model",
                        match=models.MatchValue(value=config["embedding_model"])
                    ),
                    models.FieldCondition(
                        key="pipeline_config.chunk_size",
                        match=models.MatchValue(value=config["chunk_size"])
                    ),
                    models.FieldCondition(
                        key="pipeline_config.similarity_threshold",
                        match=models.MatchValue(value=str(config["similarity_threshold"]))
                    ),
                    # Removed retrieval_limit from filter to allow matching across different limits
                ]
            )
        )
        search_results = query_response.points
        if not search_results:
            return {"context": "No relevant information found.", "chunks": []}

        context = "\n\n".join([result.payload.get("text", "") for result in search_results])
        chunks = [result.payload.get("text", "") for result in search_results]
        return {"context": context, "chunks": chunks}

    def delete_document(self, filename):
        """Delete all points associated with a file from Qdrant."""
        global UPLOADED_FILES
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_file",
                                match=models.MatchValue(value=filename)
                            )
                        ]
                    )
                )
            )
            if filename in UPLOADED_FILES:
                UPLOADED_FILES.remove(filename)
            file_path = os.path.join(os.getcwd(), filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            return f"Deleted {filename} from Qdrant and local storage."
        except Exception as e:
            return f"Error deleting {filename}: {str(e)}"

    def generate_answer(self, query, context):
        """Generate answer using LLM."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""Based solely on the provided context, provide a concise answer to the question.

Question: {question}
Context: {context}

Answer: """
        )
        chain = prompt | llm
        return chain.invoke({"question": query, "context": context}).content.strip()

    def get_pipeline_combinations(self):
        """Return all combinations of pipeline configurations using OpenAI embeddings."""
        # Generate all possible combinations to ensure all user-selectable configs are stored
        return [
            {
                "embedding_model": model["model"],
                "model_type": "openai",
                "chunk_size": chunk_size,
                "similarity_threshold": sim_threshold,
                "retrieval_limit": retrieval_limit,
                "config_id": f"{model['model']}_{chunk_size}_{sim_threshold}_{retrieval_limit}"
            }
            for model in self.embedding_models
            for chunk_size in self.chunk_sizes
            for sim_threshold in self.similarity_thresholds
            for retrieval_limit in self.retrieval_limits
        ]

    def store_vote(self, config_a, config_b, winner_config_id, query):
        """Store voting results in Qdrant."""
        point_id = str(uuid.uuid4())
        self.qdrant_client.upsert(
            collection_name=self.voting_collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=[0.0] * 1536,
                    payload={
                        "config_a": config_a,
                        "config_b": config_b,
                        "winner": winner_config_id,
                        "query": query,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            ]
        )


def compare_pipelines(query, model_a, chunk_size_a, sim_threshold_a, retrieval_limit_a, model_b, chunk_size_b, sim_threshold_b, retrieval_limit_b):
    """Compare two pipeline configurations based on user-selected parameters."""
    if not query:
        return "Please enter a question.", ""

    # Construct pipeline configurations
    config_a = {
        "embedding_model": model_a,
        "model_type": "openai",
        "chunk_size": int(chunk_size_a),
        "similarity_threshold": float(sim_threshold_a),
        "retrieval_limit": int(retrieval_limit_a),
        "config_id": f"{model_a}_{chunk_size_a}_{sim_threshold_a}_{retrieval_limit_a}"
    }
    config_b = {
        "embedding_model": model_b,
        "model_type": "openai",
        "chunk_size": int(chunk_size_b),
        "similarity_threshold": float(sim_threshold_b),
        "retrieval_limit": int(retrieval_limit_b),
        "config_id": f"{model_b}_{chunk_size_b}_{sim_threshold_b}_{retrieval_limit_b}"
    }

    result_a = evaluator.query_pipeline(query, config_a)
    result_b = evaluator.query_pipeline(query, config_b)

    answer_a = evaluator.generate_answer(query, result_a["context"])
    answer_b = evaluator.generate_answer(query, result_b["context"])

    return (
        f"Pipeline A ({config_a['config_id']}):\n{answer_a}",
        f"Pipeline B ({config_b['config_id']}):\n{answer_b}"
    )

def upload_file(file):
    """Upload and process PDF with all pipeline combinations."""
    if not file:
        return "Please upload a PDF file.", gr.Dropdown(choices=UPLOADED_FILES)

    pipeline_configs = evaluator.get_pipeline_combinations()
    filename = evaluator.process_and_store_pdf(file, pipeline_configs)
    return f"Processed: {filename}", gr.Dropdown(choices=UPLOADED_FILES)


def delete_file(filename):
    """Delete a selected file."""
    if not filename:
        return "Please select a file to delete.", gr.Dropdown(choices=UPLOADED_FILES)
    result = evaluator.delete_document(filename)
    return result, gr.Dropdown(choices=UPLOADED_FILES)


def vote(winner, model_a, chunk_size_a, sim_threshold_a, retrieval_limit_a, model_b, chunk_size_b, sim_threshold_b, retrieval_limit_b, query):
    """Record user vote for the winning pipeline."""
    # Construct pipeline configurations
    config_a = {
        "embedding_model": model_a,
        "model_type": "openai",
        "chunk_size": int(chunk_size_a),
        "similarity_threshold": float(sim_threshold_a),
        "retrieval_limit": int(retrieval_limit_a),
        "config_id": f"{model_a}_{chunk_size_a}_{sim_threshold_a}_{retrieval_limit_a}"
    }
    config_b = {
        "embedding_model": model_b,
        "model_type": "openai",
        "chunk_size": int(chunk_size_b),
        "similarity_threshold": float(sim_threshold_b),
        "retrieval_limit": int(retrieval_limit_b),
        "config_id": f"{model_b}_{chunk_size_b}_{sim_threshold_b}_{retrieval_limit_b}"
    }
    winner_config_id = config_a["config_id"] if winner == "Pipeline A" else config_b["config_id"]
    evaluator.store_vote(config_a, config_b, winner_config_id, query)
    return f"Vote recorded for {winner}"


evaluator = RAGEvaluator()

modern_theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
    text_size="md",
)

custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .container { max-width: 1200px; margin: 0 auto; padding: 0 !important; }
    .header { text-align: center; padding: 0.5rem 0; margin-bottom: 0.5rem !important; }
    .output-box { background-color: #16213E; color: white; border: 1px solid #0F3460; border-radius: 8px; padding: 8px; margin: 0.2rem 0 !important; }
    .message-input { background-color: #0F3460; color: white; border: 1px solid #E94560; border-radius: 8px; margin-bottom: 0.3rem !important; }
    .send-btn { background-color: #E94560; color: white; border: none; margin: 0.3rem 0 !important; }
    .send-btn:hover { background-color: #FF6B6B; }
    .file-upload { background-color: #16213E; border: 1px dashed #E94560; border-radius: 8px; height: 150px; margin: 0.2rem 0 !important; }
    .status-text { color: #E94560; font-size: 14px; text-align: center; margin: 0.2rem 0 !important; }
    .gr-column { padding: 0.3rem !important; }
    .gr-row { margin: 0.2rem 0 !important; }
    .gr-accordion { margin: 0.2rem 0 !important; }
    .gr-button { padding: 0.3rem 0.5rem !important; }
    .gr-dropdown { margin: 0.2rem 0 !important; }
"""


def get_config_by_id(config_id):
    """Retrieve a pipeline configuration by its config_id."""
    # Since configs are now dynamic, construct from config_id
    parts = config_id.split("_")
    if len(parts) != 4:
        return None
    return {
        "embedding_model": parts[0],
        "model_type": "openai",
        "chunk_size": int(parts[1]),
        "similarity_threshold": float(parts[2]),
        "retrieval_limit": int(parts[3]),
        "config_id": config_id
    }


with gr.Blocks(theme=modern_theme, css=custom_css) as demo:
    gr.Markdown(
        """
        # RAG Pipeline Evaluator
        <p style='color: red; font-size: 14px; margin: 0;'><b>I've uploaded the popular stories Goldilocks and The Gingerbread Man. Ask questions like “What did Goldilocks do?” or “What happened to the Gingerbread Man?”</b></p>
        """,
        elem_classes="header"
    )

    with gr.Row(elem_classes="container"):
        with gr.Column(scale=8):
            query_input = gr.Textbox(
                placeholder="Enter your question to compare pipelines...",
                show_label=False,
                elem_classes="message-input"
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Pipeline A Configuration")
                    model_a_dropdown = gr.Dropdown(
                        choices=[model["model"] for model in evaluator.embedding_models],
                        label="Model",
                        value=evaluator.embedding_models[0]["model"]
                    )
                    chunk_size_a_dropdown = gr.Dropdown(
                        choices=[str(size) for size in evaluator.chunk_sizes],
                        label="Chunk Size",
                        value=str(evaluator.chunk_sizes[0])
                    )
                    sim_threshold_a_dropdown = gr.Dropdown(
                        choices=[str(threshold) for threshold in evaluator.similarity_thresholds],
                        label="Similarity Threshold",
                        value=str(evaluator.similarity_thresholds[0])
                    )
                    retrieval_limit_a_dropdown = gr.Dropdown(
                        choices=[str(limit) for limit in evaluator.retrieval_limits],
                        label="Retrieval Limit",
                        value=str(evaluator.retrieval_limits[0])
                    )
                with gr.Column():
                    gr.Markdown("### Pipeline B Configuration")
                    model_b_dropdown = gr.Dropdown(
                        choices=[model["model"] for model in evaluator.embedding_models],
                        label="Model",
                        value=evaluator.embedding_models[1]["model"]
                    )
                    chunk_size_b_dropdown = gr.Dropdown(
                        choices=[str(size) for size in evaluator.chunk_sizes],
                        label="Chunk Size",
                        value=str(evaluator.chunk_sizes[1])
                    )
                    sim_threshold_b_dropdown = gr.Dropdown(
                        choices=[str(threshold) for threshold in evaluator.similarity_thresholds],
                        label="Similarity Threshold",
                        value=str(evaluator.similarity_thresholds[1])
                    )
                    retrieval_limit_b_dropdown = gr.Dropdown(
                        choices=[str(limit) for limit in evaluator.retrieval_limits],
                        label="Retrieval Limit",
                        value=str(evaluator.retrieval_limits[1])
                    )
            compare_btn = gr.Button("Compare Pipelines", elem_classes="send-btn")

            with gr.Row():
                output_a = gr.Textbox(label="Pipeline A Output", elem_classes="output-box")
                output_b = gr.Textbox(label="Pipeline B Output", elem_classes="output-box")

            with gr.Row():
                vote_a_btn = gr.Button("Vote for Pipeline A")
                vote_b_btn = gr.Button("Vote for Pipeline B")

            vote_status = gr.Textbox(show_label=False, interactive=False, elem_classes="status-text")

        with gr.Column(scale=4):
            with gr.Accordion("File Upload", open=False):
                file_upload = gr.File(
                    label="Drop PDF here",
                    file_types=[".pdf"],
                    elem_classes="file-upload",
                    height=150  # Reduced height to save space
                )
                upload_btn = gr.Button("Upload PDF")
                upload_status = gr.Textbox(
                    value="Upload a PDF to begin",
                    show_label=False,
                    interactive=False,
                    elem_classes="status-text"
                )

            with gr.Accordion("Manage Documents", open=False):
                file_dropdown = gr.Dropdown(
                    choices=UPLOADED_FILES,
                    label="Select File to Delete",
                    interactive=True
                )
                delete_btn = gr.Button("Delete Selected File")
                delete_status = gr.Textbox(
                    value="Select a file to delete",
                    show_label=False,
                    interactive=False,
                    elem_classes="status-text"
                )

    compare_btn.click(
        compare_pipelines,
        inputs=[
            query_input,
            model_a_dropdown,
            chunk_size_a_dropdown,
            sim_threshold_a_dropdown,
            retrieval_limit_a_dropdown,
            model_b_dropdown,
            chunk_size_b_dropdown,
            sim_threshold_b_dropdown,
            retrieval_limit_b_dropdown
        ],
        outputs=[output_a, output_b]  # Removed chunks_display output
    )

    upload_btn.click(
        upload_file,
        inputs=[file_upload],
        outputs=[upload_status, file_dropdown]
    )

    delete_btn.click(
        delete_file,
        inputs=[file_dropdown],
        outputs=[delete_status, file_dropdown]
    )

    vote_a_btn.click(
        vote,
        inputs=[
            gr.State("Pipeline A"),
            model_a_dropdown,
            chunk_size_a_dropdown,
            sim_threshold_a_dropdown,
            retrieval_limit_a_dropdown,
            model_b_dropdown,
            chunk_size_b_dropdown,
            sim_threshold_b_dropdown,
            retrieval_limit_b_dropdown,
            query_input
        ],
        outputs=[vote_status]
    )

    vote_b_btn.click(
        vote,
        inputs=[
            gr.State("Pipeline B"),
            model_a_dropdown,
            chunk_size_a_dropdown,
            sim_threshold_a_dropdown,
            retrieval_limit_a_dropdown,
            model_b_dropdown,
            chunk_size_b_dropdown,
            sim_threshold_b_dropdown,
            retrieval_limit_b_dropdown,
            query_input
        ],
        outputs=[vote_status]
    )

if __name__ == "__main__":
    print(f"===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
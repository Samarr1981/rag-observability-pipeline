from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever          # keyword search
from langchain_classic.retrievers import EnsembleRetriever        # merges FAISS + BM25
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langsmith.evaluation import evaluate, RunEvaluator, EvaluationResult
from langsmith.schemas import Run, Example
from sentence_transformers import CrossEncoder                     # cross-encoder reranker

load_dotenv()

# 1. Load PDF
loader = PyPDFLoader("rag_paper.pdf")
docs = loader.load()

# 2. Semantic chunking
splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
chunks = splitter.split_documents(docs)

# 3. Hybrid retrieval: FAISS (semantic) + BM25 (keyword) merged by EnsembleRetriever
#    Fetch k=6 from each so the reranker has enough candidates to work with.
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
bm25_retriever  = BM25Retriever.from_documents(chunks, k=6)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5],
)

# 4. Cross-encoder reranker: re-scores the merged candidate pool, keeps top 4
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, top_k: int = 4) -> list:
    candidates = ensemble_retriever.invoke(query)
    scores = _cross_encoder.predict([[query, d.page_content] for d in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

retriever = RunnableLambda(rerank)

# 5. Prompt with citations
prompt = PromptTemplate.from_template("""
You are an assistant answering questions about a research paper.
Use ONLY the context below. At the end of your answer, cite the source as:
[Source: Page {{page}}]

Context:
{context}

Question: {question}
Answer:
""")

# 6. Format retrieved chunks with page numbers
def format_docs(docs):
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
    )

# 7. Build the chain (LLM, prompt, and output parser unchanged)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Run test questions
questions = [
    "What problem does RAG solve?",
    "What retrieval method does the RAG paper use?",
    "How does RAG combine retrieval with generation?",
    "What datasets were used to evaluate RAG?",
    "What are the limitations of RAG mentioned in the paper?",
]

print("=" * 60)
for q in questions:
    print(f"Q: {q}")
    print(f"A: {chain.invoke(q)}\n")
    print("-" * 60)

# 9. LangSmith evaluation
eval_questions = [
    {"input": "What problem does RAG solve?", "output": "RAG solves hallucination and knowledge limitations in language models by retrieving relevant documents at inference time."},
    {"input": "What retrieval method does the RAG paper use?", "output": "RAG uses Dense Passage Retrieval (DPR) with a FAISS index over Wikipedia."},
    {"input": "How does RAG combine retrieval with generation?", "output": "RAG retrieves documents using a neural retriever and conditions a seq2seq model on them to generate answers."},
]

client = Client()
dataset_name = "RAG Paper Evaluation"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": q["input"]} for q in eval_questions],
        outputs=[{"answer": q["output"]} for q in eval_questions],
        dataset_id=dataset.id,
    )

# 10. Custom LLM-based correctness evaluator
class CorrectnessEvaluator(RunEvaluator):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def evaluate_run(self, run: Run, example: Example, **kwargs) -> EvaluationResult:
        predicted = run.outputs.get("answer", "")
        expected = example.outputs.get("answer", "")
        question = example.inputs.get("question", "")

        prompt = f"""You are evaluating a RAG pipeline answer.
Question: {question}
Expected answer: {expected}
Actual answer: {predicted}

Is the actual answer correct and grounded in facts? Reply with only: correct or incorrect"""

        response = self.llm.invoke(prompt).content.strip().lower()
        score = 1 if "correct" in response else 0
        return EvaluationResult(key="correctness", score=score, comment=response)
    
def rag_pipeline(inputs):
    return {"answer": chain.invoke(inputs["question"])}

results = evaluate(
    rag_pipeline,
    data=dataset_name,
    evaluators=[CorrectnessEvaluator()],
    experiment_prefix="rag-paper-eval",
)

print("\nEvaluation complete — check LangSmith for scores.")
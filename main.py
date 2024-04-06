"""
RAG-Fusion, Algorithm taken from [https://arxiv.org/abs/2402.03367].

Once the original query is received, the model sends the
original query to the large language model to generate a number of new search queries based on the original query.
The algorithm then performs vector search to find a number of relevant documents like with RAG. But, instead of
sending those documents with the queries to the large language model to generate the output, the model performs
reciprocal rank fusion. Reciprocal rank fusion is an algorithm commonly used in search to assign scores to every
document and rerank them according to the scores. The scores assigned to each document, or rrf scores, are

rrfscore = 1 / (rank + k)

where rank is the current rank of the documents sorted by distance, and k is a constant smoothing factor that determines
the weight given to the existing ranks. Upon each calculation of the score, the rrf score is accumulated with previous
scores for the same document, and when all scores are accumulated, documents are fused together and reranked
according to their scores. The model then sends the reranked results along with the generated queries and the original
queries to the large language model to produce an output.
"""

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import (
    VectorStoreIndex,
    PromptTemplate
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore
from typing import List
from tqdm.asyncio import tqdm
import asyncio
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

loader = PyMuPDFReader()
system_prompt_str = (
    "As an efficient assistant, your task is to create several related search queries from a single given query, "
    "without adding any extra details. Please generate {num_queries} related search queries, "
    "each presented on a separate line, starting from the provided input query as follows:\n"
    "Input Query: {query}\n"
    "Related Search Queries:\n"
)


class RAGFusion:

    def __init__(self, embedding_model):
        documents = loader.load(file_path="./data/RAG-Fusion.pdf")
        splitter = SentenceSplitter(chunk_size=1024)
        self.vector_index: VectorStoreIndex = VectorStoreIndex.from_documents(documents, transformations=[splitter],
                                                                              embed_model=embedding_model)
        self.query_gen_prompt = PromptTemplate(system_prompt_str)

    def generate_queries(self, llm, query_str: str, num_queries: int = 4):
        fmt_prompt = self.query_gen_prompt.format(num_queries=num_queries, query=query_str)
        response = llm.complete(fmt_prompt)
        queries = response.text.split("\n")
        return queries

    def fetch_retrievers(self):
        # vector retriever
        vector_retriever: BaseRetriever = self.vector_index.as_retriever(similarity_top_k=2)

        # bm25 retriever
        bm25_retriever: BM25Retriever = BM25Retriever.from_defaults(docstore=self.vector_index.docstore,
                                                                    similarity_top_k=2)
        return vector_retriever, bm25_retriever

    @classmethod
    async def run_queries(cls, queries, retrievers) -> dict:
        tasks = []
        for query in queries:
            for i, retriever in enumerate(retrievers):
                tasks.append(retriever.aretrieve(query))

        task_results = await tqdm.gather(*tasks)
        results_dict = {}
        for i, (query, query_result) in enumerate(zip(queries, task_results)):
            results_dict[(query, i)] = query_result

        return results_dict

    @classmethod
    def combine_results(self, results: dict, top_k: int = 2):
        smoothing_factor = 60.0  # Adjusts the impact of outlier rankings in fusion.
        combined_scores = {}
        content_to_node = {}

        # Calculate reciprocal rank fusion scores for each node.
        for score_nodes in results.values():
            for position, node in enumerate(sorted(score_nodes, key=lambda n: n.score or 0.0, reverse=True)):
                content = node.node.get_content()
                content_to_node[content] = node
                combined_scores[content] = combined_scores.get(content, 0) + 1 / (position + smoothing_factor)

        # Sort nodes by fused scores in descending order.
        sorted_nodes = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

        # Adjust node scores and prepare the final list of ranked nodes.
        ranked_nodes: List[NodeWithScore] = []
        for content, score in sorted_nodes[:top_k]:
            node_with_score = content_to_node[content]
            node_with_score.score = score  # Directly set the new score.
            ranked_nodes.append(node_with_score)

        return ranked_nodes


if __name__ == "__main__":
    query_string = "What is RAG-Fusion and how is different from traditional RAG ?"

    # change this to any of the embed model and llm of your choice
    llm = Ollama(base_url="http://localhost:11434", model="gemma:latest", temperature=0.1)
    embed_model = OllamaEmbedding(model_name="mxbai-embed-large", base_url="http://localhost:11434")
    ragFusion = RAGFusion(embedding_model=embed_model)

    generated_queries = ragFusion.generate_queries(llm=llm, query_str=query_string, num_queries=5)
    logging.info(generated_queries)

    vector_retriever, bm25_retriever = ragFusion.fetch_retrievers()
    results_dict: dict = asyncio.run(ragFusion.run_queries(generated_queries, [vector_retriever, bm25_retriever]))
    logging.info(results_dict.values())

    final_results = ragFusion.combine_results(results_dict)
    for n in final_results:
        logging.info(f"{n.score},'\n' {n.text}, '\n*****************'")

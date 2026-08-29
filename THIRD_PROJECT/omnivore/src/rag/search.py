# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from typing import Dict, Any
from rag.retriever import RAGRetriever
import time


class AdvancedRAGSearch:
    def __init__(self, retriever: RAGRetriever, llm_model: Any):
        self.retriever = retriever
        self.llm_model = llm_model
        self.history = []  # Store query history

    def query(self, question: str, top_k: int = 5, min_score: float = 0.1, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            system_prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(system_prompt), 80):
                    print(system_prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            # response = self.llm_model.invoke([system_prompt.format(context=context, question=question)])
            response = self.llm_model.invoke([system_prompt])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm_model.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

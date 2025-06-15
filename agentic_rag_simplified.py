#!/usr/bin/env python3
"""
Simplified Agentic RAG System using LangGraph
"""
import json
import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class RAGState:
    """State object to pass between nodes"""
    user_question: str
    kb_hits: List[Dict] = None
    initial_answer: str = None
    critique_result: str = None
    refined_answer: str = None
    final_answer: str = None

class AgenticRAGSystem:
    """Simplified Agentic RAG System"""
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection("kb_index")
            logger.info("Connected to existing ChromaDB collection")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB collection: {e}")
            raise
    
    def retrieve_kb(self, state: RAGState) -> RAGState:
        """Retriever Node: Fetch top-5 relevant snippets"""
        logger.info(f"Retrieving KB entries for: {state.user_question}")
        
        # Embed the user question
        query_embedding = self.embedding_model.encode([state.user_question])
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5
        )
        
        # Format results
        kb_hits = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                kb_hits.append({
                    'doc_id': results['ids'][0][i],
                    'answer_snippet': doc,
                    'source': results['metadatas'][0][i].get('source', 'unknown'),
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        state.kb_hits = kb_hits
        logger.info(f"Retrieved {len(kb_hits)} KB entries")
        return state
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """LLM Answer Node: Generate initial answer"""
        logger.info("Generating initial answer...")
        
        # Format KB hits for prompt
        kb_context = ""
        for hit in state.kb_hits:
            kb_context += f"[{hit['doc_id']}] {hit['answer_snippet']}\n\n"
        
        prompt = f"""You are a software best-practices assistant.
User Question:
{state.user_question}

Retrieved Snippets:
{kb_context}

Task:
Based on these snippets, write a concise answer to the user's question.
Cite each snippet you use by its doc_id in square brackets (e.g., [KB004]).
Return only the answer text."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful software engineering assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            state.initial_answer = response.choices[0].message.content.strip()
            logger.info("Generated initial answer")
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state.initial_answer = f"Error generating answer: {e}"
        
        return state
    
    def critique_answer(self, state: RAGState) -> RAGState:
        """Self-Critique Node: Check if answer is complete"""
        logger.info("Critiquing initial answer...")
        
        # Format KB hits for prompt
        kb_context = ""
        for hit in state.kb_hits:
            kb_context += f"[{hit['doc_id']}] {hit['answer_snippet']}\n"
        
        prompt = f"""You are a critical QA assistant. The user asked: {state.user_question}

Initial Answer:
{state.initial_answer}

KB Snippets:
{kb_context}

Task:
Determine if the initial answer fully addresses the question using only these snippets.
- If it does, respond exactly: COMPLETE
- If it misses any point or cites missing info, respond: REFINE: <short list of missing topic keywords>

Return exactly one line."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a critical QA assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            state.critique_result = response.choices[0].message.content.strip()
            logger.info(f"Critique result: {state.critique_result}")
            
        except Exception as e:
            logger.error(f"Error in critique: {e}")
            state.critique_result = "COMPLETE"
        
        return state
    
    def refine_answer(self, state: RAGState) -> RAGState:
        """Refinement Node: Improve answer if needed"""
        if not state.critique_result.startswith("REFINE"):
            state.refined_answer = state.initial_answer
            return state
        
        logger.info("Refining answer...")
        
        # Extract missing keywords
        missing_keywords = state.critique_result.replace("REFINE:", "").strip()
        
        # Create new query for additional context
        new_query = f"{state.user_question} and information on {missing_keywords}"
        
        # Get one additional snippet
        query_embedding = self.embedding_model.encode([new_query])
        additional_results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=1
        )
        
        additional_snippet = ""
        if additional_results['documents'] and additional_results['documents'][0]:
            doc_id = additional_results['ids'][0][0]
            doc_text = additional_results['documents'][0][0]
            additional_snippet = f"[{doc_id}] {doc_text}"
        
        prompt = f"""You are a software best-practices assistant refining your answer. The user asked: {state.user_question}

Initial Answer:
{state.initial_answer}

Critique: {state.critique_result}

Additional Snippet:
{additional_snippet}

Task:
Incorporate this snippet into the answer, covering the missing points.
Cite any snippet you use by doc_id in square brackets.
Return only the final refined answer."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful software engineering assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            state.refined_answer = response.choices[0].message.content.strip()
            logger.info("Generated refined answer")
            
        except Exception as e:
            logger.error(f"Error refining answer: {e}")
            state.refined_answer = state.initial_answer
        
        return state
    
    def run_pipeline(self, user_question: str) -> Dict[str, Any]:
        """Run the complete RAG pipeline"""
        logger.info(f"Starting RAG pipeline for: {user_question}")
        
        # Initialize state
        state = RAGState(user_question=user_question)
        
        # Run pipeline steps
        state = self.retrieve_kb(state)
        state = self.generate_answer(state)
        state = self.critique_answer(state)
        state = self.refine_answer(state)
        
        # Determine final answer
        if state.critique_result == "COMPLETE":
            final_answer = state.initial_answer
        else:
            final_answer = state.refined_answer or state.initial_answer
        
        state.final_answer = final_answer
        
        # Return results
        return {
            "query": user_question,
            "kb_hits": [
                {
                    "doc_id": hit["doc_id"],
                    "snippet": hit["answer_snippet"][:100] + "...",
                    "source": hit["source"]
                }
                for hit in state.kb_hits
            ],
            "initial_answer": state.initial_answer,
            "critique_result": state.critique_result,
            "refined_answer": state.refined_answer,
            "final_answer": final_answer
        }

# Test function
def test_rag_system():
    """Test the RAG system with sample queries"""
    rag = AgenticRAGSystem()
    
    test_queries = [
        "What are best practices for caching?",
        "How should I set up CI/CD pipelines?",
        "What are performance tuning tips?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        result = rag.run_pipeline(query)
        
        print(f"Retrieved KB IDs: {[hit['doc_id'] for hit in result['kb_hits']]}")
        print(f"Initial Answer: {result['initial_answer']}")
        print(f"Critique: {result['critique_result']}")
        if result['refined_answer'] and result['refined_answer'] != result['initial_answer']:
            print(f"Refined Answer: {result['refined_answer']}")
        print(f"Final JSON: {json.dumps({'answer': result['final_answer']}, indent=2)}")

if __name__ == "__main__":
    test_rag_system()
#!/usr/bin/env python3
"""
Interactive executor for the Agentic RAG System.
This script provides a command-line interface to interact with the system.
"""
import json
import os
import sys
from typing import Dict, Any
from agentic_rag_simplified import AgenticRAGSystem
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main executor function"""
    print("🤖 Agentic RAG System - Interactive Mode")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        sys.exit(1)
    
    try:
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        rag_system = AgenticRAGSystem()
        print("✅ RAG system initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        sys.exit(1)
    
    # Sample queries for testing
    sample_queries = [
        "What are best practices for caching?",
        "How should I set up CI/CD pipelines?",
        "What are performance tuning tips?",
        "How do I version my APIs?",
        "What should I consider for error handling?"
    ]
    
    print("\n📝 Sample queries you can try:")
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")
    
    print("\n" + "=" * 50)
    print("Enter your questions (type 'quit' to exit, 'test' to run all samples)")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'test':
                print("\n🧪 Running all sample queries...")
                for query in sample_queries:
                    print(f"\n{'='*60}")
                    print(f"🔍 Query: {query}")
                    print('='*60)
                    
                    try:
                        result = rag_system.run_pipeline(query)
                        display_results(result)
                    except Exception as e:
                        print(f"❌ Error processing query: {e}")
                continue
            
            if not user_input:
                print("Please enter a question or 'quit' to exit.")
                continue
            
            print(f"\n🔄 Processing: {user_input}")
            print("-" * 50)
            
            # Run RAG pipeline
            result = rag_system.run_pipeline(user_input)
            
            # Display results
            display_results(result)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception("Unexpected error in main loop")

def display_results(result: Dict[str, Any]):
    """Display RAG pipeline results in a formatted way"""
    
    print("\n📚 Retrieved KB Entries:")
    for hit in result['kb_hits']:
        print(f"  • [{hit['doc_id']}] {hit['snippet']}")
        print(f"    Source: {hit['source']}")
    
    print(f"\n💭 Initial Answer:")
    print(f"  {result['initial_answer']}")
    
    print(f"\n🔍 Critique Result:")
    print(f"  {result['critique_result']}")
    
    if result['refined_answer'] and result['refined_answer'] != result['initial_answer']:
        print(f"\n✨ Refined Answer:")
        print(f"  {result['refined_answer']}")
    
    print(f"\n📄 Final JSON Response:")
    final_json = {"answer": result['final_answer']}
    print(json.dumps(final_json, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
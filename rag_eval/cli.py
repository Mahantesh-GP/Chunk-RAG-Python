"""
Command-line runner for RAG evaluation using LlamaIndex + Azure OpenAI.

This module orchestrates the evaluation of different chunk sizes using:
- Azure OpenAI (gpt-3.5-turbo) for response generation
- Azure OpenAI (gpt-4) for evaluation
- LlamaIndex VectorStoreIndex for document indexing and retrieval
"""

import argparse
import json
import os
import time
from typing import List, Dict, Tuple

import nest_asyncio
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.llms import OpenAI

# Apply nest_asyncio for event loop handling
nest_asyncio.apply()

# Load environment variables
load_dotenv()


class RAGEvaluationRunner:
    """Orchestrates RAG evaluation across different chunk sizes using LlamaIndex."""

    def __init__(self, docs_path: str, queries_path: str, output_path: str):
        """
        Initialize the evaluation runner.

        Args:
            docs_path: Path to documents directory
            queries_path: Path to queries.json file
            output_path: Path to save evaluation_results.json
        """
        self.docs_path = docs_path
        self.queries_path = queries_path
        self.output_path = output_path
        self.results = {}

        # Initialize Azure OpenAI LLMs
        self.llm_turbo = OpenAI(
            model=os.getenv("OPENAI_DEPLOYMENT_NAME_TURBO", "gpt-35-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0.7,
        )

        self.llm_gpt4 = OpenAI(
            model=os.getenv("OPENAI_DEPLOYMENT_NAME_GPT4", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0,
        )

        # Initialize evaluators with GPT-4
        service_context_gpt4 = ServiceContext.from_defaults(llm=self.llm_gpt4)
        self.faithfulness_evaluator = FaithfulnessEvaluator(
            service_context=service_context_gpt4
        )
        self.relevancy_evaluator = RelevancyEvaluator(
            service_context=service_context_gpt4
        )

    def load_documents(self) -> List:
        """Load documents from the specified directory."""
        print(f"\n📚 Loading documents from {self.docs_path}...")
        reader = SimpleDirectoryReader(self.docs_path)
        documents = reader.load_data()
        print(f"✓ Loaded {len(documents)} document(s)")
        return documents

    def load_queries(self) -> List[str]:
        """Load evaluation queries from queries.json."""
        print(f"\n❓ Loading queries from {self.queries_path}...")
        with open(self.queries_path, "r") as f:
            queries = json.load(f)
        print(f"✓ Loaded {len(queries)} query/ies")
        return queries

    def generate_eval_questions(self, documents: List, num_questions: int = 20) -> List[str]:
        """
        Generate evaluation questions from documents using DatasetGenerator.

        Args:
            documents: List of documents
            num_questions: Number of questions to generate

        Returns:
            List of generated questions
        """
        print(f"\n🤖 Generating {num_questions} evaluation questions...")
        data_generator = DatasetGenerator.from_documents(documents)
        eval_questions = data_generator.generate_questions_from_nodes(
            num=num_questions
        )
        print(f"✓ Generated {len(eval_questions)} questions")
        return eval_questions

    def evaluate_chunk_size(
        self, chunk_size: int, documents: List, eval_questions: List[str]
    ) -> Tuple[float, float, float]:
        """
        Evaluate a specific chunk size.

        Args:
            chunk_size: Size of chunks for indexing
            documents: List of documents to index
            eval_questions: List of queries to evaluate

        Returns:
            Tuple of (avg_response_time, avg_faithfulness, avg_relevancy)
        """
        print(f"\n⏱️  Evaluating chunk_size={chunk_size}...")

        total_response_time = 0.0
        total_faithfulness = 0.0
        total_relevancy = 0.0
        num_questions = len(eval_questions)

        # Create service context with specified chunk size
        service_context = ServiceContext.from_defaults(
            llm=self.llm_turbo, chunk_size=chunk_size
        )

        # Build vector index
        print(f"   Building VectorStoreIndex...")
        vector_index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

        # Create query engine
        query_engine = vector_index.as_query_engine()

        # Evaluate each question
        for i, question in enumerate(eval_questions, 1):
            # Measure response time
            start_time = time.time()
            response = query_engine.query(question)
            elapsed_time = time.time() - start_time

            # Evaluate faithfulness (no hallucination)
            try:
                faithfulness_result = self.faithfulness_evaluator.evaluate_response(
                    response=response
                )
                faithfulness = 1.0 if faithfulness_result.passing else 0.0
            except Exception as e:
                print(f"   ⚠️  Faithfulness eval failed for Q{i}: {e}")
                faithfulness = 0.0

            # Evaluate relevancy (query match)
            try:
                relevancy_result = self.relevancy_evaluator.evaluate_response(
                    query=question, response=response
                )
                relevancy = 1.0 if relevancy_result.passing else 0.0
            except Exception as e:
                print(f"   ⚠️  Relevancy eval failed for Q{i}: {e}")
                relevancy = 0.0

            total_response_time += elapsed_time
            total_faithfulness += faithfulness
            total_relevancy += relevancy

            print(
                f"   Q{i}/{num_questions}: time={elapsed_time:.2f}s, "
                f"faith={faithfulness:.0%}, relev={relevancy:.0%}"
            )

        # Calculate averages
        avg_response_time = total_response_time / num_questions
        avg_faithfulness = total_faithfulness / num_questions
        avg_relevancy = total_relevancy / num_questions

        print(
            f"   ✓ Chunk {chunk_size}: "
            f"time={avg_response_time:.2f}s, faith={avg_faithfulness:.2%}, relev={avg_relevancy:.2%}"
        )

        return avg_response_time, avg_faithfulness, avg_relevancy

    def run_evaluation(self, chunk_sizes: List[int]) -> Dict:
        """
        Run evaluation across multiple chunk sizes.

        Args:
            chunk_sizes: List of chunk sizes to evaluate

        Returns:
            Dictionary of results keyed by chunk size
        """
        # Load documents and queries
        documents = self.load_documents()
        eval_questions = self.load_queries()

        # Generate additional questions if needed
        num_gen = int(os.getenv("NUM_EVAL_QUESTIONS", "20"))
        if len(eval_questions) < num_gen:
            additional = self.generate_eval_questions(documents, num=num_gen - len(eval_questions))
            eval_questions.extend(additional)

        results = {}

        # Evaluate each chunk size
        for chunk_size in chunk_sizes:
            avg_time, avg_faith, avg_relev = self.evaluate_chunk_size(
                chunk_size, documents, eval_questions
            )
            results[chunk_size] = {
                "avg_response_time": avg_time,
                "avg_faithfulness": avg_faith,
                "avg_relevancy": avg_relev,
            }

        return results

    def export_results(self, results: Dict):
        """Export results to JSON file."""
        print(f"\n💾 Exporting results to {self.output_path}...")

        # Format results for export
        export_data = {
            "evaluation_framework": "LlamaIndex + Azure OpenAI",
            "evaluation_method": "gpt4",
            "llm_response_model": "gpt-3.5-turbo",
            "evaluation_model": "gpt-4",
            "summary": results,
        }

        with open(self.output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Results saved to {self.output_path}")

        # Print summary table
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 80)
        print(f"{'Chunk Size':<15} {'Avg Time (s)':<20} {'Faithfulness':<20} {'Relevancy':<20}")
        print("-" * 80)
        for chunk_size, metrics in results.items():
            print(
                f"{chunk_size:<15} {metrics['avg_response_time']:<20.2f} "
                f"{metrics['avg_faithfulness']:<20.2%} {metrics['avg_relevancy']:<20.2%}"
            )
        print("=" * 80)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="RAG evaluation using LlamaIndex + Azure OpenAI"
    )
    parser.add_argument(
        "--docs",
        default=os.getenv("DOCS_PATH", "./docs/"),
        help="Path to documents directory",
    )
    parser.add_argument(
        "--queries",
        default=os.getenv("QUERIES_PATH", "./queries.json"),
        help="Path to queries.json file",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("OUTPUT_PATH", "./evaluation_results.json"),
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--chunk-sizes",
        default=os.getenv("CHUNK_SIZES", "128,256,512,1024,2048"),
        help="Comma-separated list of chunk sizes to evaluate",
    )

    args = parser.parse_args()

    # Parse chunk sizes
    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",")]

    # Verify environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and fill in credentials.")
        return

    # Run evaluation
    runner = RAGEvaluationRunner(args.docs, args.queries, args.output)
    results = runner.run_evaluation(chunk_sizes)
    runner.export_results(results)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()

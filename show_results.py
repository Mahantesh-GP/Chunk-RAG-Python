import json

with open('evaluation_results.json') as f:
    data = json.load(f)

print("=== EVALUATION SUMMARY ===\n")
for strat, metrics in data['summary'].items():
    print(f"{strat}:")
    print(f"  Avg Response Time: {metrics['avg_response_time']:.4f}s")
    print(f"  Avg Relevancy: {metrics['avg_relevancy']:.4f}")
    print(f"  Avg Faithfulness: {metrics['avg_faithfulness']:.4f}")
    print(f"  Queries Evaluated: {metrics['count']}\n")

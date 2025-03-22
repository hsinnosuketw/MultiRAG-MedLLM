from graphrag import query_graph

if __name__ == "__main__":
    question = "What is the capital of Japan?"
    query = query_graph(question)
    print(query)
    print(query_graph(query))
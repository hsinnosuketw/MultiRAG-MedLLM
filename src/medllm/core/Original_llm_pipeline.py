def main(question):
    # from langchain_ollama import OllamaLLM
    import os
    import sqlite3
    from langchain.schema import Document
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import GPT4AllEmbeddings
    from langchain.prompts import PromptTemplate

    from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
    from langchain.prompts import PromptTemplate
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_community.embeddings import GPT4AllEmbeddings
    from dotenv import load_dotenv
    from langchain.docstore.document import Document

    # Compute execution time
    import time

    # Helper function
    import extract as extract
    # Importing the prompt
    from prompt import VectorstoreQueryRewriterPrompt,\
    GraphRAGQueryRewriterPrompt, TabularRAGQueryRewriterPrompt,\
    RetrieverFilterPrompt, AnswerGenerationPrompt,\
    VectorstoreQueryRewriterPrompt_W_NER

    # GraphRAG's library
    from neo4j import GraphDatabase

    ##### Useage :  "python llm.py "question" 
    # if len(sys.argv) <= 1:
    #     print("usage: python llm.py question")
    # question = sys.argv[1]
    #####

    # 載入 .env 檔案
    load_dotenv()

    # 取得 API Key
    api_key = os.getenv("NVIDIA_API_KEY")

    # Initialize GPT4AllEmbeddings
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs
    )

###################################
####### VectorstoreRewriter #######
###################################

    drug_tag_list = []
    with open('drugTagList.txt', 'r') as drugTagList: 
        for d_t_tmp in drugTagList: 
            # process the line in some way  
            drug_tag_list.append(d_t_tmp.split("\n")[0]) 

    drug_list =[] 
    # with open('drugList.txt', 'r') as drugList: 
    with open('drugName.txt', 'r') as drugList: 
        for d_tmp in drugList: 
            # process the line in some way  
            drug_list.append(d_tmp.split("\n")[0].lower())  

    model_id_vs = "meta/llama-3.3-70b-instruct"

    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    # LLM
    llm_vs = ChatNVIDIA(model=model_id_vs, temperature=0)
    # from langchain.llms.ollama import Ollama
    # llm_vs = OllamaLLM(model="llama3.1:8b")

    prompt_vs = PromptTemplate(
        # Prompt is in prompt.py
        template=VectorstoreQueryRewriterPrompt,
        input_variables=["drug_list", "drug_tag_list", "question"],
    )
    

    d_list_ext = extract.extract_drug(question)

    if d_list_ext != []:
        prompt_vs = PromptTemplate(
            # Prompt is in prompt.py
            template=VectorstoreQueryRewriterPrompt_W_NER,
            input_variables=["question", "drug_tag_list", "d_list_ext"],
        )
        start = time.time()
        query_rewriter_vs = prompt_vs | llm_vs | StrOutputParser()
        query_vs = query_rewriter_vs.invoke({"d_list_ext":d_list_ext, "drug_tag_list": drug_tag_list,"question": question})
        end = time.time()
        print(f"Execution Time of Vectorstore Query Rewriter: {end - start}")
    else: 
        query_vs = ""
    print(query_vs)
    print(type(query_vs))
    

################################
####### GraphRAGRewriter #######
################################

    model_id_graph = "meta/llama-3.3-70b-instruct"

    # LLM
    llm_graph = ChatNVIDIA(model=model_id_graph, temperature=0)
    # llm_graph = OllamaLLM(model="llama3.1:8b")

    prompt_graph = PromptTemplate(
        # Prompt is in prompt.py
        template=GraphRAGQueryRewriterPrompt,
        input_variables=["question"],
        
    )

    query_rewriter = prompt_graph | llm_graph | StrOutputParser()

    start = time.time()
    graph_query = query_rewriter.invoke({"question": question})
    end = time.time()
    print(f"Execution Time of GraphRAG Query Rewriter: {end - start}") 

    # print(graph_query)
    
##################################
####### TabularRAGRewriter #######
##################################

    model_id_tab = "meta/llama-3.3-70b-instruct"
    llm_tab = ChatNVIDIA(model=model_id_tab, temperature=0)

    prompt_sql = PromptTemplate(
        # Prompt is in prompt.py
        template=TabularRAGQueryRewriterPrompt,
        input_variables=["question"],
    )


    # column_picker = prompt_col | llm | StrOutputParser()
    query_rewriter = prompt_sql | llm_tab | StrOutputParser()

    start = time.time()
    sql_query = query_rewriter.invoke({"question": question})
    end = time.time()
    print(f"Execution Time of TabularRAG Query Rewriter: {end - start}")


    sql_query = sql_query.replace("\n", "").replace("cpilevel", "cpiclevel")
    # print(sql_query)
    
#############################################
####### VectorstoreRewriterRetrieving #######
#############################################

    # Loading Embedding Function
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'}
    GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs,
    )

    # Loading Vectorstore（Persisted data used.）

    persist_directory="./Trial_chroma_langchain"
    collection_name = "Trial_v1"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=GPT4AllEmbeddings(
        model_name=model_name,
        gpt4all_kwargs=gpt4all_kwargs,
        ),
        persist_directory=persist_directory
    )

    '''
    def extract_drug_and_tag(input_string):
        try:
            # Split the input string by the colon
            drug, tag = input_string.split(":", 1)
            return {"drug": drug.strip(), "tag": tag.strip()}
        except ValueError:
            # Handle cases where the input is not in the expected format
            return {"error": "Input must be in the format 'A:B'"}

    q_vs = extract_drug_and_tag(query_vs)
    # Output: {'drug': 'Simvastatin', 'tag': 'Relevance'}'
    '''

    # 現在可直接使用 vectorstore 的檢索功能
    start = time.time()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    

    # Clean the retrieved drug and tags
    # tag = q_vs['tag']
    # drug = q_vs['drug']

    results = []
    print(d_list_ext)
    if d_list_ext != []:
        if len(d_list_ext) > 3:
            for drug in d_list_ext:
                print(vectorstore.similarity_search(query_vs, filter={"drug_name": f"{drug}"}, k=1))
                results.append(vectorstore.similarity_search(query_vs, filter={"drug_name": f"{drug}"}, k=1))
                # if drug.lower() not in drug_list:
                #     results = []
                # else:
                #     query = f"\"{tag}: information\""
        else: 
            for drug in d_list_ext:
                print(vectorstore.similarity_search(query_vs, filter={"drug_name": f"{drug}"}, k=5))
                results.append(vectorstore.similarity_search(query_vs, filter={"drug_name": f"{drug}"}, k=5))
    print(results)
            
    end = time.time()
    print(f"Execution Time of Vectorstore Retriever: {end - start}")

##########################################
####### GraphRAGRewriterRetrieving #######
##########################################
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "password"

    driver = GraphDatabase.driver(uri, auth=(username, password))

    def query_database(query):
        with driver.session() as session:
            result = session.run(query)
            res = []
            for record in result:
                res.append(record)
            return res

    start = time.time()
    if (len(graph_query) != 0):
        try:
            graph_result = query_database(graph_query)
        except:
            graph_result = ""
    driver.close()

    # 處理取回(Retrieved)的藥物交互作用(drug interaction)
    if (len(graph_query) != 0 and len(graph_result) != 0):
        graph_result = graph_result[0]["r"]["description"]
    else:
        graph_result = ""        
    end = time.time()
    print(f"Execution Time of GraphRAG Retriever: {end - start}")

############################################
####### TabularRAGRewriterRetrieving #######
############################################

    start = time.time()
    conn = sqlite3.connect('./drug.db')
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query.split("|")[0].strip())
        sql_result = cursor.fetchall()
    except:        
        sql_result = ""

    cursor.close()
    conn.close()
    end = time.time()
    print(f"Execution Time of TabularRAG Retriever: {end - start}")

    # Retrieval grader
    from langchain.prompts import PromptTemplate
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.output_parsers import JsonOutputParser

    model_id = 'meta/llama-3.1-405b-instruct' #"meta/llama3.3-70b-instruct"

    # LLM
    llm = ChatNVIDIA(model=model_id, temperature=0)
    # llm = OllamaLLM(model="llama3.1:8b")

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    start = time.time()
    retrieval_grader = prompt | llm | JsonOutputParser()

    vs_docs = results

    print(type(vs_docs))
    # docs = retrieval_grader.invoke(question)

    # vectorstore_info = ""

    # for drug in results:
    #     # vectorstore_info += drug.page_content
    #     vectorstore_info += drug
    end = time.time()
    print(f"Execution Time of Grader: {end - start}")

        
    from langchain_core.documents import Document
    print(f"Retrieved result from SQL: {sql_result}")
    docs = vs_docs
    tab_doc = str(sql_result).replace("\'", "")
    if len(sql_query.split("|", 1)) > 1:
        tab_doc += str(sql_query.split("|", 1)[1]).replace("\'", "")
    document_Tab = Document(
        page_content=tab_doc,
        metadata={"source": "TabularRAG"}
    )
    docs.append(document_Tab)
    print(f"Retrieved Tabular Document:{document_Tab}")
    # for curr_string in vs_docs:
    #     document_vs = Document(
    #         page_content=curr_string,
    #         metadata={"source": "vectorstore"}
    #     )
    #     docs.append(document_vs)
    # print(f"Retrieved Vectorstore Document:{document_vs}")
    document_graph = Document(
        page_content=graph_result,
        metadata={"source": "graphRAG"}
    )
    print(f"Retrieved Graph Document:{document_graph}")
    docs.append(document_graph)
    # docsdocument_Tab, document_vs, document_graph]

    # print(retrieval_grader.invoke({"question": question, "document": docs}))
    docs = retrieval_grader.invoke({"question": question, "document": docs})
    
    ### Retrieval Filter

    from langchain.prompts import PromptTemplate
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_core.output_parsers import JsonOutputParser

    model_id = 'meta/llama-3.1-405b-instruct' #"meta/llama3.3-70b-instruct"

    # LLM
    llm = ChatNVIDIA(model=model_id, temperature=0)
    # llm = OllamaLLM(model="llama3.1:8b")

    prompt = PromptTemplate(
        template=RetrieverFilterPrompt,
        input_variables=["question", "documents"],
    )


    retrieval_filter = prompt | llm | JsonOutputParser()
    documents = docs

    start = time.time()

    filtered_retrieval = retrieval_filter.invoke({"question": question, "documents": documents})
    # print(filtered_retrieval)
    
    end = time.time()
    print(f"Execution Time of LLM for Filtering Retrieved information: {end - start}")


    temp = list()
    print(f"過濾後的資料:{filtered_retrieval}")
    for f in filtered_retrieval['filtered docs']:
        t = Document(page_content=f["page_content"])
        temp.append(t)

    ### Nvidia ReRanker
    from langchain_nvidia_ai_endpoints import NVIDIARerank
    # ranker = NVIDIARerank(model= "nvidia/llama-3.2-nv-rerankqa-1b-v1", truncate="END") #model="nvidia/nv-rerankqa-mistral-4b-v3"
    #api_key="$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"
    ranker = NVIDIARerank(
        model="nv-rerank-qa-mistral-4b:1", 
        api_key=os.getenv("NVIDIA_API_KEY"),
    )

    all_docs = temp # filtered vectorstore + graph + sql

    print(f"Filtered documents{all_docs}")

    ranker.top_n = 5
    docs = ranker.compress_documents(query=question, documents=all_docs)
    # docs ### it should still contain the tags for vec, kg and sql
    model_id = "meta/llama-3.3-70b-instruct"

    # LLM
    llm = ChatNVIDIA(model=model_id, temperature=0)
    # llm = OllamaLLM(model="llama3.1:8b")

    prompt = PromptTemplate(
        template=AnswerGenerationPrompt,
        input_variables=["question", "context"],
    )

    rag_chain = prompt | llm | JsonOutputParser()

    start = time.time()
    generation = rag_chain.invoke({"context": docs, "question": question})
    final_answer = generation['answer']

    end = time.time()
    # print(f"Execution Time of Final Answer Generation: {end - start}")

    return final_answer

if __name__ == "__main__":

    answers = []
    # with open("../publicMedicineHygieneEducation/Drug_Problem.txt", "r") as drug_questions:
    #     for question in drug_questions:
    #         print(question)
    #         answers.append(main(question.strip()))
    # with open("../publicMedicineHygieneEducation/Drug_Answer.txt", "w") as drug_answers:
    #     for answer in answers:
    #         drug_answers.append(answer + "\n")
    # print(answers)
    ## single test
    question = 'Opdivo vs Opdivo Qvantig: What is the Difference?'
    print(main(question.strip()))
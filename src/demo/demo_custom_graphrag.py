import mlx.core as mx
from mlx_lm import load, generate
import networkx as nx
import json
import re

# 1. Setup
model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
adapter_path = "adapters_solverx_sft_hcx"

print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

import mlx.core as mx
from mlx_lm import load, generate
import networkx as nx
import json
import re
import sqlite3

# 1. Setup
model_path = "models/HyperCLOVAX-SEED-Think-32B-Text-8bit"
adapter_path = "adapters_solverx_sft_hcx"

print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path)

# 2. Database Setup (Mocking a real DB with Multiple Tables)
def setup_database():
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Table 1: Companies
    cursor.execute('''
        CREATE TABLE companies (
            id INTEGER PRIMARY KEY,
            name TEXT,
            location TEXT,
            industry TEXT
        )
    ''')
    
    # Table 2: Products (Linked to Company)
    cursor.execute('''
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            company_id INTEGER,
            description TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
    ''')

    # Table 3: Technologies (Linked to Product)
    cursor.execute('''
        CREATE TABLE technologies (
            id INTEGER PRIMARY KEY,
            name TEXT,
            product_id INTEGER,
            function TEXT,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )
    ''')

    # Insert Data
    cursor.execute("INSERT INTO companies VALUES (1, 'SolverX', 'Seoul', 'AI CAE')")
    cursor.execute("INSERT INTO products VALUES (1, 'SolverX Fusion', 1, '구조 해석과 열 해석을 동시에 예측하는 멀티피직스 모델')")
    cursor.execute("INSERT INTO technologies VALUES (1, 'Physics Loss', 1, '물리 법칙(보존 법칙)을 손실 함수에 포함')")
    cursor.execute("INSERT INTO technologies VALUES (2, 'Reliability Score', 1, '예측 불확실성 감지 및 기존 솔버 호출')")
    
    conn.commit()
    return conn

# 3. Fetch Data from Multiple Tables (JOINs) with Source Metadata
def fetch_knowledge_from_db(conn):
    cursor = conn.cursor()
    knowledge_list = []

    # Query 1: Company Info
    cursor.execute("SELECT id, name, location, industry FROM companies")
    for row in cursor.fetchall():
        text = f"{row[1]}은(는) {row[2]}에 위치한 {row[3]} 기업이다."
        # Tag the source: "table_name:id"
        knowledge_list.append({'text': text, 'source': f'companies:{row[0]}'})

    # Query 2: Product Info (JOIN Company)
    cursor.execute('''
        SELECT p.id, p.name, c.name, p.description 
        FROM products p 
        JOIN companies c ON p.company_id = c.id
    ''')
    for row in cursor.fetchall():
        text = f"{row[1]}은(는) {row[2]}에서 개발한 제품으로, {row[3]}이다."
        knowledge_list.append({'text': text, 'source': f'products:{row[0]}'})

    # Query 3: Technology Info (JOIN Product)
    cursor.execute('''
        SELECT t.id, t.name, p.name, t.function 
        FROM technologies t 
        JOIN products p ON t.product_id = p.id
    ''')
    for row in cursor.fetchall():
        text = f"{row[1]}은(는) {row[2]}에 적용된 기술로, {row[3]}하는 기능을 한다."
        knowledge_list.append({'text': text, 'source': f'technologies:{row[0]}'})

    return knowledge_list

# 4. Knowledge Extraction (LLM) with Ontology Schema
def extract_triples(text):
    # Define Ontology Schema in Prompt
    schema_guide = """
    [Allowed Relations]
    - is_a (정의)
    - has_feature (기능/특징)
    - located_at (위치)
    - developed_by (개발사)
    - outperforms (성능 우위)
    - operates_in (사업 분야/진출)
    - is_similar_to (유사함/비슷함)
    """
    
    prompt = f"""<|im_start|>user
다음 문장에서 지식 그래프를 구성할 수 있는 [Subject, Relation, Object] 형태의 트리플을 추출해줘.
아래 정의된 [Allowed Relations]만 사용해서 JSON 리스트로 출력해.

{schema_guide}

문장: "{text}"
출력 예시: [["SolverX", "located_at", "Seoul"]]
<|im_end|>
<|im_start|>assistant
"""
    response = generate(model, tokenizer, prompt=prompt, max_tokens=150, verbose=False)
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return []

# 5. Build Graph with Source Tracking
conn = setup_database()
knowledge_items = fetch_knowledge_from_db(conn)
G = nx.DiGraph()

print("\n[Building Knowledge Graph from Database...]")
for item in knowledge_items:
    text = item['text']
    source_id = item['source']
    
    triples = extract_triples(text)
    print(f"Source: {source_id} | Text: {text[:20]}...\n -> Triples: {triples}")
    
    if triples:
        for subj, pred, obj in triples:
            # Store 'source' metadata on the edge
            G.add_edge(subj, obj, relation=pred, source=source_id)

print(f"\nGraph Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 5.1. Ingest Unstructured Text (Natural Language)
def add_text_to_graph(text):
    print(f"\n[Ingesting Natural Language]: \"{text}\"")
    triples = extract_triples(text)
    print(f" -> Extracted Triples: {triples}")
    if triples:
        for subj, pred, obj in triples:
            G.add_edge(subj, obj, relation=pred, source="manual_input")

# Example: Adding external knowledge
raw_text = "경쟁 제품인 Ansys Discovery는 미국에 위치한 Ansys가 개발했다. SolverX Fusion은 Ansys Discovery보다 해석 정확도가 높다."
add_text_to_graph(raw_text)

# Example 2: Adding new business domain info (User Request)
new_business_text = "네이버는 식품사업에도 뛰어 들었다."
add_text_to_graph(new_business_text)

# 5.2. Deletion & Sync Logic
def delete_record_and_sync(conn, table, record_id):
    print(f"\n[Deleting Record] Table: {table}, ID: {record_id}")
    
    # 1. Delete from DB
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
    conn.commit()
    
    # 2. Sync with Graph (Remove Edges with matching source)
    target_source = f"{table}:{record_id}"
    edges_to_remove = []
    
    # Find edges to remove
    for u, v, data in G.edges(data=True):
        if data.get('source') == target_source:
            edges_to_remove.append((u, v, data['relation']))
            
    # Remove them
    for u, v, rel in edges_to_remove:
        G.remove_edge(u, v)
        print(f" -> [Graph Sync] Removed Edge: {u} --[{rel}]--> {v}")
        
    print(f" -> Total {len(edges_to_remove)} edges removed from Graph.")
    print(f" -> Updated Graph Stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Simulate Deletion: Remove 'Physics Loss' technology (ID: 1)
delete_record_and_sync(conn, 'technologies', 1)

# 6. Graph Retrieval & Reasoning
def graph_rag_query(question):
    # Simple Keyword Search to find entry nodes
    relevant_nodes = [n for n in G.nodes if n in question or n in "SolverX Fusion" or n in "네이버"] 
    
    context = []
    for node in relevant_nodes:
        if node in G:
            neighbors = G[node]
            for neighbor, attr in neighbors.items():
                relation = attr['relation']
                context.append(f"{node} --[{relation}]--> {neighbor}")
            
            # [New] Also look for incoming edges (Reverse lookup)
            for predecessor in G.predecessors(node):
                attr = G[predecessor][node]
                relation = attr['relation']
                context.append(f"{predecessor} --[{relation}]--> {node}")

    context_str = "\n".join(list(set(context)))
    
    final_prompt = f"""<|im_start|>user
다음 지식 그래프(Knowledge Graph) 정보를 바탕으로 질문에 답해줘.

[지식 그래프 정보]
{context_str}

질문: {question}
<|im_end|>
<|im_start|>assistant
"""
    print(f"\n[Context Retrieved]\n{context_str}")
    print(f"\n[Generating Answer for: '{question}']")
    response = generate(model, tokenizer, prompt=final_prompt, max_tokens=200, verbose=True)
    return response

# 7. Run Query
graph_rag_query("네이버가 진출한 새로운 사업 분야는 무엇인가요?")
def graph_rag_query(question):
    # Simple Keyword Search to find entry nodes
    relevant_nodes = [n for n in G.nodes if n in question or n in "SolverX Fusion"] 
    
    context = []
    for node in relevant_nodes:
        if node in G:
            neighbors = G[node]
            for neighbor, attr in neighbors.items():
                relation = attr['relation']
                context.append(f"{node} --[{relation}]--> {neighbor}")
            
            # [New] Also look for incoming edges (Reverse lookup)
            # e.g. If we search for "Ansys Discovery", we want to know who outperforms it.
            for predecessor in G.predecessors(node):
                attr = G[predecessor][node]
                relation = attr['relation']
                context.append(f"{predecessor} --[{relation}]--> {node}")

    context_str = "\n".join(list(set(context)))
    
    final_prompt = f"""<|im_start|>user
다음 지식 그래프(Knowledge Graph) 정보를 바탕으로 질문에 답해줘.

[지식 그래프 정보]
{context_str}

질문: {question}
<|im_end|>
<|im_start|>assistant
"""
    print(f"\n[Context Retrieved]\n{context_str}")
    print(f"\n[Generating Answer for: '{question}']")
    response = generate(model, tokenizer, prompt=final_prompt, max_tokens=200, verbose=True)
    return response

# 7. Run Query
graph_rag_query("SolverX Fusion과 Ansys Discovery의 관계를 설명해줘.")

# 8. Visualization
def visualize_graph(G, filename="knowledge_graph.png"):
    try:
        import matplotlib.pyplot as plt
        import platform
        from matplotlib import rc
        
        # Set Korean Font
        if platform.system() == 'Darwin':
            rc('font', family='AppleGothic')
        elif platform.system() == 'Windows':
            rc('font', family='Malgun Gothic')
            
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(G, k=1.5, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_family="AppleGothic" if platform.system() == 'Darwin' else "sans-serif", font_size=10)
        
        nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=20, edge_color="gray", alpha=0.6)
        
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_family="AppleGothic" if platform.system() == 'Darwin' else "sans-serif")
        
        plt.title("SolverX Knowledge Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"\\n[Visualization] Knowledge Graph saved to '{filename}'")
        
    except ImportError as e:
        print(f"\\n[Visualization] Skipped: {e}")
    except Exception as e:
        print(f"\\n[Visualization] Error: {e}")

visualize_graph(G)

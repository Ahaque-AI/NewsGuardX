import json
import os
import asyncio
from tavily import TavilyClient
import requests
from crewai.tools import tool
from fastembed import TextEmbedding
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def compute_cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def format_response(response: dict) -> str:
    lines = []
    # Add the main query, answer, and response time
    lines.append(f"Query: {response.get('query')}\n")
    lines.append("Answer:")
    lines.append(response.get("answer", "No answer provided") + "\n")
    lines.append(f"Response Time: {response.get('response_time', 'N/A')} seconds\n")

    # Format results
    results = response.get("results", [])
    if results:
        lines.append("Results:")
        for idx, result in enumerate(results, start=1):
            lines.append(f"Result {idx}:")
            lines.append(f"  Title: {result.get('title', 'N/A')}")
            lines.append(f"  URL: {result.get('url', 'N/A')}")
            lines.append(f"  Published Date: {result.get('published_date', 'N/A')}")
            lines.append(f"  Score: {result.get('score', 'N/A')}")
            content = result.get("content", "")
            if content:
                lines.append("  Content:")
                lines.append("    " + "\n    ".join(content.splitlines()))
            lines.append("")  # Blank line between results
    else:
        lines.append("No results found.")
    
    return "\n".join(lines)

@tool("Search the internet")
def search_internet(headline: str) -> str:
    """Useful to search the internet about a given topic and return relevant results"""
    print("Searching the internet...")
    top_result_to_return = 5
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": headline, "num": top_result_to_return, "tbm": "nws"})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    # check if there is an organic key
    if 'organic' not in response.json():
        return "Sorry, I couldn't find anything about that, there could be an error with your serper api key."
    else:
        results = response.json()['organic']
        string = []
        print("Results:", results[:top_result_to_return])
        for result in results[:top_result_to_return]:
            try:
                # Attempt to extract the date
                date = result.get('date', 'Date not available')
                string.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['link']}",
                    f"Date: {date}",
                    f"Snippet: {result['snippet']}",
                    "\n-----------------"
                ]))
            except KeyError:
                continue
        return '\n'.join(string)

@tool("Search the graphDB and retrieve similar events using hybrid search")
def hybrid_search(headline: str) -> str:
    """
    Retrieves similar events from the AuraDB online database using a hybrid search approach.
    
    Args:
        headline (str): The headline or query text to search for similar events.
    
    Returns:
        str: A formatted string containing the similar events and their details.
    """
    # Initialize FastEmbed model
    embedding_model = TextEmbedding()
    
    # Load AuraDB online credentials from environment variables
    NEO4J_URI = os.getenv("NEO4J_URI")         # e.g., "neo4j+s://<your-aura-instance>"
    NEO4J_USER = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    # Initialize graph instance using AuraDB connection details
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

    def retrieve_similar_events(query_text, graph, embedding_model, top_k=20, embedding_weight=0.6, structure_weight=0.4, similarity_threshold=0.5):
        # Generate embedding for the query text
        query_embedding = list(embedding_model.embed([query_text]))[0]
        
        # Query all event nodes with their embeddings and structural details
        cypher_query = """
        MATCH (e:Event)
        WHERE e.embedding IS NOT NULL AND e.text IS NOT NULL
        
        OPTIONAL MATCH (e)-[:RESULTS_IN]->(effect:Effect)
        WITH e, COUNT(effect) AS effect_count
        
        OPTIONAL MATCH (e)<-[:CAUSES]-(cause:Cause)
        WITH e, effect_count, COUNT(cause) AS cause_count
        
        OPTIONAL MATCH (e)-[:HAS_TRIGGER]->(trigger:Trigger)
        WITH e, effect_count, cause_count, COUNT(trigger) AS trigger_count
        
        WITH e, effect_count, cause_count, trigger_count,
            CASE 
                WHEN effect_count + cause_count + trigger_count = 0 
                THEN 0.0 
                ELSE 1.0 
            END AS structural_score
        
        CALL {
            WITH e
            OPTIONAL MATCH (e)-[:RESULTS_IN]->(effect:Effect)
            RETURN COLLECT(effect.text) AS effect_texts
        }
        CALL {
            WITH e
            OPTIONAL MATCH (e)<-[:CAUSES]-(cause:Cause)
            RETURN COLLECT(cause.text) AS cause_texts
        }
        CALL {
            WITH e
            OPTIONAL MATCH (e)-[:HAS_TRIGGER]->(trigger:Trigger)
            RETURN COLLECT(trigger.text) AS trigger_texts
        }
        
        RETURN e.text AS text,
            e.id AS event_id,
            e.embedding AS embedding,
            structural_score,
            effect_count,
            cause_count,
            trigger_count,
            effect_texts,
            cause_texts,
            trigger_texts
        """
        try:
            results = graph.query(cypher_query)
            
            # Compute cosine similarity and hybrid score for each event
            for record in results:
                event_embedding = record['embedding']
                cosine_sim = compute_cosine_similarity(query_embedding, event_embedding)
                record['embedding_similarity'] = cosine_sim
                record['hybrid_score'] = embedding_weight * cosine_sim + structure_weight * record['structural_score']
            
            # Filter and sort results based on hybrid score
            filtered = [r for r in results if r['hybrid_score'] >= similarity_threshold]
            filtered.sort(key=lambda r: r['hybrid_score'], reverse=True)
            return filtered[:top_k]
        
        except Exception as e:
            print(f"Error during query execution: {str(e)}")
            return []

    similar_events = retrieve_similar_events(
        headline,
        graph,
        embedding_model,
        top_k=20,
        embedding_weight=0.6,
        structure_weight=0.4,
        similarity_threshold=0.5
    )

    expected_output = ""
    for i, event in enumerate(similar_events, 1):
        expected_output += f"\n{i}. Event: {event['text']}"
        expected_output += f"\n   ID: {event.get('event_id', 'N/A')}"
        expected_output += f"\n   Hybrid Score: {round(float(event['hybrid_score']), 3)}"
        expected_output += f"\n   Embedding Similarity: {round(float(event['embedding_similarity']), 3)}"
        expected_output += f"\n   Structural Score: {round(float(event['structural_score']), 3)}"
        
        expected_output += "\n   Connected Nodes:"
        expected_output += "\n   Causes:"
        for cause in event.get('cause_texts', []):
            if cause:
                expected_output += f"\n   - {cause}"
        expected_output += "\n   Effects:"
        for effect in event.get('effect_texts', []):
            if effect:
                expected_output += f"\n   - {effect}"
        expected_output += "\n   Triggers:"
        for trigger in event.get('trigger_texts', []):
            if trigger:
                expected_output += f"\n   - {trigger}"
    
    return expected_output

@tool("Search the internet through tavilyAPI")
def search_tavily(headline: str) -> str:
    """Useful to search the internet through tavily API"""
    client = TavilyClient(os.environ['TAVILY_API_KEY'])
    response = client.search(
        query=headline,
        topic="news",
        search_depth="advanced",
        max_results=10,
        include_answer="advanced"
    )

    print("Response:", format_response(response))

    return format_response(response)

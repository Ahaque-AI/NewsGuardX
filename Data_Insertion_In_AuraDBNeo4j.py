import dotenv
import os
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import re
import hashlib
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from fastembed import TextEmbedding

# Load environment variables
dotenv.load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

print("URI:", URI)
print("AUTH:", AUTH)

# Initialize Neo4j driver and verify connectivity
driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()
print("Connection established.")

class CausalGraph:
    def __init__(self, driver, embedding_model):
        self.driver = driver
        self.embedding_model = embedding_model
        self.debug_info = {
            'total_triggers_found': 0,
            'triggers_per_event': []
        }
        
    def run_query(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def clear_database(self):
        queries = [
            "MATCH (n) DETACH DELETE n",
            "CALL apoc.schema.assert({}, {})"
        ]
        for query in queries:
            try:
                self.run_query(query)
            except Exception as e:
                print(f"Warning during cleanup: {e}")

    def create_indexes(self):
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cause) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (eff:Effect) REQUIRE eff.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Trigger) REQUIRE t.id IS UNIQUE"
        ]
        for query in constraints:
            try:
                self.run_query(query)
            except Exception as e:
                print(f"Warning during index creation: {e}")

    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        if pd.isna(text) or text is None:
            return ""
        cleaned = str(text).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.replace('"', "'").replace('\\', '')
        return cleaned if preserve_case else cleaned.lower()

    def generate_hash(self, text: str, event_id: str = "") -> str:
        text_to_hash = f"{text}_{event_id}" if event_id else text
        return hashlib.md5(text_to_hash.encode()).hexdigest()

    def extract_elements(self, tagged_text: str):
        if not isinstance(tagged_text, str) or tagged_text == 'NoTag':
            return None
        try:
            patterns = {
                'causes': r'<cause>((?:(?!</cause>).)*)</cause>',
                'effects': r'<effect>((?:(?!</effect>).)*)</effect>',
                'triggers': r'<trigger>((?:(?!</trigger>).)*)</trigger>'
            }
            elements = {}
            for key, pattern in patterns.items():
                matches = re.findall(pattern, tagged_text, re.DOTALL | re.IGNORECASE)
                if key == 'triggers':
                    elements[key] = [m.strip() for m in matches if m.strip()]
                    self.debug_info['triggers_per_event'].append(len(elements[key]))
                    self.debug_info['total_triggers_found'] += len(elements[key])
                else:
                    cleaned_matches = [m.strip() for m in matches if m.strip()]
                    elements[key] = list(dict.fromkeys(cleaned_matches))
            return elements if any(elements.values()) else None
        except Exception as e:
            print(f"Error extracting elements: {e}")
            return None

    def generate_embedding(self, text: str):
        try:
            embedding_generator = self.embedding_model.embed([text])
            return next(embedding_generator)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def create_event_graph(self, text: str, tagged_text: str, event_id: str):
        elements = self.extract_elements(tagged_text)
        if not elements:
            return

        cleaned_text = self.clean_text(text)
        embedding = self.generate_embedding(cleaned_text)

        event_query = """
        MERGE (e:Event {id: $id})
        SET e.text = $text,
            e.tagged_text = $tagged_text,
            e.embedding = $embedding,
            e.created_at = datetime()
        """
        self.run_query(event_query, params={
            'id': event_id,
            'text': cleaned_text,
            'tagged_text': tagged_text,
            'embedding': embedding
        })

        for cause in elements.get('causes', []):
            cause_id = self.generate_hash(self.clean_text(cause))
            cause_query = """
            MATCH (e:Event {id: $event_id})
            MERGE (c:Cause {id: $cause_id})
            SET c.text = $cause_text
            MERGE (c)-[r:CAUSES]->(e)
            SET r.created_at = datetime()
            """
            self.run_query(cause_query, params={
                'event_id': event_id,
                'cause_id': cause_id,
                'cause_text': cause
            })

        for effect in elements.get('effects', []):
            effect_id = self.generate_hash(self.clean_text(effect))
            effect_query = """
            MATCH (e:Event {id: $event_id})
            MERGE (eff:Effect {id: $effect_id})
            SET eff.text = $effect_text
            MERGE (e)-[r:RESULTS_IN]->(eff)
            SET r.created_at = datetime()
            """
            self.run_query(effect_query, params={
                'event_id': event_id,
                'effect_id': effect_id,
                'effect_text': effect
            })

        for trigger in elements.get('triggers', []):
            trigger_id = self.generate_hash(self.clean_text(trigger), event_id)
            trigger_query = """
            MATCH (e:Event {id: $event_id})
            MERGE (t:Trigger {id: $trigger_id})
            SET t.text = $trigger_text,
                t.event_id = $event_id
            MERGE (e)-[r:HAS_TRIGGER]->(t)
            SET r.created_at = datetime()
            """
            self.run_query(trigger_query, params={
                'event_id': event_id,
                'trigger_id': trigger_id,
                'trigger_text': trigger
            })

    def analyze_dataset(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.replace({np.nan: None})
        tagged_rows = df[df['tagged_sentence'] != 'NoTag']

        total_stats = {'causes': 0, 'effects': 0, 'triggers': 0}
        unique_elements = {'causes': set(), 'effects': set(), 'triggers': set()}

        print("Analyzing dataset...")
        for _, row in tagged_rows.iterrows():
            elements = self.extract_elements(str(row['tagged_sentence']))
            if elements:
                for key in ['causes', 'effects', 'triggers']:
                    total_stats[key] += len(elements.get(key, []))
                    unique_elements[key].update(elements.get(key, []))

        print("\nDataset Analysis:")
        print(f"Total tagged sentences: {len(tagged_rows)}")
        print("\nTotal elements found:")
        for key, value in total_stats.items():
            print(f"Total {key}: {value}")
        print("\nUnique elements:")
        for key, value in unique_elements.items():
            print(f"Unique {key}: {len(value)}")

        print("\nTrigger Statistics:")
        print(f"Total triggers found: {self.debug_info['total_triggers_found']}")
        trigger_counts = Counter(self.debug_info['triggers_per_event'])
        print("Events by trigger count:")
        for count, freq in sorted(trigger_counts.items()):
            print(f"{count} trigger(s): {freq} events")

    def load_dataset(self, csv_path: str, clear_existing: bool = True):
        if clear_existing:
            self.clear_database()
        self.create_indexes()
        self.analyze_dataset(csv_path)

        df = pd.read_csv(csv_path)
        df = df.replace({np.nan: None})
        tagged_rows = df[df['tagged_sentence'] != 'NoTag']

        print("\nLoading data into graph...")
        for idx, row in tqdm(tagged_rows.iterrows(), total=len(tagged_rows)):
            try:
                self.create_event_graph(
                    text=str(row['text']),
                    tagged_text=str(row['tagged_sentence']),
                    event_id=f"event_{idx}"
                )
            except Exception as e:
                print(f"\nError processing row {idx}: {e}")

        stats = self.get_graph_statistics()
        print("\nFinal Graph Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

    def get_graph_statistics(self):
        node_stats_query = """
        MATCH (n)
        RETURN {
            events: count(CASE WHEN n:Event THEN 1 END),
            causes: count(CASE WHEN n:Cause THEN 1 END),
            effects: count(CASE WHEN n:Effect THEN 1 END),
            triggers: count(CASE WHEN n:Trigger THEN 1 END)
        } as stats
        """
        rel_stats_query = """
        MATCH ()-[r]->()
        RETURN count(r) as relationships
        """
        node_results = self.run_query(node_stats_query)
        rel_results = self.run_query(rel_stats_query)
        stats = node_results[0]['stats']
        stats['relationships'] = rel_results[0]['relationships']
        return stats

def analyze_dataset_distribution(csv_path: str):
    df = pd.read_csv(csv_path)
    total_sentences = len(df)
    tagged_sentences = df[df['tagged_sentence'] != 'NoTag'].shape[0]

    cause_count = df['tagged_sentence'].str.count('<cause>').sum()
    effect_count = df['tagged_sentence'].str.count('<effect>').sum()
    trigger_count = df['tagged_sentence'].str.count('<trigger>').sum()

    print("Dataset Analysis:")
    print(f"Total sentences: {total_sentences}")
    print(f"Tagged sentences: {tagged_sentences}")
    print(f"Unique causes: {cause_count}")
    print(f"Unique effects: {effect_count}")
    print(f"Unique triggers: {trigger_count}")

# Initialize embedding model and causal graph
embedding_model = TextEmbedding()
causal_graph = CausalGraph(driver, embedding_model)

# Load dataset into graph
csv_path = 'Causal_dataset.csv'
causal_graph.load_dataset(csv_path)

# Analyze dataset distribution separately
analyze_dataset_distribution(csv_path)

# Close the driver when done
driver.close()

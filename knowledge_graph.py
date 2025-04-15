import networkx as nx
from typing import List, Dict, Any
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class KnowledgeGraph:
    def __init__(self):
        self.reset_graph()
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorizer = TfidfVectorizer()
        self.texts = []
        self.vectors = None
        
        # Colors for different entity types
        self.entity_colors = {
            'PERSON': '#FF9999',      # Light red
            'ORG': '#99FF99',         # Light green
            'GPE': '#9999FF',         # Light blue
            'PRODUCT': '#FFFF99',     # Light yellow
            'DATE': '#FF99FF',        # Light pink
            'MONEY': '#99FFFF',       # Light cyan
            'PERCENT': '#FFCC99',     # Light orange
            'TIME': '#CC99FF',        # Light purple
            'DEFAULT': '#CCCCCC'      # Gray
        }
        
        # Relationship types and their descriptions
        self.relation_types = {
            'nsubj': 'Subject of the sentence',
            'dobj': 'Direct object of the verb',
            'pobj': 'Object of a preposition',
            'attr': 'Attribute of the subject',
            'compound': 'Compound relationship',
            'amod': 'Adjectival modifier',
            'nmod': 'Nominal modifier',
            'poss': 'Possessive relationship',
            'appos': 'Appositional modifier',
            'conj': 'Conjunction relationship'
        }
    
    def reset_graph(self):
        """Reset the graph and start fresh"""
        self.graph = nx.DiGraph()
        self.texts = []
        self.vectors = None
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]):
        """Add an entity to the graph"""
        color = self.entity_colors.get(entity_type, self.entity_colors['DEFAULT'])
        self.graph.add_node(entity_id, 
                           type=entity_type,
                           text=properties["text"],
                           color=color,
                           label=f"{properties['text']}\n({entity_type})",
                           size=1500)  # Increased node size
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str, properties: Dict[str, Any] = None):
        """Add a relation to the graph"""
        if properties is None:
            properties = {}
        description = self.relation_types.get(relation_type, relation_type)
        self.graph.add_edge(source_id, target_id, 
                           type=relation_type,
                           label=f"{relation_type}\n({description})",
                           weight=1.0)
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using spaCy"""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities
    
    def process_text(self, text: str):
        """Process text and extract entities and relations"""
        # Clear previous graph
        self.reset_graph()
        self.graph = nx.DiGraph()
        self.texts = []
        self.vectors = None
        
        chunks = self.text_splitter.split_text(text)
        for chunk in chunks:
            doc = self.nlp(chunk)
            entities = self.extract_entities(chunk)
            
            # Add entities
            for entity in entities:
                entity_id = f"entity_{len(self.graph.nodes)}"
                self.add_entity(entity_id, entity["label"], {"text": entity["text"]})
                self.texts.append(entity["text"])
            
            # Extract relations
            for token in doc:
                if token.dep_ in self.relation_types and token.ent_type_:
                    head = token.head
                    if head.ent_type_:
                        source_id = f"entity_{list(self.graph.nodes).index(f'entity_{len(self.graph.nodes)-1}')}"
                        target_id = f"entity_{list(self.graph.nodes).index(f'entity_{len(self.graph.nodes)-2}')}"
                        self.add_relation(source_id, target_id, token.dep_)
        
        # Convert texts to vectors
        if self.texts:
            self.vectors = self.vectorizer.fit_transform(self.texts)
    
    def query_graph(self, query: str) -> List[Dict[str, Any]]:
        """Search the graph based on query"""
        results = []
        
        if self.vectors is None or len(self.texts) == 0:
            return results
        
        # Convert query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Sort results by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        for idx in sorted_indices:
            if similarities[idx] > 0.1:  # Similarity threshold
                node_id = f"entity_{idx}"
                if node_id in self.graph.nodes:
                    node_data = self.graph.nodes[node_id]
                    results.append({
                        "id": node_id,
                        "type": node_data["type"],
                        "text": node_data["text"],
                        "similarity": float(similarities[idx])
                    })
        
        return results
    
    def get_entity_statistics(self):
        """Entity type distribution analysis"""
        if not self.graph.nodes:
            return None
            
        entity_types = [self.graph.nodes[node]['type'] for node in self.graph.nodes()]
        type_counts = Counter(entity_types)
        
        # Create bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(type_counts.keys()), y=list(type_counts.values()))
        plt.title('Entity Type Distribution', fontsize=16, pad=20)
        plt.xlabel('Entity Type', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        return plt.gcf()
    
    def get_relation_statistics(self):
        """Relation type distribution analysis"""
        if not self.graph.edges:
            return None
            
        relation_types = [self.graph.edges[edge]['type'] for edge in self.graph.edges()]
        relation_counts = Counter(relation_types)
        
        # Create pie chart
        plt.figure(figsize=(10, 10))
        plt.pie(relation_counts.values(), 
                labels=[f"{k}\n({self.relation_types.get(k, '')})" for k in relation_counts.keys()],
                autopct='%1.1f%%',
                textprops={'fontsize': 12})
        plt.title('Relation Type Distribution', fontsize=16, pad=20)
        return plt.gcf()
    
    def visualize(self):
        """Visualize the graph (for Streamlit)"""
        if not self.graph.nodes:
            return None, None
            
        # Set graph size
        plt.figure(figsize=(20, 20))
        
        # Calculate node positions
        pos = nx.spring_layout(self.graph, 
                             k=3.0,  # Increased node spacing
                             iterations=300,  # More iterations
                             seed=42,
                             scale=4.0)  # Larger scale
        
        # Draw nodes
        node_colors = [self.graph.nodes[node]['color'] for node in self.graph.nodes()]
        node_sizes = [self.graph.nodes[node]['size'] * 2 for node in self.graph.nodes()]  # Larger nodes
        node_labels = {node: self.graph.nodes[node]['label'] for node in self.graph.nodes()}
        
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.9,
                             edgecolors='black',
                             linewidths=3,
                             node_shape='o')
        
        # Draw edges
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        edge_weights = [self.graph.edges[edge]['weight'] * 3 for edge in self.graph.edges()]  # Thicker edges
        
        nx.draw_networkx_edges(self.graph, pos, 
                             edge_color='#666666',
                             arrows=True,
                             arrowsize=30,  # Larger arrows
                             width=edge_weights,
                             alpha=0.8,
                             connectionstyle='arc3,rad=0.2',
                             arrowstyle='->,head_length=1.0,head_width=1.0')
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, 
                              labels=node_labels,
                              font_size=12,  # Larger font
                              font_weight='bold',
                              font_family='Arial',
                              bbox=dict(facecolor='white', 
                                      edgecolor='black',
                                      alpha=0.9,
                                      boxstyle='round,pad=0.8'))
        
        nx.draw_networkx_edge_labels(self.graph, pos,
                                   edge_labels=edge_labels,
                                   font_size=10,
                                   font_family='Arial',
                                   font_weight='bold',
                                   bbox=dict(facecolor='white', 
                                           edgecolor='black',
                                           alpha=0.9,
                                           boxstyle='round,pad=0.5'))
        
        # Add title and legend
        plt.title('Knowledge Graph', fontsize=24, pad=40, fontweight='bold')
        
        # Add legend
        legend_elements = []
        for entity_type, color in self.entity_colors.items():
            legend_elements.append(plt.Line2D([0], [0], 
                                            marker='o', 
                                            color='w', 
                                            label=entity_type,
                                            markerfacecolor=color, 
                                            markersize=15))
        
        plt.legend(handles=legend_elements, 
                  loc='upper right',
                  bbox_to_anchor=(1.1, 1.1),
                  title='Entity Types',
                  title_fontsize=14,
                  fontsize=12)
        
        # Disable axes
        plt.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        return pos, self.graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import json
import os
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Optional
import networkx as nx
import matplotlib.pyplot as plt
from openai import OpenAI
from io import BytesIO
import base64
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Initialize OpenAI client - you'll need to set your API key as an environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Models for request and response structure
class Character(BaseModel):
    name: str
    mentions: int
    description: Optional[str] = None

class CharacterInteraction(BaseModel):
    character1: str
    character2: str
    interaction_count: int
    
class BookAnalysis(BaseModel):
    book_id: int
    title: Optional[str] = None
    author: Optional[str] = None
    characters: List[Character]
    interactions: List[CharacterInteraction]
    
# Constants for processing
CHUNK_SIZE = 4000  # Size of text chunks to send to the API
MAX_CHUNKS = 10    # Maximum number of chunks to process (to limit API costs)

@app.get('/')
def root():
    return {"Hello": "World"}

@app.get('/health')
def health():
    return {"status": "healthy"}
    try:
        # Try to fetch book content
        book_text = fetch_book_content(book_id)
        if not book_text:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        
        # Return a preview for basic analysis
        text_preview = book_text[:1000] + "..."
        return {
            "book_id": book_id,
            "status": "success",
            "preview": text_preview,
            "content_length": len(book_text)
        }
    except Exception as e:
        return {
            "book_id": book_id,
            "status": "error",
            "message": str(e)
        }

@app.get('/analyse-book/{book_id}/characters')
def analyse_book_characters(book_id: int):
    """Analyze characters in a book using ChatGPT"""
    try:
        # Fetch book content
        book_text = fetch_book_content(book_id)
        if not book_text:
            raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found")
        
        # Extract title and author from the text (basic heuristic)
        title, author = extract_metadata(book_text)
        
        # Process the book in chunks to identify characters
        characters, interactions = process_book_for_characters(book_text)
        
        # Create the full analysis
        analysis = BookAnalysis(
            book_id=book_id,
            title=title,
            author=author,
            characters=characters,
            interactions=interactions
        )
        
        return analysis
    except Exception as e:
        return {
            "book_id": book_id,
            "status": "error",
            "message": str(e)
        }

# @app.get('/analyse-book/{book_id}/visualize')
# def visualize_character_network(book_id: int):
    """Generate and return a visualization of character interactions"""
    try:
        # First get the character analysis
        analysis = analyse_book_characters(book_id)
        
        if isinstance(analysis, dict) and analysis.get("status") == "error":
            return analysis
        
        # Generate the visualization as an image
        # if generate_image:
        image_data = generate_network_visualization(analysis.characters, analysis.interactions)
        
        # return {
        #     "book_id": book_id,
        #     "status": "success",
        #     "title": analysis.title,
        #     "author": analysis.author,
        #     "visualization": image_data,
        #     "character_count": len(analysis.characters)
        # }
        return image_data
    except Exception as e:
        return {
            "book_id": book_id,
            "status": "error",
            "message": str(e)
        }

    """Generate character visualization using GPT-Image-1"""
    try:
        # First get the character analysis
        analysis = analyse_book_characters(book_id)
        
        if isinstance(analysis, dict) and analysis.get("status") == "error":
            return analysis
        
        # Generate a prompt for GPT-Image-1
        prompt = create_visualization_prompt(analysis)
        
        # Generate the image using OpenAI
        image_data = generate_openai_image(prompt)
        
        return {
            "book_id": book_id,
            "status": "success",
            "title": analysis.title,
            "author": analysis.author,
            "visualization": image_data,
            "prompt_used": prompt
        }
    except Exception as e:
        return {
            "book_id": book_id,
            "status": "error",
            "message": str(e)
        }

# Helper functions

def fetch_book_content(book_id: int) -> str:
    """Fetch book content from Project Gutenberg"""
    # Try the standard URL pattern
    content_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    content_response = requests.get(content_url)
    
    # If the first URL fails, try alternative patterns
    if content_response.status_code != 200:
        alternative_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        content_response = requests.get(alternative_url)
    
    # Check if the request was successful
    if content_response.status_code == 200:
        return content_response.text
    else:
        return None

def extract_metadata(text: str) -> tuple:
    """Extract title and author from the book text"""
    # Simple heuristic: look for common patterns in Gutenberg texts
    title = "Unknown Title"
    author = "Unknown Author"
    
    lines = text.split("\n")
    for i, line in enumerate(lines[:100]):  # Only check first 100 lines
        if "Title:" in line:
            title = line.replace("Title:", "").strip()
        elif "Author:" in line:
            author = line.replace("Author:", "").strip()
    
    # Alternative: use ChatGPT to extract this info
    if title == "Unknown Title" or author == "Unknown Author":
        try:
            metadata_text = text[:2000]  # Just use the beginning of the book
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Extract the title and author of this book if possible. Return as JSON with keys 'title' and 'author'."},
                    {"role": "user", "content": metadata_text}
                ],
                response_format={"type": "json_object"}
            )
            metadata = json.loads(response.choices[0].message.content)
            if "title" in metadata and metadata["title"]:
                title = metadata["title"]
            if "author" in metadata and metadata["author"]:
                author = metadata["author"]
        except Exception as e:
            print(f"Error extracting metadata with GPT: {e}")
    
    return title, author

def process_book_for_characters(text: str) -> tuple:
    """Process the book to identify characters and their interactions with case-insensitive handling"""
    chunks = split_into_chunks(text, CHUNK_SIZE)
    
    # Limit chunks to avoid excessive API costs
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    
    all_characters = {}
    all_interactions = {}
    canonical_names = {}  # Maps lowercase name to preferred capitalization
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}")
        
        # Use ChatGPT to identify characters in this chunk
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Identify all characters mentioned in this text. For each character, note how many times they appear and any interactions with other characters. Pay attention to variations of the same character name (case differences, titles, etc). Return as JSON with 'characters' (array of objects with 'name' and 'mentions') and 'interactions' (array of objects with 'character1', 'character2', and 'interaction_count')."},
                    {"role": "user", "content": chunk}
                ],
                response_format={"type": "json_object"}
            )
            
            chunk_analysis = json.loads(response.choices[0].message.content)
            
            # Update character counts with case-insensitive handling
            for character in chunk_analysis.get("characters", []):
                # Get the character name and normalize it
                original_name = character["name"]
                lowercase_name = original_name.lower()
                
                # Determine canonical name (preferred capitalization)
                if lowercase_name in canonical_names:
                    # We've seen this character before - use existing canonical name
                    canonical_name = canonical_names[lowercase_name]
                else:
                    # First time seeing this character - store preferred capitalization
                    canonical_name = original_name
                    canonical_names[lowercase_name] = canonical_name
                
                # Update character counts using canonical name
                if canonical_name in all_characters:
                    all_characters[canonical_name].mentions += character.get("mentions", 1)
                else:
                    all_characters[canonical_name] = Character(
                        name=canonical_name, 
                        mentions=character.get("mentions", 1),
                        description=character.get("description", "")
                    )
            
            # Update interaction counts with case-insensitive handling
            for interaction in chunk_analysis.get("interactions", []):
                # Get original character names
                orig_char1 = interaction["character1"]
                orig_char2 = interaction["character2"]
                
                # Map to canonical names
                char1 = canonical_names.get(orig_char1.lower(), orig_char1)
                char2 = canonical_names.get(orig_char2.lower(), orig_char2)
                
                # Skip self-interactions
                if char1.lower() == char2.lower():
                    continue
                
                # Create a consistent key for the interaction (alphabetical order)
                if char1.lower() > char2.lower():
                    char1, char2 = char2, char1
                
                interaction_key = f"{char1.lower()}|{char2.lower()}"
                
                if interaction_key in all_interactions:
                    all_interactions[interaction_key].interaction_count += interaction.get("interaction_count", 1)
                else:
                    all_interactions[interaction_key] = CharacterInteraction(
                        character1=char1,
                        character2=char2,
                        interaction_count=interaction.get("interaction_count", 1)
                    )
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
    
    # Convert to lists for the response
    characters_list = list(all_characters.values())
    interactions_list = list(all_interactions.values())
    
    # Sort characters by mentions (most mentioned first)
    characters_list.sort(key=lambda x: x.mentions, reverse=True)
    
    return characters_list, interactions_list


def split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split the text into chunks of approximately equal size"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# def process_book_for_characters(text: str) -> tuple:
#     """Process the book to identify characters and their interactions"""
#     chunks = split_into_chunks(text, CHUNK_SIZE)
    
#     # Limit chunks to avoid excessive API costs
#     if len(chunks) > MAX_CHUNKS:
#         chunks = chunks[:MAX_CHUNKS]
    
#     all_characters = {}
#     all_interactions = {}
    
#     # Process each chunk
#     for i, chunk in enumerate(chunks):
#         print(f"Processing chunk {i+1} of {len(chunks)}")
        
#         # Use ChatGPT to identify characters in this chunk
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "Identify all characters mentioned in this text. For each character, note how many times they appear and any interactions with other characters. Assume case sensitive names of characters as same characters only, for example 'romeo' and 'ROMEO' are same characters. Return as JSON with 'characters' (array of objects with 'name' and 'mentions') and 'interactions' (array of objects with 'character1', 'character2', and 'interaction_count')."},
#                     {"role": "user", "content": chunk}
#                 ],
#                 response_format={"type": "json_object"}
#             )
            
#             chunk_analysis = json.loads(response.choices[0].message.content)
            
#             # Update character counts
#             for character in chunk_analysis.get("characters", []):
#                 name = character["name"]
#                 if name in all_characters:
#                     all_characters[name].mentions += character.get("mentions", 1)
#                 else:
#                     all_characters[name] = Character(
#                         name=name, 
#                         mentions=character.get("mentions", 1),
#                         description=character.get("description", "")
#                     )
            
#             # Update interaction counts
#             for interaction in chunk_analysis.get("interactions", []):
#                 char1 = interaction["character1"]
#                 char2 = interaction["character2"]
                
#                 # Create a consistent key for the interaction (alphabetical order)
#                 if char1 > char2:
#                     char1, char2 = char2, char1
                
#                 interaction_key = f"{char1}|{char2}"
                
#                 if interaction_key in all_interactions:
#                     all_interactions[interaction_key].interaction_count += interaction.get("interaction_count", 1)
#                 else:
#                     all_interactions[interaction_key] = CharacterInteraction(
#                         character1=char1,
#                         character2=char2,
#                         interaction_count=interaction.get("interaction_count", 1)
#                     )
#         except Exception as e:
#             print(f"Error processing chunk {i+1}: {e}")
    
#     # Convert to lists for the response
#     characters_list = list(all_characters.values())
#     interactions_list = list(all_interactions.values())
    
#     # Sort characters by mentions (most mentioned first)
#     characters_list.sort(key=lambda x: x.mentions, reverse=True)
    
#     return characters_list, interactions_list

# def split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """Split the text into chunks of approximately equal size"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def generate_network_visualization(characters: List[Character], interactions: List[CharacterInteraction]) -> str:
    """
    Generate a network visualization of character interactions.
    
    Args:
        characters: List of Character objects with name, mentions, and optional description
        interactions: List of CharacterInteraction objects defining relationships between characters
        
    Returns:
        A string containing the HTML/JavaScript code for the network visualization
    """
    import json
    from typing import List, Dict, Any, Optional
    
    # Prepare nodes (characters)
    nodes = []
    for char in characters:
        # Size node based on number of mentions
        size = 10 + (char.mentions * 0.5)  # Base size + scaling factor
        nodes.append({
            "id": char.name,
            "label": char.name,
            "title": char.description or char.name,
            "value": char.mentions,
            "size": size
        })
    
    # Prepare edges (interactions)
    edges = []
    for interaction in interactions:
        # Width of edge based on interaction count
        width = 1 + (interaction.interaction_count * 0.2)  # Base width + scaling factor
        edges.append({
            "from": interaction.character1,
            "to": interaction.character2,
            "value": interaction.interaction_count,
            "width": width,
            "title": f"{interaction.interaction_count} interactions"
        })
    
    # Create the visualization HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Character Network Visualization</title>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js"></script>
        <style type="text/css">
            #network {
                width: 800px;
                height: 600px;
                border: 1px solid lightgray;
            }
        </style>
    </head>
    <body>
        <div id="network"></div>
        
        <script type="text/javascript">
            // Network data
            const nodes = NODES_PLACEHOLDER;
            const edges = EDGES_PLACEHOLDER;
            
            // Create dataset
            const nodesDataset = new vis.DataSet(nodes);
            const edgesDataset = new vis.DataSet(edges);
            
            // Network configuration
            const options = {
                nodes: {
                    shape: 'circle',
                    font: {
                        size: 16
                    },
                    borderWidth: 2,
                    shadow: true
                },
                edges: {
                    smooth: {
                        type: 'continuous'
                    },
                    arrows: {
                        to: {
                            enabled: false
                        }
                    },
                    color: {
                        color: '#848484',
                        highlight: '#0077CC'
                    },
                    scaling: {
                        min: 1,
                        max: 10
                    }
                },
                physics: {
                    barnesHut: {
                        gravitationalConstant: -5000,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04
                    },
                    stabilization: {
                        iterations: 2000
                    }
                },
                interaction: {
                    hover: true,
                    navigationButtons: true,
                    tooltipDelay: 200
                }
            };
            
            // Create network
            const container = document.getElementById('network');
            const data = {
                nodes: nodesDataset,
                edges: edgesDataset
            };
            const network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    
    # Replace placeholders with actual data
    html = html.replace('NODES_PLACEHOLDER', json.dumps(nodes))
    html = html.replace('EDGES_PLACEHOLDER', json.dumps(edges))
    
    return html

# def generate_network_visualization(characters: List[Character], interactions: List[CharacterInteraction]) -> str:
    """Generate a network visualization of character interactions using NetworkX with improved styling"""
    # Set the backend to 'Agg' which doesn't require a GUI
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create a graph
    G = nx.Graph()
    
    # Filter characters with generic/unclear names and consolidate similar references
    filtered_characters = []
    name_mapping = {}  # To map various forms of the same character to a canonical name
    
    # Words to filter out from character names
    generic_terms = ['the', 'a', 'an', 'unnamed', 'old', 'young', 'man', 'woman', 'person']
    
    # First pass - identify primary characters and create name mapping
    for character in characters:
        name = character.name
        
        # Skip very generic names
        if name.lower() in ['the narrator', 'narrator', 'the man', 'the woman', 'unnamed']:
            continue
            
        # Clean up names with articles
        clean_name = name
        for term in generic_terms:
            if clean_name.lower().startswith(f"the {term}"):
                clean_name = clean_name[4:].strip()
            if clean_name.lower().startswith(f"{term} "):
                clean_name = clean_name[len(term):].strip()
                
        # Capitalize first letter of each word for consistency
        clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        
        # Only include characters with meaningful names and sufficient mentions
        if len(clean_name) > 1 and character.mentions >= 3:
            name_mapping[name] = clean_name
            
            # Check if we already have this character (after cleanup)
            existing = next((c for c in filtered_characters if c.name == clean_name), None)
            if existing:
                existing.mentions += character.mentions
            else:
                # Create a new character with the clean name
                filtered_characters.append(Character(
                    name=clean_name,
                    mentions=character.mentions,
                    description=character.description
                ))
    
    # Sort characters by mentions and take top 12 for clarity
    filtered_characters.sort(key=lambda x: x.mentions, reverse=True)
    top_characters = filtered_characters[:12]
    
    # Create set of top character names for quick lookup
    top_names = {character.name for character in top_characters}
    
    # Add nodes (characters)
    for character in top_characters:
        G.add_node(character.name, size=character.mentions)
    
    # Filter and add edges (interactions) - only for top characters
    filtered_interactions = []
    
    for interaction in interactions:
        # Get the clean names if they exist in our mapping
        char1 = name_mapping.get(interaction.character1, interaction.character1)
        char2 = name_mapping.get(interaction.character2, interaction.character2)
        
        # Only add if both characters are in our top characters
        if char1 in top_names and char2 in top_names and char1 != char2:
            # Check if we already have this interaction
            existing = next((i for i in filtered_interactions 
                            if (i.character1 == char1 and i.character2 == char2) or 
                               (i.character1 == char2 and i.character2 == char1)), None)
            if existing:
                existing.interaction_count += interaction.interaction_count
            else:
                filtered_interactions.append(CharacterInteraction(
                    character1=char1,
                    character2=char2,
                    interaction_count=interaction.interaction_count
                ))
    
    # Add edges to graph
    for interaction in filtered_interactions:
        G.add_edge(interaction.character1, interaction.character2, weight=interaction.interaction_count)
    
    # If graph is empty or has no edges, add some dummy connections to make it look better
    if len(G.edges) == 0 and len(G.nodes) >= 2:
        chars = list(G.nodes)
        for i in range(len(chars)-1):
            G.add_edge(chars[i], chars[i+1], weight=1)
    
    # Create the visualization with dark themed styling like the second image
    plt.figure(figsize=(14, 10))
    
    # Set dark background
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='#1e2130')
    ax.set_facecolor('#1e2130')
    
    # Fix the problem of disconnected components by ensuring the graph is connected
    # If the graph has disconnected components, add weak edges to connect them
    if not nx.is_connected(G) and len(G.nodes) > 0:
        components = list(nx.connected_components(G))
        for i in range(len(components)-1):
            # Get one node from each component
            node1 = list(components[i])[0]
            node2 = list(components[i+1])[0]
            # Add a weak edge between components
            G.add_edge(node1, node2, weight=0.1)
    
    # Use a more controlled layout algorithm for character networks
    try:
        # For better connected networks
        if len(G.edges) >= len(G.nodes) - 1:
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        else:
            # For sparser networks, use a different algorithm
            pos = nx.kamada_kawai_layout(G)
    except:
        # Fallback to simple spring layout if other algorithms fail
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Normalize node sizes based on mentions for better visuals
    max_mentions = max(G.nodes[node].get('size', 1) for node in G.nodes) if G.nodes else 1
    min_mentions = min(G.nodes[node].get('size', 1) for node in G.nodes) if G.nodes else 1
    
    # Calculate edge widths based on interaction count with better scaling
    if G.edges:
        weights = [G.edges[edge].get('weight', 1) for edge in G.edges]
        max_weight = max(weights)
        min_weight = min(weights)
        
        # Ensure difference in weights for visual distinction
        if max_weight == min_weight:
            edge_widths = [1.5] * len(G.edges)
        else:
            min_width, max_width = 0.8, 3.0
            edge_widths = [min_width + ((G.edges[edge].get('weight', 1) - min_weight) / 
                                    (max_weight - min_weight)) * (max_width - min_width) 
                          for edge in G.edges]
    else:
        edge_widths = []
    
    # Draw edges first so they appear behind nodes
    for i, edge in enumerate(G.edges):
        source, target = edge
        sx, sy = pos[source]
        tx, ty = pos[target]
        
        # Add slight curve to the edges
        ax.annotate('', xy=(tx, ty), xytext=(sx, sy),
                   arrowprops=dict(arrowstyle='->', color='white', alpha=0.7,
                                  connectionstyle='arc3,rad=0.1', linewidth=edge_widths[i]))
    
    # Draw nodes - create proper boxes for characters
    for node in G.nodes:
        x, y = pos[node]
        
        # Calculate box dimensions based on text length
        box_width = len(node) * 0.018 + 0.05  # Scale width based on text length
        box_height = 0.04  # Fixed height
        
        # Create proper rectangular box
        rect = patches.Rectangle(
            (x - box_width/2, y - box_height/2),  # Lower left corner
            box_width,                           # Width
            box_height,                          # Height
            linewidth=1,
            edgecolor='white',
            facecolor='#222222',
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add character name text
        ax.text(
            x, y,                     # Center position
            node,                     # Character name
            color='white',
            fontsize=10,
            fontweight='bold',
            ha='center',              # Horizontal alignment
            va='center',              # Vertical alignment
            zorder=3                  # Ensure text is on top
        )
    
    # Turn off axis
    ax.axis('off')
    
    # Add a title
    plt.title('Character Relationship Network', fontsize=16, color='white', pad=20)
    
    # Adjust plot to ensure all nodes are visible and well-spaced
    plt.tight_layout()
    
    # Manually set axis limits with some padding to ensure all nodes are visible
    if pos:
        all_xs = [xy[0] for xy in pos.values()]
        all_ys = [xy[1] for xy in pos.values()]
        
        if all_xs and all_ys:  # Make sure there are values
            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)
            
            # Add padding
            padding = 0.2
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
    
    # Save to a BytesIO object with high DPI for better quality
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, facecolor='#1e2130', bbox_inches='tight')
    plt.close()
    
    # Convert to base64 for embedding in response
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str
    
# def create_visualization_prompt(analysis: BookAnalysis) -> str:
#     """Create a prompt for GPT-Image-1 based on character analysis with more precise box instructions"""
#     # Get the top characters with improved filtering
#     filtered_characters = []
    
#     # Words to filter out from character names
#     generic_terms = ['the', 'a', 'an', 'unnamed', 'old', 'young', 'man', 'woman', 'person', 'narrator']
    
#     # Filter and clean character names
#     for character in analysis.characters:
#         name = character.name
        
#         # Skip very generic names
#         if name.lower() in ['the narrator', 'narrator', 'the man', 'the woman', 'unnamed']:
#             continue
            
#         # Clean up names with articles
#         clean_name = name
#         for term in generic_terms:
#             if clean_name.lower().startswith(f"the {term}"):
#                 clean_name = clean_name[4:].strip()
#             if clean_name.lower().startswith(f"{term} "):
#                 clean_name = clean_name[len(term):].strip()
                
#         # Capitalize first letter of each word for consistency
#         clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        
#         # Only include characters with meaningful names and sufficient mentions
#         if len(clean_name) > 1 and character.mentions >= 3:
#             # Create a new character with the clean name
#             filtered_characters.append(Character(
#                 name=clean_name,
#                 mentions=character.mentions,
#                 description=character.description
#             ))
    
#     # Sort characters by mentions and take top 12 for clarity
#     filtered_characters.sort(key=lambda x: x.mentions, reverse=True)
#     top_characters = filtered_characters[:12]
    
#     # Create a prompt describing the character network
#     prompt = f"Create a professional character relationship diagram for the book '{analysis.title}' by {analysis.author}. "
#     prompt += "Make the visualization look like a formal directed graph diagram with the following specifications: "
    
#     # Style specifications
#     prompt += "Style: Dark navy background (#1e2130), with character names displayed in WHITE TEXT inside BLACK RECTANGULAR BOXES with white borders. "
#     prompt += "Each character should be represented by a distinct rectangular box with a thin white border. "
#     prompt += "Use white arrows to show relationships between characters. Thicker arrows indicate stronger relationships. "
    
#     # Character specifications
#     prompt += "Include these main characters as rectangular nodes with character names centered inside the boxes: "
    
#     # Add character descriptions
#     for character in top_characters:
#         prompt += f"{character.name}, "
    
#     prompt = prompt.rstrip(", ") + ". "
    
#     # Add information about the strongest connections
#     # Filter interactions to only include top characters
#     top_names = {c.name for c in top_characters}
#     relevant_interactions = [
#         i for i in analysis.interactions 
#         if i.character1 in top_names and i.character2 in top_names and i.character1 != i.character2
#     ]
    
#     # Sort by interaction strength
#     top_interactions = sorted(relevant_interactions, key=lambda x: x.interaction_count, reverse=True)[:10]
    
#     if top_interactions:
#         prompt += "Show these key relationships with arrows between characters: "
#         for interaction in top_interactions:
#             prompt += f"from {interaction.character1} to {interaction.character2}, "
        
#         prompt = prompt.rstrip(", ") + ". "
    
#     # Add style guidance with emphasis on proper boxes
#     prompt += "Make the diagram look exactly like a professionally designed character relationship visualization for literary analysis. "
#     prompt += "The style should match the second image example - minimalist and elegant design with clean white text inside black boxes on dark background. "
#     prompt += "IMPORTANT: Each character name must be CONTAINED COMPLETELY within its own black rectangular box with a white border. "
#     prompt += "Arrange the nodes in a balanced layout that clearly shows the relationships between characters. "
#     prompt += "Do not include any legends, labels or additional text other than character names in the boxes."
    
#     return prompt

# # def generate_network_visualization(characters: List[Character], interactions: List[CharacterInteraction]) -> str:
#     """Generate a network visualization of character interactions using NetworkX with improved styling"""
#     # Set the backend to 'Agg' which doesn't require a GUI
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
    
#     # Create a graph
#     G = nx.Graph()
    
#     # Filter characters with generic/unclear names and consolidate similar references
#     filtered_characters = []
#     name_mapping = {}  # To map various forms of the same character to a canonical name
    
#     # Words to filter out from character names
#     generic_terms = ['the', 'a', 'an', 'unnamed', 'old', 'young', 'man', 'woman', 'person']
    
#     # First pass - identify primary characters and create name mapping
#     for character in characters:
#         name = character.name
        
#         # Skip very generic names
#         if name.lower() in ['the narrator', 'narrator', 'the man', 'the woman', 'unnamed']:
#             continue
            
#         # Clean up names with articles
#         clean_name = name
#         for term in generic_terms:
#             if clean_name.lower().startswith(f"the {term}"):
#                 clean_name = clean_name[4:].strip()
#             if clean_name.lower().startswith(f"{term} "):
#                 clean_name = clean_name[len(term):].strip()
                
#         # Capitalize first letter of each word for consistency
#         clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        
#         # Only include characters with meaningful names and sufficient mentions
#         if len(clean_name) > 1 and character.mentions >= 3:
#             name_mapping[name] = clean_name
            
#             # Check if we already have this character (after cleanup)
#             existing = next((c for c in filtered_characters if c.name == clean_name), None)
#             if existing:
#                 existing.mentions += character.mentions
#             else:
#                 # Create a new character with the clean name
#                 filtered_characters.append(Character(
#                     name=clean_name,
#                     mentions=character.mentions,
#                     description=character.description
#                 ))
    
#     # Sort characters by mentions and take top 12 for clarity
#     filtered_characters.sort(key=lambda x: x.mentions, reverse=True)
#     top_characters = filtered_characters[:12]
    
#     # Create set of top character names for quick lookup
#     top_names = {character.name for character in top_characters}
    
#     # Add nodes (characters)
#     for character in top_characters:
#         G.add_node(character.name, size=character.mentions)
    
#     # Filter and add edges (interactions) - only for top characters
#     filtered_interactions = []
    
#     for interaction in interactions:
#         # Get the clean names if they exist in our mapping
#         char1 = name_mapping.get(interaction.character1, interaction.character1)
#         char2 = name_mapping.get(interaction.character2, interaction.character2)
        
#         # Only add if both characters are in our top characters
#         if char1 in top_names and char2 in top_names and char1 != char2:
#             # Check if we already have this interaction
#             existing = next((i for i in filtered_interactions 
#                             if (i.character1 == char1 and i.character2 == char2) or 
#                                (i.character1 == char2 and i.character2 == char1)), None)
#             if existing:
#                 existing.interaction_count += interaction.interaction_count
#             else:
#                 filtered_interactions.append(CharacterInteraction(
#                     character1=char1,
#                     character2=char2,
#                     interaction_count=interaction.interaction_count
#                 ))
    
#     # Add edges to graph
#     for interaction in filtered_interactions:
#         G.add_edge(interaction.character1, interaction.character2, weight=interaction.interaction_count)
    
#     # If graph is empty or has no edges, add some dummy connections to make it look better
#     if len(G.edges) == 0 and len(G.nodes) >= 2:
#         chars = list(G.nodes)
#         for i in range(len(chars)-1):
#             G.add_edge(chars[i], chars[i+1], weight=1)
    
#     # Create the visualization with dark themed styling like the second image
#     plt.figure(figsize=(14, 10))
    
#     # Set dark background
#     plt.style.use('dark_background')
    
#     # Fix the problem of disconnected components by ensuring the graph is connected
#     # If the graph has disconnected components, add weak edges to connect them
#     if not nx.is_connected(G) and len(G.nodes) > 0:
#         components = list(nx.connected_components(G))
#         for i in range(len(components)-1):
#             # Get one node from each component
#             node1 = list(components[i])[0]
#             node2 = list(components[i+1])[0]
#             # Add a weak edge between components
#             G.add_edge(node1, node2, weight=0.1)
    
#     # Use a more controlled layout algorithm for character networks
#     try:
#         # For better connected networks
#         if len(G.edges) >= len(G.nodes) - 1:
#             pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
#         else:
#             # For sparser networks, use a different algorithm
#             pos = nx.kamada_kawai_layout(G)
#     except:
#         # Fallback to simple spring layout if other algorithms fail
#         pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
#     # Normalize node sizes based on mentions for better visuals
#     max_mentions = max(G.nodes[node].get('size', 1) for node in G.nodes) if G.nodes else 1
#     min_mentions = min(G.nodes[node].get('size', 1) for node in G.nodes) if G.nodes else 1
    
#     # Define node size range
#     min_size, max_size = 2000, 4000
    
#     # If all nodes have same mentions, use a default size
#     if max_mentions == min_mentions:
#         node_sizes = [3000] * len(G.nodes)
#     else:
#         # Calculate node sizes with better scaling
#         node_sizes = [min_size + ((G.nodes[node].get('size', 1) - min_mentions) / 
#                                (max_mentions - min_mentions)) * (max_size - min_size) 
#                      for node in G.nodes]
    
#     # Calculate edge widths based on interaction count with better scaling
#     if G.edges:
#         weights = [G.edges[edge].get('weight', 1) for edge in G.edges]
#         max_weight = max(weights)
#         min_weight = min(weights)
        
#         # Ensure difference in weights for visual distinction
#         if max_weight == min_weight:
#             edge_widths = [2.0] * len(G.edges)
#         else:
#             min_width, max_width = 1.0, 4.0
#             edge_widths = [min_width + ((G.edges[edge].get('weight', 1) - min_weight) / 
#                                     (max_weight - min_weight)) * (max_width - min_width) 
#                           for edge in G.edges]
#     else:
#         edge_widths = []
    
#     # Draw the nodes - use rectangles for a more professional look
#     for i, node in enumerate(G.nodes):
#         x, y = pos[node]
        
#         # Calculate rectangle size based on character name length
#         text_width = len(node) * 0.01 + 0.08
#         height = 0.04
        
#         # Create rectangle
#         rect = plt.Rectangle((x - text_width/2, y - height/2), text_width, height, 
#                            facecolor='#1a1a1a', edgecolor='white', linewidth=1, alpha=0.9)
#         plt.gca().add_patch(rect)
        
#         # Add character name
#         plt.text(x, y, node, ha='center', va='center', color='white', 
#                 fontsize=10, fontweight='bold')
    
#     # Draw edges with arrows to show relationships
#     for i, edge in enumerate(G.edges):
#         source, target = edge
#         sx, sy = pos[source]
#         tx, ty = pos[target]
        
#         # Add slight curve to the edges
#         plt.annotate('', xy=(tx, ty), xytext=(sx, sy),
#                    arrowprops=dict(arrowstyle='->', color='white', alpha=0.7,
#                                   connectionstyle='arc3,rad=0.1', linewidth=edge_widths[i]))
    
#     # Turn off axis
#     plt.axis('off')
    
#     # Add a title
#     plt.title('Character Relationship Network', fontsize=16, color='white', pad=20)
    
#     # Adjust plot to ensure all nodes are visible and well-spaced
#     plt.tight_layout()
    
#     # Manually set axis limits with some padding to ensure all nodes are visible
#     if pos:
#         all_xs = [xy[0] for xy in pos.values()]
#         all_ys = [xy[1] for xy in pos.values()]
        
#         if all_xs and all_ys:  # Make sure there are values
#             x_min, x_max = min(all_xs), max(all_xs)
#             y_min, y_max = min(all_ys), max(all_ys)
            
#             # Add padding
#             padding = 0.1
#             plt.xlim(x_min - padding, x_max + padding)
#             plt.ylim(y_min - padding, y_max + padding)
    
#     # Save to a BytesIO object with high DPI for better quality
#     buf = BytesIO()
#     plt.savefig(buf, format='png', dpi=300, facecolor='#1e2130', bbox_inches='tight')
#     plt.close()
    
#     # Convert to base64 for embedding in response
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
    
#     return img_str
# # def generate_network_visualization(characters: List[Character], interactions: List[CharacterInteraction]) -> str:
#     """Generate a network visualization of character interactions using NetworkX with improved styling"""
#     # Set the backend to 'Agg' which doesn't require a GUI
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
    
#     # Create a graph
#     G = nx.Graph()
    
#     # Filter characters with generic/unclear names and consolidate similar references
#     filtered_characters = []
#     name_mapping = {}  # To map various forms of the same character to a canonical name
    
#     # Words to filter out from character names
#     generic_terms = ['the', 'a', 'an', 'unnamed', 'old', 'young', 'man', 'woman', 'person']
    
#     # First pass - identify primary characters and create name mapping
#     for character in characters:
#         name = character.name
        
#         # Skip very generic names
#         if name.lower() in ['the narrator', 'narrator', 'the man', 'the woman', 'unnamed']:
#             continue
            
#         # Clean up names with articles
#         clean_name = name
#         for term in generic_terms:
#             if clean_name.lower().startswith(f"the {term}"):
#                 clean_name = clean_name[4:].strip()
#             if clean_name.lower().startswith(f"{term} "):
#                 clean_name = clean_name[len(term):].strip()
                
#         # Capitalize first letter of each word for consistency
#         clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        
#         # Only include characters with meaningful names and sufficient mentions
#         if len(clean_name) > 1 and character.mentions >= 3:
#             name_mapping[name] = clean_name
            
#             # Check if we already have this character (after cleanup)
#             existing = next((c for c in filtered_characters if c.name == clean_name), None)
#             if existing:
#                 existing.mentions += character.mentions
#             else:
#                 # Create a new character with the clean name
#                 filtered_characters.append(Character(
#                     name=clean_name,
#                     mentions=character.mentions,
#                     description=character.description
#                 ))
    
#     # Sort characters by mentions and take top 12 for clarity
#     filtered_characters.sort(key=lambda x: x.mentions, reverse=True)
#     top_characters = filtered_characters[:12]
    
#     # Create set of top character names for quick lookup
#     top_names = {character.name for character in top_characters}
    
#     # Add nodes (characters)
#     for character in top_characters:
#         G.add_node(character.name, size=character.mentions)
    
#     # Filter and add edges (interactions) - only for top characters
#     filtered_interactions = []
    
#     for interaction in interactions:
#         # Get the clean names if they exist in our mapping
#         char1 = name_mapping.get(interaction.character1, interaction.character1)
#         char2 = name_mapping.get(interaction.character2, interaction.character2)
        
#         # Only add if both characters are in our top characters
#         if char1 in top_names and char2 in top_names and char1 != char2:
#             # Check if we already have this interaction
#             existing = next((i for i in filtered_interactions 
#                             if (i.character1 == char1 and i.character2 == char2) or 
#                                (i.character1 == char2 and i.character2 == char1)), None)
#             if existing:
#                 existing.interaction_count += interaction.interaction_count
#             else:
#                 filtered_interactions.append(CharacterInteraction(
#                     character1=char1,
#                     character2=char2,
#                     interaction_count=interaction.interaction_count
#                 ))
    
#     # Add edges to graph
#     for interaction in filtered_interactions:
#         G.add_edge(interaction.character1, interaction.character2, weight=interaction.interaction_count)
    
#     # If graph is empty or has no edges, add some dummy connections to make it look better
#     if len(G.edges) == 0 and len(G.nodes) >= 2:
#         chars = list(G.nodes)
#         for i in range(len(chars)-1):
#             G.add_edge(chars[i], chars[i+1], weight=1)
    
#     # Create the visualization with dark themed styling like the second image
#     plt.figure(figsize=(14, 10))
    
#     # Set dark background
#     plt.style.use('dark_background')
    
#     # Fix the problem of disconnected components by ensuring the graph is connected
#     # If the graph has disconnected components, add weak edges to connect them
#     if not nx.is_connected(G) and len(G.nodes) > 0:
#         components = list(nx.connected_components(G))
#         for i in range(len(components)-1):
#             # Get one node from each component
#             node1 = list(components[i])[0]
#             node2 = list(components[i+1])[0]
#             # Add a weak edge between components
#             G.add_edge(node1, node2, weight=0.1)
    
#     # Use a more controlled layout algorithm for character networks
#     try:
#         # For better connected networks
#         if len(G.edges) >= len(G.nodes) - 1:
#             pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
#         else:
#             # For sparser networks, use a different algorithm
#             pos = nx.kamada_kawai_layout(G)
#     except:
#         # Fallback to simple spring layout if other algorithms fail
#         pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
#     # Normalize node sizes based on mentions for better visuals
#     max_mentions = max(G.nodes[node].get('size', 1) for node in G.nodes) if G.nodes else 1
#     min_mentions = min(G.nodes[node].get('size', 1) for node in G.nodes) if G.nodes else 1
    
#     # Define node size range
#     min_size, max_size = 2000, 4000
    
#     # If all nodes have same mentions, use a default size
#     if max_mentions == min_mentions:
#         node_sizes = [3000] * len(G.nodes)
#     else:
#         # Calculate node sizes with better scaling
#         node_sizes = [min_size + ((G.nodes[node].get('size', 1) - min_mentions) / 
#                                (max_mentions - min_mentions)) * (max_size - min_size) 
#                      for node in G.nodes]
    
#     # Calculate edge widths based on interaction count with better scaling
#     if G.edges:
#         weights = [G.edges[edge].get('weight', 1) for edge in G.edges]
#         max_weight = max(weights)
#         min_weight = min(weights)
        
#         # Ensure difference in weights for visual distinction
#         if max_weight == min_weight:
#             edge_widths = [2.0] * len(G.edges)
#         else:
#             min_width, max_width = 1.0, 4.0
#             edge_widths = [min_width + ((G.edges[edge].get('weight', 1) - min_weight) / 
#                                     (max_weight - min_weight)) * (max_width - min_width) 
#                           for edge in G.edges]
#     else:
#         edge_widths = []
    
#     # Draw the nodes - use rectangles for a more professional look
#     for i, node in enumerate(G.nodes):
#         x, y = pos[node]
        
#         # Calculate rectangle size based on character name length
#         text_width = len(node) * 0.01 + 0.08
#         height = 0.04
        
#         # Create rectangle
#         rect = plt.Rectangle((x - text_width/2, y - height/2), text_width, height, 
#                            facecolor='#1a1a1a', edgecolor='white', linewidth=1, alpha=0.9)
#         plt.gca().add_patch(rect)
        
#         # Add character name
#         plt.text(x, y, node, ha='center', va='center', color='white', 
#                 fontsize=10, fontweight='bold')
    
#     # Draw edges with arrows to show relationships
#     for i, edge in enumerate(G.edges):
#         source, target = edge
#         sx, sy = pos[source]
#         tx, ty = pos[target]
        
#         # Add slight curve to the edges
#         plt.annotate('', xy=(tx, ty), xytext=(sx, sy),
#                    arrowprops=dict(arrowstyle='->', color='white', alpha=0.7,
#                                   connectionstyle='arc3,rad=0.1', linewidth=edge_widths[i]))
    
#     # Turn off axis
#     plt.axis('off')
    
#     # Add a title
#     plt.title('Character Relationship Network', fontsize=16, color='white', pad=20)
    
#     # Adjust plot to ensure all nodes are visible and well-spaced
#     plt.tight_layout()
    
#     # Manually set axis limits with some padding to ensure all nodes are visible
#     if pos:
#         all_xs = [xy[0] for xy in pos.values()]
#         all_ys = [xy[1] for xy in pos.values()]
        
#         if all_xs and all_ys:  # Make sure there are values
#             x_min, x_max = min(all_xs), max(all_xs)
#             y_min, y_max = min(all_ys), max(all_ys)
            
#             # Add padding
#             padding = 0.1
#             plt.xlim(x_min - padding, x_max + padding)
#             plt.ylim(y_min - padding, y_max + padding)
    
#     # Save to a BytesIO object with high DPI for better quality
#     buf = BytesIO()
#     plt.savefig(buf, format='png', dpi=300, facecolor='#1e2130', bbox_inches='tight')
#     plt.close()
    
#     # Convert to base64 for embedding in response
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
    
#     return img_str

# # def generate_network_visualization(characters: List[Character], interactions: List[CharacterInteraction]) -> str:
#     """Generate a network visualization of character interactions using NetworkX with improved styling"""
#     # Set the backend to 'Agg' which doesn't require a GUI
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
    
#     # Create a graph
#     G = nx.Graph()
    
#     # Filter characters with generic/unclear names and consolidate similar references
#     filtered_characters = []
#     name_mapping = {}  # To map various forms of the same character to a canonical name
    
#     # Words to filter out from character names
#     generic_terms = ['the', 'a', 'an', 'unnamed', 'old', 'young', 'man', 'woman', 'person', 'narrator']
    
#     # First pass - identify primary characters and create name mapping
#     for character in characters:
#         name = character.name
        
#         # Skip very generic names
#         if name.lower() in ['the narrator', 'narrator', 'the man', 'the woman', 'unnamed']:
#             continue
            
#         # Clean up names with articles
#         clean_name = name
#         for term in generic_terms:
#             if clean_name.lower().startswith(f"the {term}"):
#                 clean_name = clean_name[4:].strip()
#             if clean_name.lower().startswith(f"{term} "):
#                 clean_name = clean_name[len(term):].strip()
                
#         # Capitalize first letter of each word for consistency
#         clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        
#         # Only include characters with meaningful names and sufficient mentions
#         if len(clean_name) > 1 and character.mentions >= 3:
#             name_mapping[name] = clean_name
            
#             # Check if we already have this character (after cleanup)
#             existing = next((c for c in filtered_characters if c.name == clean_name), None)
#             if existing:
#                 existing.mentions += character.mentions
#             else:
#                 # Create a new character with the clean name
#                 filtered_characters.append(Character(
#                     name=clean_name,
#                     mentions=character.mentions,
#                     description=character.description
#                 ))
    
#     # Sort characters by mentions and take top 15 for clarity
#     filtered_characters.sort(key=lambda x: x.mentions, reverse=True)
#     top_characters = filtered_characters[:15]
    
#     # Create set of top character names for quick lookup
#     top_names = {character.name for character in top_characters}
    
#     # Add nodes (characters)
#     for character in top_characters:
#         G.add_node(character.name, size=character.mentions)
    
#     # Filter and add edges (interactions) - only for top characters
#     filtered_interactions = []
    
#     for interaction in interactions:
#         # Get the clean names if they exist in our mapping
#         char1 = name_mapping.get(interaction.character1, interaction.character1)
#         char2 = name_mapping.get(interaction.character2, interaction.character2)
        
#         # Only add if both characters are in our top characters
#         if char1 in top_names and char2 in top_names and char1 != char2:
#             # Check if we already have this interaction
#             existing = next((i for i in filtered_interactions 
#                             if (i.character1 == char1 and i.character2 == char2) or 
#                                (i.character1 == char2 and i.character2 == char1)), None)
#             if existing:
#                 existing.interaction_count += interaction.interaction_count
#             else:
#                 filtered_interactions.append(CharacterInteraction(
#                     character1=char1,
#                     character2=char2,
#                     interaction_count=interaction.interaction_count
#                 ))
    
#     # Add edges to graph
#     for interaction in filtered_interactions:
#         G.add_edge(interaction.character1, interaction.character2, weight=interaction.interaction_count)
    
#     # Create the visualization with dark themed styling like the second image
#     plt.figure(figsize=(12, 10))
    
#     # Set dark background
#     plt.style.use('dark_background')
    
#     # Use a better layout algorithm for character networks
#     pos = nx.kamada_kawai_layout(G)
    
#     # Normalize node sizes based on mentions for better visuals
#     max_mentions = max(G.nodes[node].get('size', 1) for node in G.nodes) if G.nodes else 1
#     min_size, max_size = 800, 3000  # Min and max node sizes
    
#     # Calculate node sizes with better scaling
#     node_sizes = [min_size + (G.nodes[node].get('size', 1) / max_mentions) * (max_size - min_size) 
#                  for node in G.nodes]
    
#     # Calculate edge widths based on interaction count with better scaling
#     max_weight = max((G.edges[edge].get('weight', 1) for edge in G.edges), default=1)
#     min_width, max_width = 0.5, 2.5  # Min and max edge widths
    
#     edge_widths = [min_width + (G.edges[edge].get('weight', 1) / max_weight) * (max_width - min_width) 
#                   for edge in G.edges]
    
#     # Draw with improved styling
#     # Nodes - rectangles with character names
#     for node, (x, y) in pos.items():
#         size = G.nodes[node].get('size', 1) / max_mentions
#         width = max(len(node) * 0.1, 0.3)  # Adjust width based on name length
#         height = 0.15  # Fixed height
        
#         # Create rectangle node
#         rect = plt.Rectangle((x - width/2, y - height/2), width, height, 
#                             facecolor='#222222', edgecolor='white', alpha=0.9, zorder=2)
#         plt.gca().add_patch(rect)
        
#         # Add character name in white
#         plt.text(x, y, node, color='white', fontsize=9, fontweight='bold',
#                 ha='center', va='center', zorder=3)
    
#     # Draw edges as arrows to show relationship direction
#     for edge in G.edges:
#         source, target = edge
#         sx, sy = pos[source]
#         tx, ty = pos[target]
        
#         # Determine edge weight for thickness
#         weight = G.edges[edge].get('weight', 1)
#         width = min_width + (weight / max_weight) * (max_width - min_width)
        
#         # Draw arrow
#         plt.annotate('', xy=(tx, ty), xytext=(sx, sy),
#                     arrowprops=dict(arrowstyle='->', color='white', alpha=0.6,
#                                    connectionstyle='arc3,rad=0.1', linewidth=width),
#                     zorder=1)
    
#     plt.axis('off')
#     plt.tight_layout()
    
#     # Save to a BytesIO object
#     buf = BytesIO()
#     plt.savefig(buf, format='png', dpi=300, facecolor='#1e2130')
#     plt.close()
    
#     # Convert to base64 for embedding in response
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
    
#     return img_str

# # def generate_network_visualization(characters: List[Character], interactions: List[CharacterInteraction]) -> str:
#     """Generate a network visualization of character interactions using NetworkX"""
#     # Create a graph
#     G = nx.Graph()
    
#     # Only include the top 20 characters by mentions to avoid cluttered graphs
#     top_characters = characters[:20]
#     # top_characters = characters
    
#     # Add nodes (characters)
#     for character in top_characters:
#         G.add_node(character.name, size=character.mentions)
    
#     # Add edges (interactions)
#     for interaction in interactions:
#         # Only add if both characters are in our top characters
#         if interaction.character1 in [c.name for c in top_characters] and interaction.character2 in [c.name for c in top_characters]:
#             G.add_edge(interaction.character1, interaction.character2, weight=interaction.interaction_count)
    
#     # Create the visualization
#     plt.figure(figsize=(12, 8))
    
#     # Node positions
#     pos = nx.spring_layout(G, seed=42)
    
#     # Node sizes based on mentions
#     node_sizes = [G.nodes[node].get('size', 1) * 20 for node in G.nodes]
    
#     # Edge widths based on interaction count
#     edge_widths = [G.edges[edge].get('weight', 1) / 2 for edge in G.edges]
    
#     # Draw the graph
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
#     nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
#     nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
#     plt.axis('off')
#     plt.title('Character Interaction Network')
    
#     # Save to a BytesIO object
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close()
    
#     # Convert to base64 for embedding in response
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
    
#     return img_str

def create_visualization_prompt(analysis: BookAnalysis) -> str:
    """Create a prompt for GPT-Image-1 based on character analysis"""
    # Get the top 10 characters
    top_characters = analysis.characters[:10]
    
    # Create a prompt describing the character network
    prompt = f"Create a character relationship diagram for the book '{analysis.title}' by {analysis.author}. "
    prompt += "The visualization should be a network graph with the following characters as nodes: "
    
    # Add character descriptions
    for character in top_characters:
        prompt += f"{character.name} (mentioned {character.mentions} times), "
    
    prompt = prompt.rstrip(", ") + ". "
    
    # Add information about the strongest connections
    top_interactions = sorted(analysis.interactions, key=lambda x: x.interaction_count, reverse=True)[:10]
    if top_interactions:
        prompt += "The strongest character relationships are: "
        for interaction in top_interactions:
            if interaction.character1 in [c.name for c in top_characters] and interaction.character2 in [c.name for c in top_characters]:
                prompt += f"{interaction.character1} and {interaction.character2} (interaction strength: {interaction.interaction_count}), "
        
        prompt = prompt.rstrip(", ") + ". "
    
    # Add style guidance
    prompt += "Create the visualization in a clean, modern style with clear labels. Use a color scheme that reflects the book's themes. Make connections between characters visible as lines with thickness representing relationship strength."
    
    return prompt

def generate_openai_image(prompt: str) -> str:
    """Generate an image using OpenAI's GPT-Image-1 model"""
    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            quality="medium",
            size="1024x1024",
            response_format="b64_json"
        )
        
        # Extract the base64-encoded image
        image_data = response.data[0].b64_json
        
        return image_data
    except Exception as e:
        print(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {e}")
    

def generate_html_network_visualization(characters: List[Character], interactions: List[CharacterInteraction]) -> str:
    """Generate a mobile-responsive HTML visualization of character interactions using D3.js"""
    # Filter characters with generic/unclear names and consolidate similar references
    filtered_characters = []
    name_mapping = {}  # To map various forms of the same character to a canonical name
    
    # Words to filter out from character names
    generic_terms = ['the', 'a', 'an', 'unnamed', 'old', 'young', 'man', 'woman', 'person']
    
    # First pass - identify primary characters and create name mapping
    for character in characters:
        name = character.name
        
        # Skip very generic names
        if name.lower() in ['the narrator', 'narrator', 'the man', 'the woman', 'unnamed']:
            continue
            
        # Clean up names with articles
        clean_name = name
        for term in generic_terms:
            if clean_name.lower().startswith(f"the {term}"):
                clean_name = clean_name[4:].strip()
            if clean_name.lower().startswith(f"{term} "):
                clean_name = clean_name[len(term):].strip()
                
        # Capitalize first letter of each word for consistency
        clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        
        # Only include characters with meaningful names and sufficient mentions
        if len(clean_name) > 1 and character.mentions >= 3:
            name_mapping[name] = clean_name
            
            # Check if we already have this character (after cleanup)
            existing = next((c for c in filtered_characters if c.name == clean_name), None)
            if existing:
                existing.mentions += character.mentions
            else:
                # Create a new character with the clean name
                filtered_characters.append(Character(
                    name=clean_name,
                    mentions=character.mentions,
                    description=character.description
                ))
    
    # Sort characters by mentions and take top 12 for clarity (fewer for mobile)
    filtered_characters.sort(key=lambda x: x.mentions, reverse=True)
    top_characters = filtered_characters[:12]
    
    # Create set of top character names for quick lookup
    top_names = {character.name for character in top_characters}
    
    # Filter and process interactions - only for top characters
    filtered_interactions = []
    
    for interaction in interactions:
        # Get the clean names if they exist in our mapping
        char1 = name_mapping.get(interaction.character1, interaction.character1)
        char2 = name_mapping.get(interaction.character2, interaction.character2)
        
        # Only add if both characters are in our top characters
        if char1 in top_names and char2 in top_names and char1 != char2:
            # Check if we already have this interaction
            existing = next((i for i in filtered_interactions 
                            if (i.character1 == char1 and i.character2 == char2) or 
                               (i.character1 == char2 and i.character2 == char1)), None)
            if existing:
                existing.interaction_count += interaction.interaction_count
            else:
                filtered_interactions.append(CharacterInteraction(
                    character1=char1,
                    character2=char2,
                    interaction_count=interaction.interaction_count
                ))
    
    # Prepare data for D3.js visualization
    nodes_data = []
    for character in top_characters:
        nodes_data.append({
            "id": character.name,
            "name": character.name,
            "value": character.mentions,
            "description": character.description or f"Mentioned {character.mentions} times in the book."
        })
    
    links_data = []
    for interaction in filtered_interactions:
        links_data.append({
            "source": interaction.character1,
            "target": interaction.character2,
            "value": interaction.interaction_count
        })
    
    # Generate HTML with embedded D3.js visualization optimized for mobile
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Character Network Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        /* Base styles */
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #1e2130;
            color: white;
            overflow-x: hidden;
            touch-action: manipulation;
        }}
        
        .container {{
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            padding: 10px;
        }}
        
        h1 {{
            text-align: center;
            margin: 10px 0;
            font-size: calc(1.2rem + 1vw);
            word-wrap: break-word;
        }}
        
        #visualization {{
            width: 100%;
            height: calc(100vh - 130px); /* Adjust for header and controls */
            border: 1px solid #2d3748;
            border-radius: 5px;
            overflow: hidden;
            touch-action: none; /* Prevent browser handling of touch gestures */
        }}
        
        .links line {{
            stroke: #fff;
            stroke-opacity: 0.6;
        }}
        
        .nodes text {{
            font-weight: bold;
            font-size: 10px; /* Smaller font for mobile */
            text-shadow: 0 1px 2px rgba(0,0,0,0.8);
        }}
        
        .tooltip {{
            position: absolute;
            padding: 8px;
            background-color: rgba(0, 0, 0, 0.9);
            border-radius: 5px;
            color: white;
            pointer-events: none;
            z-index: 100;
            max-width: 80%;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }}
        
        .controls {{
            display: flex;
            justify-content: center;
            margin: 10px 0;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .controls button {{
            padding: 8px 12px;
            background-color: #4a5568;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            touch-action: manipulation;
            -webkit-tap-highlight-color: transparent;
        }}
        
        .controls button:hover, .controls button:active {{
            background-color: #2d3748;
        }}
        
        .info-bar {{
            font-size: 12px;
            text-align: center;
            padding: 5px;
            opacity: 0.8;
        }}
        
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            z-index: 1000;
        }}
        
        /* Mobile optimizations */
        @media (max-width: 768px) {{
            .container {{
                padding: 5px;
            }}
            
            h1 {{
                font-size: 1.2rem;
                margin: 5px 0;
            }}
            
            .controls button {{
                padding: 6px 10px;
                font-size: 12px;
                flex-grow: 1;
            }}
            
            .nodes text {{
                font-size: 9px;
            }}
            
            .tooltip {{
                font-size: 12px;
                padding: 5px;
            }}
            
            #visualization {{
                height: calc(100vh - 110px);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Character Relationship Network</h1>
        <div class="controls">
            <button id="resetZoom">Reset View</button>
            <button id="toggleLabels">Toggle Labels</button>
            <button id="togglePhysics">Freeze/Unfreeze</button>
        </div>
        <div id="loading">Loading visualization...</div>
        <div id="visualization"></div>
        <div class="info-bar">Tap a character to see details. Drag to reposition.</div>
    </div>
    
    <script>
        // Network data
        const graph = {{
            "nodes": {json.dumps(nodes_data)},
            "links": {json.dumps(links_data)}
        }};
        
        // Determine if mobile device
        const isMobile = window.innerWidth <= 768;
        
        // Adjust force parameters based on device
        const forceStrength = isMobile ? -200 : -300;
        const linkDistance = isMobile ? 100 : 150;
        
        // Create the visualization
        const width = document.getElementById('visualization').clientWidth;
        const height = document.getElementById('visualization').clientHeight;
        
        // Create tooltip
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
        
        // Create SVG container
        const svg = d3.select("#visualization")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", `0 0 ${{width}} ${{height}}`)
            .attr("preserveAspectRatio", "xMidYMid meet");
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
            
        svg.call(zoom);
        
        // Create main container group
        const g = svg.append("g");
        
        // Create marker definitions for arrowheads
        svg.append("defs").selectAll("marker")
            .data(["arrow"])
            .enter()
            .append("marker")
            .attr("id", d => d)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)  // Adjusted for mobile
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("fill", "#fff")
            .attr("d", "M0,-5L10,0L0,5");
        
        // Create simulation
        const simulation = d3.forceSimulation(graph.nodes)
            .force("link", d3.forceLink(graph.links).id(d => d.id).distance(linkDistance))
            .force("charge", d3.forceManyBody().strength(forceStrength))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collide", d3.forceCollide().radius(d => Math.sqrt(d.value) * 1.5 + 15));
        
        // Create links
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(graph.links)
            .enter()
            .append("line")
            .attr("stroke-width", d => Math.max(1, Math.sqrt(d.value) * 0.5))
            .attr("stroke", "white")
            .attr("opacity", 0.6)
            .attr("marker-end", "url(#arrow)");
        
        // Create nodes group
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            .data(graph.nodes)
            .enter()
            .append("g");
        
        // Add text labels
        const labels = node.append("text")
            .text(d => d.name)
            .attr("x", 0)
            .attr("y", 0)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .attr("fill", "white")
            .style("pointer-events", "none");
        
        // Create invisible drag area (larger than text for better touch targets)
        node.append("circle")
            .attr("r", d => Math.max(15, d.name.length * 3.5))
            .style("fill", "transparent")
            .style("cursor", "pointer");
        
        // Drag behavior
        const drag = d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
            
        node.call(drag);
        
        // Touch/click handlers (optimized for mobile)
        node.on("touchstart mouseenter", function(event, d) {{
            // Prevent default touch behavior
            event.preventDefault();
            
            tooltip.transition()
                .duration(200)
                .style("opacity", .9);
                
            tooltip.html(`<strong>${{d.name}}</strong><br/>${{d.description}}`)
                .style("left", (event.type === "touchstart" ? 
                    event.touches[0].pageX : event.pageX) + "px")
                .style("top", (event.type === "touchstart" ? 
                    event.touches[0].pageY - 40 : event.pageY - 28) + "px");
                
            // Highlight connected links
            link.style("opacity", l => 
                (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1
            );
            
            // Highlight connected nodes
            node.style("opacity", n => 
                (n.id === d.id || graph.links.some(l => 
                    (l.source.id === d.id && l.target.id === n.id) || 
                    (l.target.id === d.id && l.source.id === n.id)
                )) ? 1 : 0.3
            );
        }})
        .on("touchend mouseleave", function() {{
            setTimeout(() => {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }}, 3000);  // Keep tooltip visible longer on mobile
                
            // Reset highlighting with delay for mobile
            setTimeout(() => {{
                link.style("opacity", 0.6);
                node.style("opacity", 1);
            }}, 1000);
        }});
        
        // Simulation tick event
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
                
            node
                .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        // Reset zoom handler
        document.getElementById("resetZoom").addEventListener("click", function(event) {{
            event.preventDefault();
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        }});
        
        // Toggle labels handler
        let labelsVisible = true;
        document.getElementById("toggleLabels").addEventListener("click", function(event) {{
            event.preventDefault();
            labelsVisible = !labelsVisible;
            labels.style("display", labelsVisible ? "block" : "none");
        }});
        
        // Toggle physics handler
        let physicsActive = true;
        document.getElementById("togglePhysics").addEventListener("click", function(event) {{
            event.preventDefault();
            physicsActive = !physicsActive;
            
            if (physicsActive) {{
                simulation.alphaTarget(0.3).restart();
                // Release all fixed positions
                graph.nodes.forEach(node => {{
                    node.fx = null;
                    node.fy = null;
                }});
            }} else {{
                simulation.alphaTarget(0);
                // Fix all nodes in their current positions
                graph.nodes.forEach(node => {{
                    node.fx = node.x;
                    node.fy = node.y;
                }});
            }}
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            // Keep nodes fixed if physics is turned off
            if (physicsActive) {{
                d.fx = null;
                d.fy = null;
            }}
        }}
        
        // Initial layout optimization for mobile
        if (isMobile) {{
            // Run simulation for a bit to better position nodes
            for (let i = 0; i < 300; ++i) simulation.tick();
        }}
        
        // Hide loading indicator when visualization is ready
        simulation.on("end", () => {{
            document.getElementById("loading").style.display = "none";
        }});
        
        // Hide loading after timeout in case simulation doesn't end
        setTimeout(() => {{
            document.getElementById("loading").style.display = "none";
        }}, 3000);
    </script>
</body>
</html>
    """
    
    return html

@app.get('/analyse-book/{book_id}/html-visualization')
def html_character_network(book_id: int):
    """Generate and return an HTML visualization of character interactions"""
    try:
        # First get the character analysis
        analysis = analyse_book_characters(book_id)
        
        if isinstance(analysis, dict) and analysis.get("status") == "error":
            return analysis
        
        # Generate the HTML visualization
        html_content = generate_html_network_visualization(analysis.characters, analysis.interactions)
        
        # Return HTML content with the correct content type
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return {
            "book_id": book_id,
            "status": "error",
            "message": str(e)
        }
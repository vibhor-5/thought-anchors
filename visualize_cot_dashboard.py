import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import scipy.special

# Import utility functions from existing codebase
from cluster_chunks import (
    load_model_and_tokenizer, get_problem_dirs, load_problem_and_solutions,
    get_residual_stream_activations
)

# Set up paths
cots_dir = Path("cots")
analysis_dir = Path("analysis")
output_dir = Path("visualization_output")
output_dir.mkdir(exist_ok=True)

# Cache for activations and attention
activation_cache = {}
attention_cache = {}

def load_chunks_data(problem_dir: Path, seed_dir: Path) -> List[Dict]:
    """Load chunks data from a seed directory."""
    chunks_file = seed_dir / "chunks.json"
    
    if not chunks_file.exists():
        return []
        
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    return chunks_data

def get_token_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layers: List[int]
) -> Dict[int, torch.Tensor]:
    """Get token activations for specific layers with caching."""
    # Create a cache key
    cache_key = f"{text[:100]}_{'-'.join(map(str, layers))}"
    
    # Check if already in cache
    if cache_key in activation_cache:
        return activation_cache[cache_key]
    
    # Get activations
    activations = get_residual_stream_activations(model, tokenizer, text, layers)
    
    # Cache the result
    activation_cache[cache_key] = activations
    
    return activations

def get_token_attention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layer: int,
    head: Optional[int] = None
) -> torch.Tensor:
    """Get attention weights for a specific layer and optionally head."""
    # Create a cache key
    cache_key = f"{text[:100]}_{layer}_{head}"
    
    # Check if already in cache
    if cache_key in attention_cache:
        return attention_cache[cache_key]
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # Extract attention weights
    # Shape: [batch, num_heads, seq_len, seq_len]
    attention = outputs.attentions[layer][0]  # Layer index, batch index
    
    if head is not None:
        attention = attention[head]  # Select specific head
    
    # Cache the result
    attention_cache[cache_key] = attention
    
    return attention

def get_token_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str
) -> torch.Tensor:
    """Get next token logits for each position in the text."""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Get logits
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Extract logits
    # Shape: [batch, seq_len, vocab_size]
    logits = outputs.logits[0]  # Batch index
    
    return logits

def calculate_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of token distribution."""
    # Convert logits to probabilities using softmax
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1) / torch.log(torch.tensor(2.0))
    
    return entropy

def get_top_tokens(
    logits: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 10
) -> List[List[Tuple[str, float]]]:
    """Get top k tokens and their probabilities for each position."""
    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get top k tokens and probabilities
    top_probs, top_indices = torch.topk(probs, k, dim=-1)
    
    # Convert to list of tuples (token, probability)
    result = []
    for pos in range(top_indices.shape[0]):
        pos_tokens = []
        for i in range(k):
            token_id = top_indices[pos, i].item()
            token = tokenizer.decode([token_id])
            prob = top_probs[pos, i].item()
            pos_tokens.append((token, prob))
        result.append(pos_tokens)
    
    return result

def extract_examples_for_visualization(
    problem_dirs: List[Path],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: List[int],
    max_examples: int = 10
) -> List[Dict]:
    """Extract examples with all necessary data for visualization."""
    examples = []
    example_count = 0
    
    for problem_dir in tqdm(problem_dirs, desc="Processing problems"):
        # Load problem and solutions
        problem, solutions = load_problem_and_solutions(problem_dir)
        if not solutions:
            continue
        
        # Process all seeds
        analysis_problem_dir = analysis_dir / problem_dir.name
        seed_dirs = [d for d in analysis_problem_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        
        for seed_dir in seed_dirs:
            seed_id = seed_dir.name
            
            # Find the corresponding solution
            solution_dict = None
            for sol in solutions:
                if sol.get("seed") == int(seed_id.replace("seed_", "")):
                    solution_dict = sol
                    break
            
            if not solution_dict:
                continue
                
            full_text = solution_dict["solution"]
            
            # Load chunks data
            chunks_data = load_chunks_data(problem_dir, seed_dir)
            
            if not chunks_data:
                continue
            
            # Extract chunks
            chunks = [item["text"] for item in chunks_data]
            
            # Get token indices for each chunk in the full text
            chunk_token_ranges = []
            full_tokens = tokenizer(full_text, return_tensors="pt").to(model.device)
            
            # Find token ranges for each chunk
            current_pos = 0
            for chunk in chunks:
                # Find the chunk in the full text starting from current position
                chunk_start = full_text.find(chunk, current_pos)
                if chunk_start == -1:
                    # If exact match not found, try with some flexibility
                    chunk_words = chunk.split()
                    for i in range(current_pos, len(full_text) - len(chunk)):
                        if full_text[i:i+len(chunk_words[0])] == chunk_words[0]:
                            potential_match = full_text[i:i+len(chunk)]
                            if potential_match.split() == chunk_words:
                                chunk_start = i
                                break
                
                if chunk_start == -1:
                    print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
                    continue
                    
                chunk_end = chunk_start + len(chunk)
                current_pos = chunk_end
                
                # Convert character positions to token indices
                chunk_tokens = tokenizer(full_text[:chunk_start], return_tensors="pt")
                start_idx = len(chunk_tokens.input_ids[0]) - 1
                
                chunk_tokens = tokenizer(full_text[:chunk_end], return_tensors="pt")
                end_idx = len(chunk_tokens.input_ids[0])
                
                chunk_token_ranges.append((max(0, start_idx), end_idx))
            
            # Get token-level information
            token_ids = full_tokens.input_ids[0].tolist()
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
            
            # Get activations for the full text
            activations = get_token_activations(model, tokenizer, full_text, layers)
            
            # Get logits for next token prediction
            logits = get_token_logits(model, tokenizer, full_text)
            
            # Calculate entropy
            entropy = calculate_token_entropy(logits)
            
            # Get top tokens
            top_tokens = get_top_tokens(logits, tokenizer)
            
            # Get attention weights for the last layer
            attention = get_token_attention(model, tokenizer, full_text, layers[-1])
            
            # Create example dictionary
            example = {
                "problem_id": problem_dir.name,
                "seed_id": seed_id,
                "problem": problem,
                "full_text": full_text,
                "chunks_data": chunks_data,
                "chunk_token_ranges": chunk_token_ranges,
                "token_ids": token_ids,
                "tokens": tokens,
                "entropy": entropy.cpu().numpy().tolist(),
                "top_tokens": top_tokens,
                "attention": attention.cpu().numpy().tolist(),
                "layers": layers
            }
            
            # Save activations to disk for reuse
            activation_file = output_dir / f"{problem_dir.name}_{seed_id}_activations.pt"
            torch.save({layer: act.cpu() for layer, act in activations.items()}, activation_file)
            
            examples.append(example)
            example_count += 1
            
            if example_count >= max_examples:
                return examples
    
    return examples

def create_dashboard(examples: List[Dict]):
    """Create an interactive dashboard for CoT visualization."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    
    # Define category colors - 11 distinct colors
    category_colors = {
        "Initializing": "#8dd3c7",
        "Deduction": "#ffffb3",
        "Adding Knowledge": "#bebada",
        "Example Testing": "#fb8072",
        "Uncertainty Estimation": "#80b1d3",
        "Backtracking": "#fdb462",
        "Comparative Analysis": "#b3de69",
        "Question Posing": "#fccde5",
        "Summary": "#d9d9d9",
        "Metaphorical Thinking": "#bc80bd",
        "Final Answer": "#ccebc5",
        "Unknown": "#cccccc"  # Added an extra color for Unknown
    }
    
    # Add custom CSS for token highlighting
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .token {
                    padding: 2px;
                    margin: 1px;
                    border-radius: 3px;
                    transition: background-color 0.3s;
                }
                .token:hover {
                    background-color: #f0f0f0;
                }
                .token-selected {
                    background-color: #ffcc00 !important;
                    font-weight: bold;
                }
                .token-attended {
                    background-color: rgba(255, 0, 0, 0.3);
                }
                .category-label {
                    padding: 5px 10px;
                    border-radius: 5px;
                    font-weight: bold;
                    color: black;
                    margin-right: 10px;
                    white-space: nowrap;
                }
                .chunk-container {
                    display: flex;
                    flex-direction: row;
                    align-items: flex-start;
                    margin-bottom: 10px;
                    padding: 5px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .chunk-content {
                    flex: 1;
                }
                .cot-solution-panel {
                    height: 80vh;
                    overflow-y: auto;
                }
                .analysis-panel {
                    height: 40vh;
                    overflow-y: auto;
                }
            </style>
            {%scripts%}
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Define layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("CoT Reasoning Visualization Dashboard", className="text-center my-4"),
                html.Hr(),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Select Example"),
                dcc.Dropdown(
                    id="example-dropdown",
                    options=[
                        {"label": f"Problem {ex['problem_id']} - Seed {ex['seed_id']}", "value": i}
                        for i, ex in enumerate(examples)
                    ],
                    value=0
                ),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Problem Statement", className="mt-4"),
                html.Div(id="problem-statement", className="p-3 border rounded"),
            ], width=12)
        ]),
        
        dbc.Row([
            # Left column - CoT Solution
            dbc.Col([
                html.H4("CoT Solution with Chunks", className="mt-4"),
                html.Div(id="cot-solution", className="p-3 border rounded cot-solution-panel"),
            ], width=6),
            
            # Right column - Analysis panels in a 2x2 grid
            dbc.Col([
                dbc.Row([
                    # Top row - Token info and Top tokens
                    dbc.Col([
                        html.H4("Token Analysis", className="mt-4"),
                        dbc.Card([
                            dbc.CardHeader("Selected Token Information"),
                            dbc.CardBody([
                                html.P("Click on a token in the solution to see its analysis."),
                                html.Div(id="token-info")
                            ])
                        ], className="analysis-panel")
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Top Next Tokens", className="mt-4"),
                        html.Div(id="top-tokens-table", className="analysis-panel")
                    ], width=6)
                ]),
                
                dbc.Row([
                    # Bottom row - Attention and Entropy
                    dbc.Col([
                        html.H4("Attention Visualization", className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Layer:"),
                                dcc.Dropdown(
                                    id="layer-dropdown",
                                    options=[],  # Will be populated dynamically
                                    value=None
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Label("Head:"),
                                dcc.Dropdown(
                                    id="head-dropdown",
                                    options=[],  # Will be populated dynamically
                                    value="avg"
                                ),
                            ], width=6),
                        ], className="mb-2"),
                        dcc.Graph(id="attention-graph", className="analysis-panel")
                    ], width=6),
                    
                    dbc.Col([
                        html.H4("Token Entropy", className="mt-4"),
                        dcc.Graph(id="entropy-graph", className="analysis-panel")
                    ], width=6)
                ])
            ], width=6)
        ]),
        
        # Store the selected token index
        dcc.Store(id="selected-token-index", data=None),
        
        html.Hr(),
        html.Footer("CoT Analysis Dashboard", className="text-center my-4"),
        
        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                # This div will be updated whenever any callback is running
                html.Div(id="loading-output")
            ]
        )
    ], fluid=True)
    
    # Define callbacks
    @app.callback(
        [
            Output("problem-statement", "children"),
            Output("cot-solution", "children"),
            Output("layer-dropdown", "options"),
            Output("layer-dropdown", "value")
        ],
        [Input("example-dropdown", "value")]
    )
    def update_example(example_idx):
        if example_idx is None:
            raise PreventUpdate
        
        example = examples[example_idx]
        problem = example["problem"]
        
        # Convert problem dictionary to formatted text
        if isinstance(problem, dict):
            problem_text = []
            for key, value in problem.items():
                problem_text.append(html.P(f"{key.capitalize()}: {value}"))
            problem_display = html.Div(problem_text)
        else:
            problem_display = html.P(str(problem))
        
        # Create the CoT solution with chunks
        chunks_data = example["chunks_data"]
        chunk_token_ranges = example["chunk_token_ranges"]
        tokens = example["tokens"]
        
        # Create token spans with chunk boundaries
        token_spans = []
        for i, (start_idx, end_idx) in enumerate(chunk_token_ranges):
            if i >= len(chunks_data):
                continue
                
            category = chunks_data[i].get("category", "Unknown")
            category_color = category_colors.get(category, category_colors["Unknown"])
            
            # Create a div for this chunk
            chunk_tokens = []
            for j in range(start_idx, end_idx):
                if j < len(tokens):
                    token_span = html.Span(
                        tokens[j],
                        id={"type": "token", "index": j},
                        className="token",
                        style={"cursor": "pointer"}
                    )
                    chunk_tokens.append(token_span)
            
            # Create chunk div with category label in a row layout
            chunk_div = html.Div(
                [
                    html.Div(
                        category,
                        className="category-label",
                        style={"backgroundColor": category_color}
                    ),
                    html.Div(
                        chunk_tokens,
                        className="chunk-content"
                    )
                ],
                className="chunk-container"
            )
            
            token_spans.append(chunk_div)
        
        # Get available layers for this example
        layers = example.get("layers", [])
        layer_options = [{"label": f"Layer {layer}", "value": layer} for layer in layers]
        default_layer = layers[-1] if layers else None
        
        return problem_display, html.Div(token_spans), layer_options, default_layer
    
    @app.callback(
        Output("head-dropdown", "options"),
        [Input("example-dropdown", "value"),
         Input("layer-dropdown", "value")]
    )
    def update_head_options(example_idx, layer):
        if example_idx is None or layer is None:
            raise PreventUpdate
        
        example = examples[example_idx]
        
        # Get attention shape to determine number of heads
        attention = np.array(example["attention"])
        
        # Check if attention is 3D (with heads dimension)
        if len(attention.shape) == 3:
            num_heads = attention.shape[0]
            head_options = [{"label": "Average", "value": "avg"}] + [
                {"label": f"Head {i}", "value": i} for i in range(num_heads)
            ]
        else:
            # If attention is already averaged or only has one head
            head_options = [{"label": "Average", "value": "avg"}]
        
        return head_options
    
    @app.callback(
        Output("selected-token-index", "data"),
        [Input({"type": "token", "index": dash.dependencies.ALL}, "n_clicks")],
        [State({"type": "token", "index": dash.dependencies.ALL}, "id")]
    )
    def update_selected_token(n_clicks, token_ids):
        if not n_clicks or not any(n_clicks):
            raise PreventUpdate
        
        # Get the ID of the component that triggered the callback
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        # Extract the token index from the triggered component's ID
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # If it's a JSON string (for pattern-matching callbacks), parse it
        if '{' in triggered_id:
            import json
            triggered_id = json.loads(triggered_id)
            if 'index' in triggered_id:
                return triggered_id['index']
        
        # Fallback to the old method if the above doesn't work
        valid_clicks = [(i, clicks) for i, clicks in enumerate(n_clicks) if clicks is not None]
        if not valid_clicks:
            raise PreventUpdate
        
        max_idx, _ = max(valid_clicks, key=lambda x: x[1])
        token_idx = token_ids[max_idx]["index"]
        
        return token_idx
    
    @app.callback(
        [
            Output("token-info", "children"),
            Output("attention-graph", "figure"),
            Output("entropy-graph", "figure"),
            Output("top-tokens-table", "children"),
            Output({"type": "token", "index": dash.dependencies.ALL}, "className"),
            Output({"type": "token", "index": dash.dependencies.ALL}, "style")
        ],
        [
            Input("selected-token-index", "data"),
            Input("layer-dropdown", "value"),
            Input("head-dropdown", "value")
        ],
        [
            State("example-dropdown", "value"),
            State({"type": "token", "index": dash.dependencies.ALL}, "id")
        ],
        prevent_initial_call=True
    )
    def update_token_analysis(token_idx, layer, head, example_idx, token_ids):
        if token_idx is None or example_idx is None or layer is None:
            raise PreventUpdate
        
        # Set default head value if None
        if head is None:
            head = "avg"
        
        example = examples[example_idx]
        tokens = example["tokens"]
        
        if token_idx >= len(tokens):
            raise PreventUpdate
        
        # Get token information
        token = tokens[token_idx]
        token_id = example["token_ids"][token_idx]
        
        # Find which chunk this token belongs to
        chunk_idx = None
        for i, (start_idx, end_idx) in enumerate(example["chunk_token_ranges"]):
            if start_idx <= token_idx < end_idx:
                chunk_idx = i
                break
        
        chunk_category = "Unknown"
        if chunk_idx is not None and chunk_idx < len(example["chunks_data"]):
            chunk_category = example["chunks_data"][chunk_idx].get("category", "Unknown")
        
        # Create token info card
        token_info = dbc.Card([
            dbc.CardBody([
                html.H5(f"Token: '{token}'", className="card-title"),
                html.P(f"Token ID: {token_id}"),
                html.P(f"Position: {token_idx}"),
                html.P(f"Chunk: {chunk_idx}"),
                html.P(f"Category: {chunk_category}"),
                html.P(f"Entropy: {example['entropy'][token_idx]:.4f}"),
                html.P(f"Layer: {layer}"),
                html.P(f"Head: {head}")
            ])
        ])
        
        # Create attention visualization
        attention = np.array(example["attention"])
        
        # Process attention based on selected layer and head
        if len(attention.shape) == 3:  # [num_heads, seq_len, seq_len]
            if head == "avg":
                # Average across heads
                token_attention = attention.mean(axis=0)[token_idx, :token_idx+1]
            else:
                # Select specific head
                try:
                    head_idx = int(head)
                    token_attention = attention[head_idx, token_idx, :token_idx+1]
                except (ValueError, TypeError):
                    # Fall back to average if head is not a valid integer
                    token_attention = attention.mean(axis=0)[token_idx, :token_idx+1]
        else:
            # Already averaged or single head
            token_attention = attention[token_idx, :token_idx+1]
        
        # Ensure token_attention is a 1D array
        if isinstance(token_attention, list):
            token_attention = np.array(token_attention)
        if len(token_attention.shape) > 1:
            token_attention = token_attention.flatten()
        
        # Create attention figure
        attention_fig = go.Figure()
        
        # Add attention bars
        attention_fig.add_trace(go.Bar(
            x=[tokens[i] for i in range(token_idx+1)],
            y=token_attention,
            marker_color='rgba(255, 0, 0, 0.6)',
            name='Attention Weight'
        ))
        
        attention_fig.update_layout(
            title=f"Attention from '{token}' to Previous Tokens (Layer {layer}, Head {head})",
            xaxis_title="Tokens",
            yaxis_title="Attention Weight",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        # Create entropy visualization
        entropy = example["entropy"]
        
        # Create entropy figure
        entropy_fig = go.Figure()
        
        # Add entropy line
        entropy_fig.add_trace(go.Scatter(
            x=list(range(len(entropy))),
            y=entropy,
            mode='lines+markers',
            name='Token Entropy',
            line=dict(color='rgba(50, 171, 96, 0.6)')
        ))
        
        # Add vertical line for selected token
        entropy_fig.add_shape(
            type="line",
            x0=token_idx,
            x1=token_idx,
            y0=0,
            y1=max(entropy),
            line=dict(color="red", width=2, dash="dash")
        )
        
        entropy_fig.update_layout(
            title="Token Entropy Across Sequence",
            xaxis_title="Token Position",
            yaxis_title="Entropy",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300
        )
        
        # Create top tokens table
        top_tokens = example["top_tokens"][token_idx]
        
        top_tokens_table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Rank"),
                    html.Th("Token"),
                    html.Th("Probability")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(i+1),
                    html.Td(token),
                    html.Td(f"{prob:.4f}")
                ]) for i, (token, prob) in enumerate(top_tokens)
            ])
        ], bordered=True, hover=True, striped=True, size="sm")
        
        # Update token classes and styles to highlight selected token and attended tokens
        token_classes = []
        token_styles = []
        
        # Get attention weights for highlighting
        attended_weights = token_attention.tolist()
        
        # Normalize attention weights for opacity, but with a minimum threshold
        max_weight = max(attended_weights) if attended_weights else 1.0
        if max_weight == 0:
            max_weight = 1.0
            
        for token_id_obj in token_ids:
            idx = token_id_obj["index"]
            
            # Default style
            style = {"cursor": "pointer"}
            
            # Default class
            if idx == token_idx:
                # Selected token
                token_class = "token token-selected"
            elif idx < token_idx and idx < len(attended_weights):
                # Attended token - opacity based on attention weight but with a minimum value
                token_class = "token token-attended"
                
                # Calculate opacity with a minimum of 0.2 and scale the rest between 0.3 and 1.0
                raw_opacity = attended_weights[idx] / max_weight
                if raw_opacity > 0.01:  # If there's any meaningful attention
                    opacity = 0.3 + (raw_opacity * 0.7)  # Scale between 0.3 and 1.0
                else:
                    opacity = 0  # No highlighting for extremely low attention
                    
                style = {
                    "cursor": "pointer",
                    "backgroundColor": f"rgba(255, 0, 0, {opacity})"
                }
            else:
                # Regular token
                token_class = "token"
            
            token_classes.append(token_class)
            token_styles.append(style)
        
        return token_info, attention_fig, entropy_fig, top_tokens_table, token_classes, token_styles
    
    @app.callback(
        Output("loading-output", "children"),
        [Input("selected-token-index", "data")]
    )
    def update_loading(token_idx):
        return ""
    
    return app

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create CoT visualization dashboard")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name")
    parser.add_argument("--layers", type=list, default=list(range(32)), help="Layer to extract activations from")
    parser.add_argument("--max_examples", type=int, default=10, help="Maximum number of examples to process")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Get problem directories
    problem_dirs = get_problem_dirs(cots_dir)
    
    # Extract examples for visualization
    examples = extract_examples_for_visualization(
        problem_dirs, model, tokenizer, [*args.layers], args.max_examples
    )
    
    # Create dashboard
    app = create_dashboard(examples)
    
    # Run the app
    app.run(debug=True)

if __name__ == "__main__":
    main() 
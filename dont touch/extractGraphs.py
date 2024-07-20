import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objs as go
from openai import OpenAI
from dash.exceptions import PreventUpdate
import re
import wikipedia

# OpenAI API setup
from api_key import API_KEY, BACKGROUND_STORY

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

# Global variables
G = nx.DiGraph()
file_content = ""


def extract_wikipedia_content(url):
    try:
        # Extract the title from the URL
        title = url.split("/wiki/")[-1].replace("_", " ")

        # Fetch the Wikipedia page
        page = wikipedia.page(title)

        # Get the summary
        summary = wikipedia.summary(title, sentences=10)  # You can adjust the number of sentences

        # Combine the title and summary
        content = f"Title: {page.title}\n\nSummary:\n{summary}"

        return content
    except wikipedia.exceptions.DisambiguationError as e:
        return f"DisambiguationError: {e.options}"
    except wikipedia.exceptions.PageError:
        return "PageError: The page does not exist."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def extract_entities_and_relationships(text):
    try:
        max_tokens = 15000
        truncated_text = text[:max_tokens]

        print("Input text (truncated):", truncated_text[:500])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": BACKGROUND_STORY},
                {"role": "user",
                 "content": f"Extract entities and relationships from the following text. Return the result as a list of tuples, where each tuple contains three elements (entity1, relationship, entity2). Each element must be 4 words or less. Format the response as a valid Python list of tuples:\n\n{truncated_text}"}
            ]
        )

        content = response.choices[0].message.content
        print("API response:", content)

        # Try to extract tuples using regex
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, content)

        entities_relationships = []
        for match in matches:
            entity1 = summarize_relationship(match[0].strip().strip("'\""))
            relationship = summarize_relationship(match[1].strip().strip("'\""))
            entity2 = summarize_relationship(match[2].strip().strip("'\""))

            if entity1 and relationship and entity2:  # Only add if all parts are valid
                entities_relationships.append((entity1, relationship, entity2))

        print("Extracted and summarized entities and relationships:", entities_relationships)
        return entities_relationships

    except Exception as e:
        print(f"Error in API call or parsing: {e}")
        return []


def build_knowledge_graph(entities_and_relationships):
    G = nx.DiGraph()

    for item in entities_and_relationships:
        if len(item) == 3 and all(len(part.split()) <= 4 for part in item):
            entity1, relationship, entity2 = item
            # Capitalize the first letter of each word in entities
            entity1 = ' '.join(word.capitalize() for word in entity1.split())
            entity2 = ' '.join(word.capitalize() for word in entity2.split())
            G.add_node(entity1)
            G.add_node(entity2)
            G.add_edge(entity1, entity2, relationship=relationship)
        else:
            print(f"Skipping invalid relationship: {item}")

    print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G


def highlight_relationship(text, term):
    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    return pattern.sub(lambda m: f'<span class="highlight">{m.group()}</span>', text)


def summarize_relationship(text):
    words = text.split()
    if len(words) <= 4:
        return text

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a concise summarizer. Your task is to summarize the given phrase into 4 words or less, maintaining the core meaning."},
                {"role": "user", "content": f"Summarize this in 4 words or less: {text}"}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return summary if len(summary.split()) <= 4 else None
    except Exception as e:
        print(f"Error in summarizing: {e}")
        return None  # If we can't summarize, we'll return None and filter it out later


app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.H1("Wikipedia Knowledge Graph Visualizer", className='app-title'),
    html.Div([
        dcc.Input(
            id='wikipedia-url',
            type='text',
            placeholder='Enter Wikipedia URL...',
            className='url-input'
        ),
        html.Button('Generate', id='generate-button', n_clicks=0, className='generate-button'),
    ], className='input-container'),

    # Full app (initially hidden)
    html.Div([
        dcc.Loading(
            id="loading",
            type="circle",
            children=[dcc.Graph(id='knowledge-graph', className='knowledge-graph')]
        ),
        html.Div([
            html.Div([
                html.H3("Wikipedia Content", className='section-title'),
                html.Div(id='text-display', className='document-text')
            ], className='wikipedia-content-container'),
            html.Div([
                html.H3("Node Details", className='section-title'),
                dcc.Loading(
                    id="loading-node-details",
                    type="circle",
                    children=[html.Div(id='node-details', className='node-details')]
                ),
                html.Button('Generate Node Graph', id='generate-node-graph-button',
                            className='generate-node-graph-button')
            ], className='node-details-container'),
        ], className='bottom-container')
    ], id='full-app', style={'display': 'none'})
], className='app-container')

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Wikipedia Knowledge Graph Visualizer</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            body, html {
                height: 100%;
                margin: 0;
                font-family: 'Roboto', sans-serif;
                background-color: #f0f4f8;
            }
            .app-container {
                display: flex;
                flex-direction: column;
                height: 100vh;
                padding: 20px;
                box-sizing: border-box;
            }
            #landing-page {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .app-title {
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 2.5em;
                text-align: center;
            }
            .input-container {
                display: flex;
                margin-bottom: 20px;
                justify-content: center;
                width: 100%;
            }
            .url-input {
                width: 60%;
                padding: 12px 20px;
                font-size: 16px;
                border: 2px solid #bdc3c7;
                border-radius: 50px;
                transition: all 0.3s ease;
            }
            .url-input:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3);
            }
            .generate-button, .generate-node-graph-button {
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                color: white;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .generate-button {
                background: linear-gradient(45deg, #3498db, #2980b9);
                margin-left: 10px;
            }
            .generate-button:hover {
                background: linear-gradient(45deg, #2980b9, #3498db);
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            }
            .generate-node-graph-button {
                background: linear-gradient(45deg, #2ecc71, #27ae60);
                margin-top: 15px;
            }
            .generate-node-graph-button:hover {
                background: linear-gradient(45deg, #27ae60, #2ecc71);
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            }
            .knowledge-graph {
                flex-grow: 1;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
                margin-bottom: 20px;
                min-height: 500px;
            }
            .bottom-container {
                display: flex;
                justify-content: space-between;
                height: 300px;
            }
            .node-details-container, .wikipedia-content-container {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                width: 48%;
                overflow-y: auto;
            }
            .section-title {
                color: #2c3e50;
                margin-top: 0;
            }
            .document-text {
                line-height: 1.6;
            }
            .highlight {
                background-color: #fff176;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
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


@app.callback(
    [Output('wikipedia-url', 'value'),
     Output('generate-button', 'n_clicks')],
    [Input('generate-node-graph-button', 'n_clicks')],
    [State('node-details', 'children')]
)
def update_url_and_generate(n_clicks, node_details):
    if n_clicks is None or not node_details or isinstance(node_details, str):
        raise PreventUpdate

    # Extract the node name from the node details
    node_name = node_details[0]['props']['children']

    # Generate Wikipedia URL for the node
    wikipedia_url = f"https://en.wikipedia.org/wiki/{node_name.replace(' ', '_')}"

    # Return the new URL and increment the generate button's n_clicks
    return wikipedia_url, n_clicks


def get_wikipedia_summary(title):
    try:
        return wikipedia.summary(title, sentences=3)  # Adjust the number of sentences as needed
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this topic."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def highlight_relationship(text, term):
    if not term:
        return text
    # Escape special regex characters in the term
    escaped_term = re.escape(term)
    # Create a case-insensitive pattern that matches the whole term
    pattern = re.compile(r'\b(' + escaped_term + r')\b', re.IGNORECASE)
    return pattern.sub(r'<span class="highlight">\1</span>', text)


def process_text(text):
    lines = text.split('\n')
    processed_text = [line.strip() for line in lines if line.strip()]
    highlighted_text = [f'<p>{line}</p>' for line in processed_text]
    return html.Div([
        dcc.Markdown(
            children='\n'.join(highlighted_text),
            dangerously_allow_html=True,
        )
    ], className='document-text')


def create_graph_figure(G):
    if not G:
        return {}

    # Find the main entity (node with most connections)
    main_entity = max(G.nodes(), key=lambda n: G.degree(n))

    # Create subgraph
    subgraph = nx.ego_graph(G, main_entity, radius=1)
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text = [], [], []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.capitalize())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            size=30,
            color='#1f77b4',
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=16)
    )

    edge_labels = []
    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_labels.append(
            go.Scatter(
                x=[(x0 + x1) / 2],
                y=[(y0 + y1) / 2],
                text=[edge[2]['relationship']],
                mode='text',
                textposition='middle center',
                textfont=dict(size=14, color='red'),
                hoverinfo='text',
                hoverlabel=dict(bgcolor='white'),
                showlegend=False
            )
        )

    figure = {
        'data': [edge_trace, node_trace] + edge_labels,
        'layout': go.Layout(
            title=f'Knowledge Graph centered on {main_entity}',
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
    }

    return figure


import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import networkx as nx
import plotly.graph_objs as go
from openai import OpenAI
from dash.exceptions import PreventUpdate
import re
import wikipedia

# OpenAI API setup
from api_key import API_KEY, BACKGROUND_STORY

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

# Global variables
G = nx.DiGraph()
file_content = ""


def extract_wikipedia_content(url):
    try:
        # Extract the title from the URL
        title = url.split("/wiki/")[-1].replace("_", " ")

        # Fetch the Wikipedia page
        page = wikipedia.page(title)

        # Get the summary
        summary = wikipedia.summary(title, sentences=10)  # You can adjust the number of sentences

        # Combine the title and summary
        content = f"Title: {page.title}\n\nSummary:\n{summary}"

        return content
    except wikipedia.exceptions.DisambiguationError as e:
        return f"DisambiguationError: {e.options}"
    except wikipedia.exceptions.PageError:
        return "PageError: The page does not exist."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def extract_entities_and_relationships(text):
    try:
        max_tokens = 15000
        truncated_text = text[:max_tokens]

        print("Input text (truncated):", truncated_text[:500])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": BACKGROUND_STORY},
                {"role": "user",
                 "content": f"Extract entities and relationships from the following text. Return the result as a list of tuples, where each tuple contains three elements (entity1, relationship, entity2). Each element must be 4 words or less. Format the response as a valid Python list of tuples:\n\n{truncated_text}"}
            ]
        )

        content = response.choices[0].message.content
        print("API response:", content)

        # Try to extract tuples using regex
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, content)

        entities_relationships = []
        for match in matches:
            entity1 = summarize_relationship(match[0].strip().strip("'\""))
            relationship = summarize_relationship(match[1].strip().strip("'\""))
            entity2 = summarize_relationship(match[2].strip().strip("'\""))

            if entity1 and relationship and entity2:  # Only add if all parts are valid
                entities_relationships.append((entity1, relationship, entity2))

        print("Extracted and summarized entities and relationships:", entities_relationships)
        return entities_relationships

    except Exception as e:
        print(f"Error in API call or parsing: {e}")
        return []


def build_knowledge_graph(entities_and_relationships):
    G = nx.DiGraph()

    for item in entities_and_relationships:
        if len(item) == 3 and all(len(part.split()) <= 4 for part in item):
            entity1, relationship, entity2 = item
            # Capitalize the first letter of each word in entities
            entity1 = ' '.join(word.capitalize() for word in entity1.split())
            entity2 = ' '.join(word.capitalize() for word in entity2.split())
            G.add_node(entity1)
            G.add_node(entity2)
            G.add_edge(entity1, entity2, relationship=relationship)
        else:
            print(f"Skipping invalid relationship: {item}")

    print(f"Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G


def highlight_relationship(text, term):
    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    return pattern.sub(lambda m: f'<span class="highlight">{m.group()}</span>', text)


def summarize_relationship(text):
    words = text.split()
    if len(words) <= 4:
        return text

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a concise summarizer. Your task is to summarize the given phrase into 4 words or less, maintaining the core meaning."},
                {"role": "user", "content": f"Summarize this in 4 words or less: {text}"}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return summary if len(summary.split()) <= 4 else None
    except Exception as e:
        print(f"Error in summarizing: {e}")
        return None  # If we can't summarize, we'll return None and filter it out later


app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.H1("Wikipedia Knowledge Graph Visualizer", className='app-title'),
    html.Div([
        dcc.Input(
            id='wikipedia-url',
            type='text',
            placeholder='Enter Wikipedia URL...',
            className='url-input'
        ),
        html.Button('Generate', id='generate-button', n_clicks=0, className='generate-button'),
    ], className='input-container'),

    # Full app (initially hidden)
    html.Div([
        dcc.Loading(
            id="loading",
            type="circle",
            children=[dcc.Graph(id='knowledge-graph', className='knowledge-graph')]
        ),
        html.Div([
            html.Div([
                html.H3("Wikipedia Content", className='section-title'),
                html.Div(id='text-display', className='document-text')
            ], className='wikipedia-content-container'),
            html.Div([
                html.H3("Node Details", className='section-title'),
                dcc.Loading(
                    id="loading-node-details",
                    type="circle",
                    children=[html.Div(id='node-details', className='node-details')]
                ),
                html.Button('Generate Node Graph', id='generate-node-graph-button',
                            className='generate-node-graph-button')
            ], className='node-details-container'),
        ], className='bottom-container')
    ], id='full-app', style={'display': 'none'})
], className='app-container')

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Wikipedia Knowledge Graph Visualizer</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        <style>
            body, html {
                height: 100%;
                margin: 0;
                font-family: 'Roboto', sans-serif;
                background-color: #f0f4f8;
            }
            .app-container {
                display: flex;
                flex-direction: column;
                height: 100vh;
                padding: 20px;
                box-sizing: border-box;
            }
            #landing-page {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .app-title {
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 2.5em;
                text-align: center;
            }
            .input-container {
                display: flex;
                margin-bottom: 20px;
                justify-content: center;
                width: 100%;
            }
            .url-input {
                width: 60%;
                padding: 12px 20px;
                font-size: 16px;
                border: 2px solid #bdc3c7;
                border-radius: 50px;
                transition: all 0.3s ease;
            }
            .url-input:focus {
                outline: none;
                border-color: #3498db;
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3);
            }
            .generate-button, .generate-node-graph-button {
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                color: white;
                border: none;
                border-radius: 50px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .generate-button {
                background: linear-gradient(45deg, #3498db, #2980b9);
                margin-left: 10px;
            }
            .generate-button:hover {
                background: linear-gradient(45deg, #2980b9, #3498db);
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            }
            .generate-node-graph-button {
                background: linear-gradient(45deg, #2ecc71, #27ae60);
                margin-top: 15px;
            }
            .generate-node-graph-button:hover {
                background: linear-gradient(45deg, #27ae60, #2ecc71);
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            }
            .knowledge-graph {
                flex-grow: 1;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
                margin-bottom: 20px;
                min-height: 500px;
            }
            .bottom-container {
                display: flex;
                justify-content: space-between;
                height: 300px;
            }
            .node-details-container, .wikipedia-content-container {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                width: 48%;
                overflow-y: auto;
            }
            .section-title {
                color: #2c3e50;
                margin-top: 0;
            }
            .document-text {
                line-height: 1.6;
            }
            .highlight {
                background-color: #fff176;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
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


@app.callback(
    [Output('wikipedia-url', 'value'),
     Output('generate-button', 'n_clicks')],
    [Input('generate-node-graph-button', 'n_clicks')],
    [State('node-details', 'children')]
)
def update_url_and_generate(n_clicks, node_details):
    if n_clicks is None or not node_details or isinstance(node_details, str):
        raise PreventUpdate

    # Extract the node name from the node details
    node_name = node_details[0]['props']['children']

    # Generate Wikipedia URL for the node
    wikipedia_url = f"https://en.wikipedia.org/wiki/{node_name.replace(' ', '_')}"

    # Return the new URL and increment the generate button's n_clicks
    return wikipedia_url, n_clicks


def get_wikipedia_summary(title):
    try:
        return wikipedia.summary(title, sentences=3)  # Adjust the number of sentences as needed
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this topic."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def highlight_relationship(text, term):
    if not term:
        return text
    # Escape special regex characters in the term
    escaped_term = re.escape(term)
    # Create a case-insensitive pattern that matches the whole term
    pattern = re.compile(r'\b(' + escaped_term + r')\b', re.IGNORECASE)
    return pattern.sub(r'<span class="highlight">\1</span>', text)


def process_text(text):
    lines = text.split('\n')
    processed_text = [line.strip() for line in lines if line.strip()]
    highlighted_text = [f'<p>{line}</p>' for line in processed_text]
    return html.Div([
        dcc.Markdown(
            children='\n'.join(highlighted_text),
            dangerously_allow_html=True,
        )
    ], className='document-text')


def create_graph_figure(G):
    if not G:
        return {}

    # Find the main entity (node with most connections)
    main_entity = max(G.nodes(), key=lambda n: G.degree(n))

    # Create subgraph
    subgraph = nx.ego_graph(G, main_entity, radius=1)
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text = [], [], []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.capitalize())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            size=30,
            color='#1f77b4',
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=16)
    )

    edge_labels = []
    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_labels.append(
            go.Scatter(
                x=[(x0 + x1) / 2],
                y=[(y0 + y1) / 2],
                text=[edge[2]['relationship']],
                mode='text',
                textposition='middle center',
                textfont=dict(size=14, color='red'),
                hoverinfo='text',
                hoverlabel=dict(bgcolor='white'),
                showlegend=False
            )
        )

    figure = {
        'data': [edge_trace, node_trace] + edge_labels,
        'layout': go.Layout(
            title=f'Knowledge Graph centered on {main_entity}',
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
    }

    return figure


@app.callback(
    [Output('full-app', 'style'),
     Output('knowledge-graph', 'figure'),
     Output('text-display', 'children'),
     Output('node-details', 'children'),
     Output('generate-button', 'children'),
     Output('generate-node-graph-button', 'style')],
    [Input('generate-button', 'n_clicks'),
     Input('generate-node-graph-button', 'n_clicks'),
     Input('knowledge-graph', 'clickData')],
    [State('wikipedia-url', 'value'),
     State('node-details', 'children')]
)
def update_app(generate_clicks, node_graph_clicks, clickData, url, current_node_details):
    global G
    ctx = callback_context
    if not ctx.triggered:
        # Return default values if the callback is triggered on initial load
        return {'display': 'none'}, {}, "", "", "Generate Graph", {'display': 'none'}

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'generate-button':
        if not url:
            # Return current state if no URL is provided
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        file_content = extract_wikipedia_content(url)
        if file_content.startswith("Error:"):
            return {'display': 'none'}, {}, html.Div(file_content), "", "Generate Graph", {'display': 'none'}

        entities_relationships = extract_entities_and_relationships(file_content)
        G = build_knowledge_graph(entities_relationships)

        figure = create_graph_figure(G)
        highlighted_text = process_text(file_content)

        return {'display': 'block'}, figure, highlighted_text, "", "Generate Graph", {'display': 'none'}

    elif triggered_id == 'generate-node-graph-button':
        if not current_node_details or isinstance(current_node_details, str):
            # Return current state if no node is selected
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        clicked_node = current_node_details[0]['props']['children']
        new_url = f"https://en.wikipedia.org/wiki/{clicked_node.replace(' ', '_')}"

        file_content = extract_wikipedia_content(new_url)
        if file_content.startswith("Error:"):
            return {'display': 'none'}, {}, html.Div(file_content), "", "Generate Graph", {'display': 'none'}

        entities_relationships = extract_entities_and_relationships(file_content)
        G = build_knowledge_graph(entities_relationships)

        figure = create_graph_figure(G)
        highlighted_text = process_text(file_content)

        return {'display': 'block'}, figure, highlighted_text, "", "Generate Graph", {'display': 'none'}

    elif triggered_id == 'knowledge-graph':
        if not clickData:
            # Return current state if no node is clicked
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        clicked_node = clickData['points'][0]['text']

        details = [html.H4(clicked_node)]

        summary = get_wikipedia_summary(clicked_node)
        details.append(html.Div([
            html.H5("Wikipedia Summary:"),
            html.P(summary)
        ], style={'marginBottom': '20px', 'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'}))

        details.append(html.H5("Relationships:"))

        graph_node = next((node for node in G.nodes() if node.lower() == clicked_node.lower()), None)

        if graph_node:
            for neighbor in G.neighbors(graph_node):
                edge_data = G.get_edge_data(graph_node, neighbor)
                details.append(html.P(f"{edge_data['relationship']} {neighbor}"))

            if not list(G.neighbors(graph_node)):
                details.append(html.P("No connections found for this node."))
        else:
            details.append(html.P("This node is not present in the current graph."))

        wikipedia_url = f"https://en.wikipedia.org/wiki/{clicked_node.replace(' ', '_')}"
        details.append(html.P([
            "Wikipedia URL: ",
            html.A(wikipedia_url, href=wikipedia_url, target="_blank")
        ]))

        # Only update the node details and the 'Generate Node Graph' button visibility
        return dash.no_update, dash.no_update, dash.no_update, details, dash.no_update, {'display': 'block'}

    # If none of the conditions are met, return the current state
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True)

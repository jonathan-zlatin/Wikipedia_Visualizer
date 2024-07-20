import dash
from dash import dcc, html
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
    html.Div([
        html.H1("Wikipedia Knowledge Graph Visualizer", style={'textAlign': 'center', 'color': '#2c3e50'}),
        dcc.Input(
            id='wikipedia-url',
            type='text',
            placeholder='Enter Wikipedia URL...',
            style={'width': '100%', 'margin': '10px 0', 'padding': '10px', 'borderRadius': '5px',
                   'border': '1px solid #ddd'}
        ),
        html.Button('Generate Graph', id='generate-button', n_clicks=0,
                    style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px 20px',
                           'borderRadius': '5px', 'cursor': 'pointer'}),
        dcc.Dropdown(
            id='relationship-filter',
            multi=True,
            placeholder='Select relationships to filter...',
            style={'width': '100%', 'marginTop': '10px'}
        ),
        dcc.Loading(
            id="loading",
            type="default",
            children=[dcc.Graph(id='knowledge-graph')]
        ),
        html.Div([
            html.H3("Node Details"),
            dcc.Loading(
                id="loading-node-details",
                type="default",
                children=[html.Div(id='node-details')]
            ),
            html.Button('Generate Graph for This Node', id='generate-node-graph-button',
                        style={'display': 'none', 'backgroundColor': '#2ecc71', 'color': 'white', 'border': 'none',
                               'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer', 'marginTop': '10px'})
        ], style={'width': '30%', 'float': 'right', 'padding': '10px', 'border': '1px solid #ddd',
                  'borderRadius': '5px', 'backgroundColor': 'rgba(255, 255, 255, 0.9)'}),
        html.Div([
            html.H2("Wikipedia Content"),
            html.Div(id='text-display', className='document-text')
        ], style={'margin-top': '20px', 'clear': 'both'})
    ], id='content-container')
])

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body, html {
                height: 100%;
                margin: 0;
                font-family: Arial, sans-serif;
            }
            #background-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                background-image: url('light-bg.jpg');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                opacity: 0.3; /* Adjust this value to change the background opacity */
            }
            #content-container {
                position: relative;
                z-index: 1;
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
                border-radius: 10px;
                margin: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .document-text {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-height: 500px;
                overflow-y: auto;
                padding: 20px;
                background-color: rgba(249, 249, 249, 0.9);
                border: 1px solid #e0e0e0;
                border-radius: 5px;
            }
            .document-text h1, .document-text h2, .document-text h3 {
                color: #2c3e50;
            }
            .document-text p {
                margin-bottom: 15px;
            }
            .highlight {
                background-color: #fff176;
                padding: 2px 4px;
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div id="background-container"></div>
        <div id="content-container">
            {%app_entry%}
        </div>
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


@app.callback(
    [Output('relationship-filter', 'options'),
     Output('generate-button', 'children')],
    [Input('generate-button', 'n_clicks'),
     Input('generate-node-graph-button', 'n_clicks')],
    [State('wikipedia-url', 'value')]
)
def update_output(generate_clicks, node_graph_clicks, url):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    global file_content, G
    if ctx.triggered_id == 'generate-button' or ctx.triggered_id == 'generate-node-graph-button':
        if url:
            file_content = extract_wikipedia_content(url)

            if file_content.startswith("Error:"):
                return [], "Generate Graph"

            entities_relationships = extract_entities_and_relationships(file_content)
            print("Extracted entities and relationships:", entities_relationships)

            G = build_knowledge_graph(entities_relationships)

            relationship_options = [{'label': data['relationship'], 'value': data['relationship']}
                                    for _, _, data in G.edges(data=True)]

            return relationship_options, "Graph Generated"

    return [], "Generate Graph"


@app.callback(
    [Output('knowledge-graph', 'figure'),
     Output('text-display', 'children')],
    [Input('generate-button', 'n_clicks'),
     Input('relationship-filter', 'value')],
    [State('wikipedia-url', 'value')]
)
def update_graph(generate_clicks, selected_relationships, url):
    global file_content
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'generate-button' and url:
        file_content = extract_wikipedia_content(url)
        if file_content.startswith("Error:"):
            return {}, html.Div(file_content)

    # Process the text
    lines = file_content.split('\n')
    processed_text = [f'<p>{line.strip()}</p>' for line in lines if line.strip()]
    highlighted_text = '\n'.join(processed_text)

    # Find the main entity (node with most connections)
    if G:
        main_entity = max(G.nodes(), key=lambda n: G.degree(n))
    else:
        return dash.no_update, html.Div([
            dcc.Markdown(
                children=highlighted_text,
                dangerously_allow_html=True,
            )
        ], className='document-text')

    print(f"Creating subgraph for main entity: {main_entity}")
    subgraph = nx.ego_graph(G, main_entity, radius=1)

    if selected_relationships:
        subgraph = nx.DiGraph(((u, v, d) for u, v, d in subgraph.edges(data=True)
                               if d['relationship'] in selected_relationships))

    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=2, color='#888'),  # Increase width here
        hoverinfo='none', mode='lines'
    )

    node_x, node_y, node_text = [], [], []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.capitalize())

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        marker=dict(showscale=False, size=30, color='#1f77b4'),  # Increase size here
        text=node_text, textposition="top center",
        textfont=dict(size=16)  # Increase text size here
    )

    edge_labels = []
    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_labels.append(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            text=[edge[2]['relationship']],
            mode='text',
            textposition='middle center',
            textfont=dict(size=14, color='red'),
            hoverinfo='text',
            hoverlabel=dict(bgcolor='white'),
            showlegend=False
        ))

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

    print("Returning from update_graph function")
    return figure, html.Div([
        dcc.Markdown(
            children=highlighted_text,
            dangerously_allow_html=True,
        )
    ], className='document-text')


def get_wikipedia_summary(title):
    try:
        return wikipedia.summary(title, sentences=3)  # Adjust the number of sentences as needed
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this topic."
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.callback(
    [Output('node-details', 'children'),
     Output('generate-node-graph-button', 'style')],
    Input('knowledge-graph', 'clickData')
)
def display_node_details(clickData):
    if not clickData:
        return "Click on a node to see details", {'display': 'none'}

    # Extract the node name, removing any additional information
    full_text = clickData['points'][0]['text']
    clicked_node = full_text.strip()

    # Find the corresponding node in the graph (case-insensitive)
    node = next((n for n in G.nodes() if n.lower() == clicked_node.lower()), None)

    if node is None:
        return f"Node '{clicked_node}' not found in the graph.", {'display': 'none'}

    details = [html.H4(node)]

    # Fetch and display Wikipedia summary
    summary = get_wikipedia_summary(node)
    details.append(html.Div([
        html.H5("Wikipedia Summary:"),
        html.P(summary)
    ], style={'marginBottom': '20px', 'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'}))

    # Display relationships
    details.append(html.H5("Relationships:"))
    for neighbor in G.neighbors(node):
        edge_data = G.get_edge_data(node, neighbor)
        details.append(html.P(f"{edge_data['relationship']} {neighbor}"))

    if not list(G.neighbors(node)):  # If there are no neighbors
        details.append(html.P("No connections found for this node."))

    # Add clickable Wikipedia URL
    wikipedia_url = f"https://en.wikipedia.org/wiki/{node.replace(' ', '_')}"
    details.append(html.P([
        "Wikipedia URL: ",
        html.A(wikipedia_url, href=wikipedia_url, target="_blank")
    ]))

    return details, {'display': 'block', 'marginTop': '10px'}


if __name__ == '__main__':
    app.run_server(debug=True)

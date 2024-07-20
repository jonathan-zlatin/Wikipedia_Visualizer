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
        title = url.split("/wiki/")[-1].replace("_", " ")
        page = wikipedia.page(title)
        summary = wikipedia.summary(title, sentences=10)
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": BACKGROUND_STORY},
                {"role": "user", "content": f"Extract entities and relationships from the following text. Return the result as a list of tuples, where each tuple contains three elements (entity1, relationship, entity2). Each element must be 4 words or less. Format the response as a valid Python list of tuples:\n\n{truncated_text}"}
            ]
        )
        content = response.choices[0].message.content
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
        matches = re.findall(pattern, content)
        entities_relationships = []
        for match in matches:
            entity1 = summarize_relationship(match[0].strip().strip("'\""))
            relationship = summarize_relationship(match[1].strip().strip("'\""))
            entity2 = summarize_relationship(match[2].strip().strip("'\""))
            if entity1 and relationship and entity2:
                entities_relationships.append((entity1, relationship, entity2))
        return entities_relationships
    except Exception as e:
        print(f"Error in API call or parsing: {e}")
        return []

def build_knowledge_graph(entities_and_relationships):
    G = nx.DiGraph()
    for item in entities_and_relationships:
        if len(item) == 3 and all(len(part.split()) <= 4 for part in item):
            entity1, relationship, entity2 = item
            entity1 = ' '.join(word.capitalize() for word in entity1.split())
            entity2 = ' '.join(word.capitalize() for word in entity2.split())
            G.add_node(entity1)
            G.add_node(entity2)
            G.add_edge(entity1, entity2, relationship=relationship)
    return G

def summarize_relationship(text):
    words = text.split()
    if len(words) <= 4:
        return text
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise summarizer. Your task is to summarize the given phrase into 4 words or less, maintaining the core meaning."},
                {"role": "user", "content": f"Summarize this in 4 words or less: {text}"}
            ]
        )
        summary = response.choices[0].message.content.strip()
        return summary if len(summary.split()) <= 4 else None
    except Exception as e:
        print(f"Error in summarizing: {e}")
        return None

def get_wikipedia_summary(title):
    try:
        return wikipedia.summary(title, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this topic."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def highlight_relationship(text, term):
    if not term:
        return text
    escaped_term = re.escape(term)
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
    main_entity = max(G.nodes(), key=lambda n: G.degree(n))
    subgraph = nx.ego_graph(G, main_entity, radius=1)
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text = [], [], []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.capitalize())
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        marker=dict(showscale=False, size=30, color='#1f77b4'),
        text=node_text, textposition="top center", textfont=dict(size=16)
    )
    edge_labels = []
    for edge in subgraph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_labels.append(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], text=[edge[2]['relationship']],
            mode='text', textposition='middle center', textfont=dict(size=14, color='red'),
            hoverinfo='text', hoverlabel=dict(bgcolor='white'), showlegend=False
        ))
    return {
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

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Load the HTML layout
with open('layout.html', 'r') as file:
    app.index_string = file.read()

# Define the app layout
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

@app.callback(
    [Output('wikipedia-url', 'value'),
     Output('generate-button', 'n_clicks')],
    [Input('generate-node-graph-button', 'n_clicks')],
    [State('node-details', 'children')]
)
def update_url_and_generate(n_clicks, node_details):
    if n_clicks is None or not node_details or isinstance(node_details, str):
        raise PreventUpdate
    node_name = node_details[0]['props']['children']
    wikipedia_url = f"https://en.wikipedia.org/wiki/{node_name.replace(' ', '_')}"
    return wikipedia_url, n_clicks

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
        return {'display': 'none'}, dash.no_update, "", "", "Generate Graph", {'display': 'none'}

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id in ['generate-button', 'generate-node-graph-button']:
        if triggered_id == 'generate-button' and not url:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if triggered_id == 'generate-node-graph-button':
            if not current_node_details or isinstance(current_node_details, str):
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            clicked_node = current_node_details[0]['props']['children']
            url = f"https://en.wikipedia.org/wiki/{clicked_node.replace(' ', '_')}"

        file_content = extract_wikipedia_content(url)
        if file_content.startswith("Error:"):
            return {'display': 'none'}, dash.no_update, html.Div(file_content), "", "Generate Graph", {'display': 'none'}

        entities_relationships = extract_entities_and_relationships(file_content)
        G = build_knowledge_graph(entities_relationships)

        figure = create_graph_figure(G)
        highlighted_text = process_text(file_content)

        return {'display': 'block'}, figure, highlighted_text, "", "Generate Graph", {'display': 'none'}

    elif triggered_id == 'knowledge-graph':
        if not clickData:
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

        return dash.no_update, dash.no_update, dash.no_update, details, dash.no_update, {'display': 'block'}

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
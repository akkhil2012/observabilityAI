import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import io
import base64
from datetime import datetime
import json
import random

# Configure Streamlit page
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend methods for anomaly detection
class AnomalyDetector:
    def __init__(self):
        self.subdomain_data = {
            "RelAPG": {"nodes": 3, "connections": [(0, 1), (1, 2), (0, 2)]},
            "IdentityAPG": {"nodes": 3, "connections": [(0, 1), (1, 2)]},
            "PreferenceAPG": {"nodes": 3, "connections": [(0, 1), (1, 2), (2, 0)]}
        }

    def detect_anomalies(self, subdomain, node_id, features_config):
        """
        Mock anomaly detection method
        Returns count of anomalies and feature values
        """
        # Simulate anomaly detection based on selected features and thresholds
        anomaly_count = random.randint(5, 50)
        feature_values = {}

        for feature, threshold in features_config.items():
            if feature == "ResponseTime":
                feature_values[feature] = round(random.uniform(100, 2000), 2)
            elif feature == "Client":
                feature_values[feature] = f"Client_{random.randint(1, 100)}"
            elif feature == "ErrorCode":
                feature_values[feature] = random.choice([200, 404, 500, 503])
            elif feature == "ErrorMessage":
                feature_values[feature] = random.choice([
                    "Connection timeout", "Invalid request", "Server error", "Not found"
                ])

        return {
            "anomaly_count": anomaly_count,
            "feature_values": feature_values,
            "threshold_config": features_config,
            "subdomain": subdomain,
            "node_id": node_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def generate_pdf_report(analysis_results):
    """Generate PDF report for anomaly analysis results"""
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Anomaly Detection Report")

    # Metadata
    p.setFont("Helvetica", 12)
    y_position = height - 100

    p.drawString(50, y_position, f"Subdomain: {analysis_results['subdomain']}")
    y_position -= 20
    p.drawString(50, y_position, f"Node ID: {analysis_results['node_id']}")
    y_position -= 20
    p.drawString(50, y_position, f"Timestamp: {analysis_results['timestamp']}")
    y_position -= 20
    p.drawString(50, y_position, f"Anomaly Count: {analysis_results['anomaly_count']}")
    y_position -= 40

    # Feature values
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y_position, "Feature Analysis:")
    y_position -= 30

    p.setFont("Helvetica", 10)
    for feature, value in analysis_results['feature_values'].items():
        threshold = analysis_results['threshold_config'][feature]
        p.drawString(70, y_position, f"{feature}: {value} (Threshold: {threshold})")
        y_position -= 15

    p.save()
    buffer.seek(0)
    return buffer

def create_topological_graph(subdomain, node_configs=None):
    """Create interactive topological graph using plotly"""
    detector = AnomalyDetector()
    graph_data = detector.subdomain_data[subdomain]

    # Create networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(graph_data["nodes"]))
    G.add_edges_from(graph_data["connections"])

    # Get node positions
    pos = nx.spring_layout(G, seed=42)

    # Create plotly figure
    fig = go.Figure()

    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Check if node has been configured
        if node_configs and node in node_configs:
            node_text.append(f"Node {node}<br>Configured ✓")
            node_colors.append('lightgreen')
        else:
            node_text.append(f"Node {node}<br>Click to configure")
            node_colors.append('lightblue')

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"Node {i}" for i in range(len(node_x))],
        textposition="middle center",
        hovertext=node_text,
        marker=dict(
            size=50,
            color=node_colors,
            line=dict(width=2, color='darkblue')
        )
    ))

    fig.update_layout(
        title=f"Topological Graph - {subdomain}",
        title_font_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Click on nodes to configure anomaly detection parameters",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=12)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig

# Initialize session state
if 'selected_subdomain' not in st.session_state:
    st.session_state.selected_subdomain = None
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False
if 'node_configs' not in st.session_state:
    st.session_state.node_configs = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'current_node' not in st.session_state:
    st.session_state.current_node = None

# Main dashboard
st.title("🔍 Anomaly Detection Dashboard")
st.markdown("---")

# Sidebar for subdomain selection
with st.sidebar:
    st.header("Configuration")

    # SubDomain dropdown
    subdomain_options = ["RelAPG", "IdentityAPG", "PreferenceAPG"]
    selected_subdomain = st.selectbox(
        "SubDomain",
        options=["Select SubDomain..."] + subdomain_options,
        key="subdomain_selector"
    )

    if selected_subdomain != "Select SubDomain..." and selected_subdomain != st.session_state.selected_subdomain:
        st.session_state.selected_subdomain = selected_subdomain
        st.session_state.show_graph = True
        st.session_state.node_configs = {}
        st.session_state.analysis_results = {}
        st.rerun()

# Main content area
if st.session_state.show_graph and st.session_state.selected_subdomain:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Topological Graph - {st.session_state.selected_subdomain}")

        # Create and display graph
        fig = create_topological_graph(
            st.session_state.selected_subdomain, 
            st.session_state.node_configs
        )

        # Display graph with click handling
        selected_points = st.plotly_chart(fig, use_container_width=True, key="topology_graph")

        # Node selection buttons
        st.markdown("### Node Configuration")
        node_cols = st.columns(3)

        for i in range(3):
            with node_cols[i]:
                if st.button(f"Configure Node {i}", key=f"node_btn_{i}"):
                    st.session_state.current_node = i

    with col2:
        if st.session_state.current_node is not None:
            st.subheader("Features to determine anomalies")

            node_id = st.session_state.current_node

            # Primary feature selection
            st.markdown("**Select Features:**")
            primary_options = ["ResponseTime", "Client", "ErrorCode", "ErrorMessage"]
            selected_features = st.multiselect(
                "Primary Features",
                primary_options,
                key=f"features_{node_id}"
            )

            # Secondary threshold selection for each selected feature
            feature_config = {}
            if selected_features:
                st.markdown("**Set Thresholds:**")
                secondary_options = [0.01, 0.05, 0.10, 0.20]

                for feature in selected_features:
                    threshold = st.selectbox(
                        f"Threshold for {feature}",
                        secondary_options,
                        key=f"threshold_{feature}_{node_id}"
                    )
                    feature_config[feature] = threshold

            # Trigger button
            if st.button("🚀 Trigger Analysis", key=f"trigger_{node_id}"):
                if feature_config:
                    detector = AnomalyDetector()
                    results = detector.detect_anomalies(
                        st.session_state.selected_subdomain,
                        node_id,
                        feature_config
                    )

                    st.session_state.node_configs[node_id] = feature_config
                    st.session_state.analysis_results[node_id] = results

                    st.success(f"✅ Analysis completed for Node {node_id}")
                    st.rerun()
                else:
                    st.error("Please select at least one feature and set thresholds")

            # Display results if available
            if node_id in st.session_state.analysis_results:
                results = st.session_state.analysis_results[node_id]

                st.markdown("### Analysis Results")
                st.metric("Anomaly Count", results["anomaly_count"])

                st.markdown("**Feature Values:**")
                for feature, value in results["feature_values"].items():
                    st.text(f"{feature}: {value}")

                # PDF download
                if st.button(f"📄 Download PDF Report", key=f"pdf_{node_id}"):
                    pdf_buffer = generate_pdf_report(results)
                    st.download_button(
                        label="Download Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"anomaly_report_node_{node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key=f"download_{node_id}"
                    )

                # Continue/Abort flow
                st.markdown("### Flow Control")
                flow_col1, flow_col2 = st.columns(2)

                with flow_col1:
                    if st.button("➡️ Continue Flow", key=f"continue_{node_id}"):
                        next_node = (node_id + 1) % 3
                        st.session_state.current_node = next_node
                        st.info(f"Continuing to Node {next_node}")
                        st.rerun()

                with flow_col2:
                    if st.button("⛔ Abort Flow", key=f"abort_{node_id}"):
                        st.session_state.current_node = None
                        st.warning("Flow aborted")
                        st.rerun()

# Display summary
if st.session_state.node_configs:
    st.markdown("---")
    st.subheader("Configuration Summary")

    summary_data = []
    for node_id, config in st.session_state.node_configs.items():
        for feature, threshold in config.items():
            summary_data.append({
                "Node": f"Node {node_id}",
                "Feature": feature,
                "Threshold": threshold,
                "Status": "✅ Configured"
            })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Anomaly Detection Dashboard - Real-time Topological Analysis*")

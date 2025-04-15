import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

from utils import preprocess_data

def run_unsupervised_learning(data):
    """Run the unsupervised learning page."""
    st.header("Unsupervised Learning")
    
    # Create tabs for clustering and association rules
    tab1, tab2 = st.tabs(["Clustering", "Association Rules"])
    
    with tab1:
        run_clustering(data)
    
    with tab2:
        run_association_rules(data)

def run_clustering(data):
    """Run clustering models on the data."""
    st.subheader("Clustering Analysis")
    st.write("""
    Clustering groups similar data points together. We'll identify natural groupings among students 
    based on their characteristics and test scores.
    """)
    
    # Feature selection
    st.subheader("Feature Selection")
    
    feature_cols = list(data.columns)
    selected_features = st.multiselect(
        "Select features for clustering:",
        options=feature_cols,
        default=['math score', 'reading score', 'writing score']
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
        return
    
    # Preprocessing
    st.subheader("Data Preprocessing")
    
    # Get data for clustering
    cluster_data = data[selected_features].copy()
    
    # Handle categorical features
    categorical_cols = cluster_data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = [col for col in selected_features if col not in categorical_cols]
    
    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        cluster_data[col] = le.fit_transform(cluster_data[col])
        encoders[col] = le
    
    # Scaling
    perform_scaling = st.checkbox("Scale features (recommended)", value=True)
    
    if perform_scaling:
        scaler = StandardScaler()
        cluster_data[selected_features] = scaler.fit_transform(cluster_data[selected_features])
    
    # Dimensionality reduction for visualization
    st.subheader("Dimensionality Reduction")
    
    perform_pca = st.checkbox("Apply PCA for visualization", value=True)
    
    if perform_pca and len(selected_features) > 2:
        n_components = min(3, len(selected_features))
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(cluster_data[selected_features])
        
        # Create new DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Display explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        st.write("Explained variance ratio by principal components:")
        
        variance_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(n_components)],
            'Explained Variance (%)': [f'{var:.2%}' for var in explained_variance],
            'Cumulative Variance (%)': [f'{var:.2%}' for var in cumulative_variance]
        })
        
        st.dataframe(variance_df)
        
        # Show PCA loadings
        st.write("PCA Feature Loadings:")
        
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=selected_features
        )
        
        st.dataframe(loadings)
    else:
        pca_df = None
    
    # Clustering model selection
    st.subheader("Clustering Method")
    
    clustering_methods = {
        "K-Means": KMeans,
        "DBSCAN": DBSCAN,
        "Hierarchical Clustering": AgglomerativeClustering
    }
    
    selected_method = st.selectbox(
        "Select clustering method:",
        options=list(clustering_methods.keys())
    )
    
    # Clustering parameters
    st.subheader("Clustering Parameters")
    
    if selected_method == "K-Means":
        n_clusters = st.slider("Number of clusters (K):", min_value=2, max_value=10, value=3, step=1)
        random_state = st.slider("Random state:", min_value=0, max_value=100, value=42, step=1)
        clustering_params = {
            'n_clusters': n_clusters,
            'random_state': random_state
        }
    
    elif selected_method == "DBSCAN":
        eps = st.slider("Epsilon (neighborhood size):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.slider("Minimum samples:", min_value=2, max_value=20, value=5, step=1)
        clustering_params = {
            'eps': eps,
            'min_samples': min_samples
        }
    
    elif selected_method == "Hierarchical Clustering":
        n_clusters = st.slider("Number of clusters:", min_value=2, max_value=10, value=3, step=1)
        linkage = st.selectbox("Linkage method:", ["ward", "complete", "average", "single"])
        clustering_params = {
            'n_clusters': n_clusters,
            'linkage': linkage
        }
    
    # Perform clustering
    if st.button("Perform Clustering"):
        with st.spinner("Performing clustering..."):
            # Initialize and fit clustering model
            model = clustering_methods[selected_method](**clustering_params)
            labels = model.fit_predict(cluster_data[selected_features])
            
            # Add cluster labels to original data
            result_data = data.copy()
            result_data['Cluster'] = labels
            
            # Display clustering results
            st.subheader("Clustering Results")
            
            # Show cluster distribution
            fig = plt.figure(figsize=(10, 6))
            cluster_counts = result_data['Cluster'].value_counts().sort_index()
            
            # Handle case where cluster labels might start from -1 (noise points in DBSCAN)
            if -1 in cluster_counts.index:
                cluster_labels = ['Noise'] + [f'Cluster {i}' for i in range(cluster_counts.index.max()+1)]
            else:
                cluster_labels = [f'Cluster {i}' for i in range(cluster_counts.index.max()+1)]
            
            plt.bar(cluster_labels, cluster_counts.values)
            plt.xlabel('Cluster')
            plt.ylabel('Number of Students')
            plt.title('Distribution of Students Across Clusters')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
            
            # Visualize clusters
            st.subheader("Cluster Visualization")
            
            if perform_pca and pca_df is not None and n_components >= 2:
                # Add cluster labels to PCA dataframe
                pca_df['Cluster'] = labels
                
                # 2D visualization
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Cluster',
                    title='Cluster Visualization (PCA)',
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 3D visualization if we have at least 3 components
                if n_components >= 3:
                    fig = px.scatter_3d(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Cluster',
                        title='3D Cluster Visualization (PCA)',
                        labels={
                            'PC1': 'Principal Component 1',
                            'PC2': 'Principal Component 2',
                            'PC3': 'Principal Component 3'
                        },
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            elif len(numerical_cols) >= 2:
                # If no PCA but we have numerical features, visualize with the first two
                viz_cols = numerical_cols[:2]
                
                fig = px.scatter(
                    result_data,
                    x=viz_cols[0],
                    y=viz_cols[1],
                    color='Cluster',
                    title=f'Cluster Visualization ({viz_cols[0]} vs {viz_cols[1]})',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster profiling - show characteristics of each cluster
            st.subheader("Cluster Profiling")
            
            # Exclude Cluster column for profiling
            profile_cols = [col for col in result_data.columns if col != 'Cluster']
            
            # Select columns for profiling
            profile_features = st.multiselect(
                "Select features for cluster profiling:",
                options=profile_cols,
                default=numerical_cols
            )
            
            if profile_features:
                # Compute cluster profiles for numerical features
                num_profile_features = [f for f in profile_features if f in numerical_cols]
                
                if num_profile_features:
                    profiles = result_data.groupby('Cluster')[num_profile_features].mean()
                    
                    # Create radar chart for numerical features
                    st.write("Average Values by Cluster:")
                    st.dataframe(profiles)
                    
                    # Normalize the data for radar chart
                    scaler = MinMaxScaler()
                    profiles_norm = pd.DataFrame(
                        scaler.fit_transform(profiles),
                        columns=profiles.columns,
                        index=profiles.index
                    )
                    
                    # Create radar chart using Plotly
                    fig = go.Figure()
                    
                    for cluster in profiles_norm.index:
                        cluster_label = f"Cluster {cluster}"
                        if cluster == -1:
                            cluster_label = "Noise"
                        
                        fig.add_trace(go.Scatterpolar(
                            r=profiles_norm.loc[cluster].values,
                            theta=profiles_norm.columns,
                            fill='toself',
                            name=cluster_label
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=True,
                        title="Cluster Profiles (Normalized)"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Compute cluster profiles for categorical features
                cat_profile_features = [f for f in profile_features if f in categorical_cols]
                
                if cat_profile_features:
                    # For each categorical feature, show distribution within clusters
                    for feature in cat_profile_features:
                        st.write(f"Distribution of {feature} by Cluster:")
                        
                        # Get cross-tabulation of feature and cluster
                        cross_tab = pd.crosstab(
                            result_data['Cluster'],
                            result_data[feature],
                            normalize='index'
                        ) * 100  # Convert to percentage
                        
                        # Create a grouped bar chart
                        fig = px.bar(
                            cross_tab.reset_index().melt(id_vars='Cluster'),
                            x='Cluster',
                            y='value',
                            color='variable',
                            barmode='group',
                            title=f'Distribution of {feature} by Cluster (%)',
                            labels={'value': 'Percentage', 'variable': feature}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

def run_association_rules(data):
    """Run association rule mining on the data."""
    st.subheader("Association Rule Mining")
    st.write("""
    Association rule mining finds interesting relationships between variables in the dataset.
    For example, we can discover patterns like "students with feature X tend to have feature Y".
    """)
    
    # Prepare the data for association rule mining
    # We need to convert the data into binary format (one-hot encoding)
    
    # Feature selection
    st.subheader("Feature Selection")
    
    feature_cols = list(data.columns)
    selected_features = st.multiselect(
        "Select features for association rule mining:",
        options=feature_cols,
        default=['gender', 'race/ethnicity', 'lunch', 'test preparation course']
    )
    
    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return
    
    # Define thresholds for mining
    st.subheader("Mining Parameters")
    
    min_support = st.slider(
        "Minimum support (frequency of itemsets):",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01
    )
    
    min_confidence = st.slider(
        "Minimum confidence (reliability of rules):",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Add options for binning numerical features
    st.subheader("Numerical Feature Binning")
    
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_numerical = [col for col in selected_features if col in numerical_cols]
    
    binning_info = {}
    
    for col in selected_numerical:
        st.write(f"Binning for {col}:")
        n_bins = st.slider(
            f"Number of bins for {col}:",
            min_value=2,
            max_value=10,
            value=3,
            key=f"bins_{col}"
        )
        
        # Get min and max values for the column
        min_val = data[col].min()
        max_val = data[col].max()
        
        # Create bin edges
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        
        # Create bin labels
        bin_labels = [f"{col}_{i+1}" for i in range(n_bins)]
        
        binning_info[col] = {
            'edges': bin_edges,
            'labels': bin_labels
        }
    
    # Process the data and mine association rules
    if st.button("Mine Association Rules"):
        with st.spinner("Mining association rules..."):
            # Prepare the data
            trans_data = data[selected_features].copy()
            
            # Convert categorical features to string for one-hot encoding
            categorical_cols = trans_data.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                trans_data[col] = col + "_" + trans_data[col].astype(str)
            
            # Bin numerical features
            for col, info in binning_info.items():
                trans_data[col] = pd.cut(
                    trans_data[col],
                    bins=info['edges'],
                    labels=info['labels'],
                    include_lowest=True
                )
            
            # Create one-hot encoded DataFrame
            encoded_cols = []
            for col in trans_data.columns:
                dummy = pd.get_dummies(trans_data[col], prefix='', prefix_sep='')
                encoded_cols.append(dummy)
            
            oht_df = pd.concat(encoded_cols, axis=1)
            
            # Mine frequent itemsets
            frequent_itemsets = apriori(
                oht_df,
                min_support=min_support,
                use_colnames=True
            )
            
            if frequent_itemsets.empty:
                st.warning("No frequent itemsets found with the current minimum support. Try lowering the threshold.")
                return
            
            # Generate association rules
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=min_confidence
            )
            
            if rules.empty:
                st.warning("No association rules found with the current minimum confidence. Try lowering the threshold.")
                return
            
            # Display results
            st.subheader("Association Rules")
            
            # Sort rules by lift
            sorted_rules = rules.sort_values('lift', ascending=False)
            
            # Function to convert frozenset to string
            def format_itemset(itemset):
                return ', '.join(list(itemset))
            
            # Create a more readable DataFrame for display
            display_rules = pd.DataFrame({
                'Antecedent': sorted_rules['antecedents'].apply(format_itemset),
                'Consequent': sorted_rules['consequents'].apply(format_itemset),
                'Support': sorted_rules['support'],
                'Confidence': sorted_rules['confidence'],
                'Lift': sorted_rules['lift']
            })
            
            st.dataframe(display_rules)
            
            # Visualization of top rules by lift
            st.subheader("Top Rules by Lift")
            
            top_n = min(20, len(display_rules))
            top_rules = display_rules.head(top_n)
            
            # Create a user-friendly label that shows the rule
            top_rules['Rule'] = top_rules.apply(
                lambda x: f"{x['Antecedent']} → {x['Consequent']}", axis=1
            )
            
            fig = px.bar(
                top_rules,
                x='Lift',
                y='Rule',
                orientation='h',
                title=f'Top {top_n} Rules by Lift',
                color='Lift',
                color_continuous_scale=px.colors.sequential.Viridis,
                height=max(400, top_n * 25)  # Adjust height based on number of rules
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bubble chart for visualizing support, confidence, and lift
            st.subheader("Rule Visualization (Support, Confidence, Lift)")
            
            fig = px.scatter(
                display_rules,
                x='Support',
                y='Confidence',
                size='Lift',
                hover_name='Antecedent',  # Changed from 'Rule' to existing column 'Antecedent'
                title="Association Rules (Bubble Size = Lift)",
                color='Lift',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network graph of rules
            st.subheader("Rule Network (Top Rules)")
            
            # Create a simplified network from top rules
            top_n_network = min(10, len(display_rules))
            network_rules = display_rules.head(top_n_network)
            
            # Extract all unique items from antecedents and consequents
            all_items = set()
            for _, row in network_rules.iterrows():
                ant_items = row['Antecedent'].split(', ')
                cons_items = row['Consequent'].split(', ')
                all_items.update(ant_items)
                all_items.update(cons_items)
            
            # Create nodes
            nodes = pd.DataFrame({
                'id': list(range(len(all_items))),
                'name': list(all_items)
            })
            
            # Create a map of item names to node IDs
            name_to_id = {name: i for i, name in enumerate(nodes['name'])}
            
            # Create edges
            edges = []
            for _, row in network_rules.iterrows():
                ant_items = row['Antecedent'].split(', ')
                cons_items = row['Consequent'].split(', ')
                for ant in ant_items:
                    for con in cons_items:
                        edges.append({
                            'source': name_to_id[ant],
                            'target': name_to_id[con],
                            'value': row['Lift']
                        })
            
            # Create a DataFrame for edges
            edges_df = pd.DataFrame(edges)
            
            # Create network graph using Plotly
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Random positions for nodes
            np.random.seed(42)
            node_positions = {i: (np.random.rand() * 2 - 1, np.random.rand() * 2 - 1) for i in range(len(nodes))}
            
            # Add edges to trace
            for _, edge in edges_df.iterrows():
                x0, y0 = node_positions[edge['source']]
                x1, y1 = node_positions[edge['target']]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            node_trace = go.Scatter(
                x=[pos[0] for pos in node_positions.values()],
                y=[pos[1] for pos in node_positions.values()],
                mode='markers+text',
                text=nodes['name'],
                textposition='bottom center',
                marker=dict(
                    showscale=True,
                    colorscale='Viridis',
                    size=15,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left'
                    ),
                    line=dict(width=2)
                )
            )
            
            # Count connections for each node
            connections = {i: 0 for i in range(len(nodes))}
            for _, edge in edges_df.iterrows():
                connections[edge['source']] += 1
                connections[edge['target']] += 1
            
            node_adjacencies = [connections[i] for i in range(len(nodes))]
            node_trace.marker.color = node_adjacencies
            
            # Create the figure
            fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title='Association Rules Network',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            ))
            
            st.plotly_chart(fig, use_container_width=True)

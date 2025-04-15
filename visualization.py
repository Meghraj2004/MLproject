import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def run_data_exploration(data):
    """Run the data exploration and visualization page."""
    st.header("Data Exploration and Visualization")
    
    # Display dataset overview
    st.subheader("Dataset Overview")
    
    # Show the first few rows of the dataset
    with st.expander("Preview Dataset"):
        st.dataframe(data.head())
    
    # Display basic dataset information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Rows", data.shape[0])
    
    with col2:
        st.metric("Number of Columns", data.shape[1])
    
    with col3:
        n_missing = data.isna().sum().sum()
        st.metric("Missing Values", n_missing)
    
    # Display dataset summary
    with st.expander("Dataset Summary Statistics"):
        st.dataframe(data.describe())
    
    # Display data types
    with st.expander("Column Data Types"):
        dtypes_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes.values
        })
        st.dataframe(dtypes_df)
    
    # Feature distribution tab
    st.subheader("Feature Distributions")
    
    # Separate categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Categorical variable distribution
    if categorical_cols:
        st.write("### Categorical Variables")
        
        # Select a categorical feature to visualize
        cat_feature = st.selectbox(
            "Select a categorical feature:",
            options=categorical_cols
        )
        
        # Create a count plot
        fig = px.histogram(
            data,
            x=cat_feature,
            title=f"Distribution of {cat_feature}",
            color=cat_feature,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display value counts
        st.write(f"Value counts for {cat_feature}:")
        st.dataframe(data[cat_feature].value_counts())
    
    # Numerical variable distribution
    if numerical_cols:
        st.write("### Numerical Variables")
        
        # Select a numerical feature to visualize
        num_feature = st.selectbox(
            "Select a numerical feature:",
            options=numerical_cols
        )
        
        # Create a histogram with density plot
        fig = px.histogram(
            data,
            x=num_feature,
            title=f"Distribution of {num_feature}",
            marginal="box",
            histnorm="probability density",
            color_discrete_sequence=['#3366CC']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        st.write(f"Statistics for {num_feature}:")
        
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum'],
            'Value': [
                round(data[num_feature].mean(), 2),
                round(data[num_feature].median(), 2),
                round(data[num_feature].std(), 2),
                data[num_feature].min(),
                data[num_feature].max()
            ]
        })
        
        st.dataframe(stats_df)
    
    # Relationships between variables
    st.subheader("Relationships Between Variables")
    
    # Create tabs for different types of visualizations
    tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Scatter Plots", "Group Analysis"])
    
    with tab1:
        # Correlation heatmap
        st.write("### Correlation Heatmap")
        
        # Select numerical features for correlation analysis
        selected_num_features = st.multiselect(
            "Select numerical features for correlation analysis:",
            options=numerical_cols,
            default=numerical_cols
        )
        
        if len(selected_num_features) < 2:
            st.warning("Please select at least two numerical features for correlation analysis.")
        else:
            # Calculate correlation matrix
            corr_matrix = data[selected_num_features].corr()
            
            # Create a heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=px.colors.diverging.RdBu_r,
                title="Correlation Matrix"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Find and display the strongest correlations
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            # Remove self-correlation (always 1.0)
            corr_pairs = corr_pairs[corr_pairs < 1.0]
            
            st.write("### Strongest Correlations:")
            
            top_corr = corr_pairs.head(5)
            bottom_corr = corr_pairs.tail(5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top Positive Correlations:")
                st.dataframe(pd.DataFrame({
                    'Features': top_corr.index,
                    'Correlation': top_corr.values
                }))
            
            with col2:
                st.write("Top Negative Correlations:")
                st.dataframe(pd.DataFrame({
                    'Features': bottom_corr.index,
                    'Correlation': bottom_corr.values
                }))
    
    with tab2:
        # Scatter plots
        st.write("### Scatter Plots")
        
        # Select features for scatter plot
        x_feature = st.selectbox(
            "Select feature for x-axis:",
            options=numerical_cols,
            key="scatter_x"
        )
        
        y_feature = st.selectbox(
            "Select feature for y-axis:",
            options=[col for col in numerical_cols if col != x_feature],
            key="scatter_y"
        )
        
        # Select an optional color feature
        color_feature = st.selectbox(
            "Select a feature for color coding (optional):",
            options=['None'] + categorical_cols,
            key="scatter_color"
        )
        
        # Create scatter plot
        if color_feature == 'None':
            fig = px.scatter(
                data,
                x=x_feature,
                y=y_feature,
                title=f"{y_feature} vs {x_feature}"
            )
        else:
            fig = px.scatter(
                data,
                x=x_feature,
                y=y_feature,
                color=color_feature,
                title=f"{y_feature} vs {x_feature} (colored by {color_feature})"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation information
        correlation = data[[x_feature, y_feature]].corr().iloc[0, 1]
        
        st.write(f"Correlation between {x_feature} and {y_feature}: **{correlation:.4f}**")
        
        # Calculate and display the regression line equation
        if color_feature == 'None':
            x = data[x_feature].values
            y = data[y_feature].values
            
            # Remove NaN values
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            st.write(f"Regression Line: **{y_feature} = {slope:.4f} × {x_feature} + {intercept:.4f}**")
            st.write(f"R-squared: **{r_value**2:.4f}**")
            st.write(f"p-value: **{p_value:.6f}**")
    
    with tab3:
        # Group analysis
        st.write("### Group Analysis")
        
        # Select categorical feature for grouping
        group_feature = st.selectbox(
            "Select a categorical feature for grouping:",
            options=categorical_cols,
            key="group_feature"
        )
        
        # Select numerical feature to analyze
        analysis_feature = st.selectbox(
            "Select a numerical feature to analyze:",
            options=numerical_cols,
            key="analysis_feature"
        )
        
        # Select type of visualization
        viz_type = st.radio(
            "Select visualization type:",
            options=["Box Plot", "Violin Plot", "Bar Chart"]
        )
        
        # Create visualization
        if viz_type == "Box Plot":
            fig = px.box(
                data,
                x=group_feature,
                y=analysis_feature,
                title=f"{analysis_feature} by {group_feature}",
                color=group_feature,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Violin Plot":
            fig = px.violin(
                data,
                x=group_feature,
                y=analysis_feature,
                title=f"{analysis_feature} by {group_feature}",
                color=group_feature,
                box=True,
                points="all",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            # Calculate mean, median, or another statistic
            stat_method = st.selectbox(
                "Select statistic:",
                options=["Mean", "Median", "Minimum", "Maximum", "Standard Deviation"]
            )
            
            if stat_method == "Mean":
                grouped_stats = data.groupby(group_feature)[analysis_feature].mean().reset_index()
                y_title = f"Mean {analysis_feature}"
            elif stat_method == "Median":
                grouped_stats = data.groupby(group_feature)[analysis_feature].median().reset_index()
                y_title = f"Median {analysis_feature}"
            elif stat_method == "Minimum":
                grouped_stats = data.groupby(group_feature)[analysis_feature].min().reset_index()
                y_title = f"Minimum {analysis_feature}"
            elif stat_method == "Maximum":
                grouped_stats = data.groupby(group_feature)[analysis_feature].max().reset_index()
                y_title = f"Maximum {analysis_feature}"
            else:  # Standard Deviation
                grouped_stats = data.groupby(group_feature)[analysis_feature].std().reset_index()
                y_title = f"Standard Deviation of {analysis_feature}"
            
            fig = px.bar(
                grouped_stats,
                x=group_feature,
                y=analysis_feature,
                title=f"{y_title} by {group_feature}",
                color=group_feature,
                text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display statistics by group
        st.write(f"Statistics of {analysis_feature} by {group_feature} groups:")
        
        group_stats = data.groupby(group_feature)[analysis_feature].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).reset_index()
        
        # Round numerical columns
        numeric_cols = ['mean', 'median', 'std', 'min', 'max']
        group_stats[numeric_cols] = group_stats[numeric_cols].round(2)
        
        st.dataframe(group_stats)
        
        # Perform ANOVA test if there are more than 2 groups
        if len(data[group_feature].unique()) > 1:
            st.write("### Analysis of Variance (ANOVA)")
            st.write("Testing if the means of the groups are significantly different.")
            
            groups = []
            for name, group in data.groupby(group_feature):
                groups.append(group[analysis_feature].dropna())
            
            # Perform ANOVA test
            f_statistic, p_value = stats.f_oneway(*groups)
            
            st.write(f"F-statistic: **{f_statistic:.4f}**")
            st.write(f"p-value: **{p_value:.6f}**")
            
            alpha = 0.05
            if p_value < alpha:
                st.success(f"The means of {analysis_feature} are significantly different across {group_feature} groups (p < {alpha}).")
            else:
                st.info(f"There is no significant difference in the means of {analysis_feature} across {group_feature} groups (p > {alpha}).")
    
    # Multivariate analysis
    st.subheader("Multivariate Analysis")
    
    # Create a pair plot
    if len(numerical_cols) >= 2:
        st.write("### Pair Plot")
        
        # Select features for the pair plot
        pair_features = st.multiselect(
            "Select features for the pair plot (2-5 features recommended):",
            options=numerical_cols,
            default=numerical_cols[:min(4, len(numerical_cols))]
        )
        
        # Select an optional color feature
        pair_color = st.selectbox(
            "Select a feature for color coding (optional):",
            options=['None'] + categorical_cols,
            key="pair_color"
        )
        
        if len(pair_features) < 2:
            st.warning("Please select at least two features for the pair plot.")
        else:
            # Limit to a maximum of 5 features to avoid performance issues
            if len(pair_features) > 5:
                st.warning("Too many features selected. Using only the first 5 to avoid performance issues.")
                pair_features = pair_features[:5]
            
            with st.spinner("Creating pair plot..."):
                # Create a subset of the data
                pair_data = data[pair_features].copy()
                
                if pair_color != 'None':
                    pair_data[pair_color] = data[pair_color]
                
                # Create a pair plot using Plotly
                if pair_color == 'None':
                    fig = px.scatter_matrix(
                        pair_data,
                        dimensions=pair_features,
                        title="Pair Plot",
                        opacity=0.7
                    )
                else:
                    fig = px.scatter_matrix(
                        pair_data,
                        dimensions=pair_features,
                        color=pair_color,
                        title=f"Pair Plot (colored by {pair_color})",
                        opacity=0.7
                    )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
    
    # Summary of findings
    st.subheader("Summary of Findings")
    
    # Automatically generate insights
    insights = []
    
    # Check for correlations
    if len(numerical_cols) >= 2:
        corr_matrix = data[numerical_cols].corr()
        
        # Find strongest positive correlation
        pos_corr = corr_matrix.unstack().sort_values(ascending=False)
        pos_corr = pos_corr[pos_corr < 1.0]  # Remove self-correlations
        
        if not pos_corr.empty:
            top_pair = pos_corr.index[0]
            top_corr = pos_corr.iloc[0]
            
            if top_corr > 0.7:
                insights.append(f"Strong positive correlation ({top_corr:.2f}) between {top_pair[0]} and {top_pair[1]}.")
        
        # Find strongest negative correlation
        neg_corr = corr_matrix.unstack().sort_values(ascending=True)
        
        if not neg_corr.empty:
            bottom_pair = neg_corr.index[0]
            bottom_corr = neg_corr.iloc[0]
            
            if bottom_corr < -0.3:  # Using a lower threshold since negative correlations are often weaker
                insights.append(f"Negative correlation ({bottom_corr:.2f}) between {bottom_pair[0]} and {bottom_pair[1]}.")
    
    # Check for outliers in numerical variables
    for col in numerical_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        
        if len(outliers) > 0:
            insights.append(f"Found {len(outliers)} outliers in {col} ({(len(outliers) / len(data) * 100):.1f}% of the data).")
    
    # Check for skewness in numerical variables
    for col in numerical_cols:
        skew = data[col].skew()
        
        if abs(skew) > 1.0:
            direction = "right" if skew > 0 else "left"
            insights.append(f"{col} is significantly skewed to the {direction} (skewness = {skew:.2f}).")
    
    # Check for class imbalance in categorical variables
    for col in categorical_cols:
        value_counts = data[col].value_counts()
        if len(value_counts) > 1:
            max_count = value_counts.max()
            min_count = value_counts.min()
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 3:
                insights.append(f"Class imbalance detected in {col} (ratio of most common to least common: {imbalance_ratio:.1f}).")
    
    # Check for relationships between categorical and numerical variables
    if categorical_cols and numerical_cols:
        for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns to avoid too many tests
            for num_col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                # Perform ANOVA
                groups = []
                for name, group in data.groupby(cat_col):
                    groups.append(group[num_col].dropna())
                
                if all(len(g) > 0 for g in groups):  # Ensure all groups have data
                    f_statistic, p_value = stats.f_oneway(*groups)
                    
                    if p_value < 0.01:
                        insights.append(f"Significant relationship found between {cat_col} and {num_col} (p < 0.01).")
    
    # Display insights
    if insights:
        for i, insight in enumerate(insights, 1):
            st.write(f"{i}. {insight}")
    else:
        st.write("No significant insights detected in the preliminary analysis.")

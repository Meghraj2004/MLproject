import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

import plotly.express as px
import plotly.graph_objects as go

from utils import (
    preprocess_data, 
    create_target_from_scores, 
    calculate_regression_metrics, 
    calculate_classification_metrics,
    get_feature_importance
)

def run_supervised_learning(data):
    """Run the supervised learning page."""
    st.header("Supervised Learning")
    
    # Create tabs for classification and regression
    tab1, tab2 = st.tabs(["Classification", "Regression"])
    
    with tab1:
        run_classification(data)
    
    with tab2:
        run_regression(data)

def run_classification(data):
    """Run classification models on the data."""
    st.subheader("Classification Models")
    st.write("""
    Classification models predict categorical outcomes. Here, we'll create a target variable 
    based on whether a student passed or failed (using the average test score threshold).
    """)
    
    # Create binary target variable
    threshold = st.slider(
        "Select pass/fail threshold (average score):", 
        min_value=40, 
        max_value=90, 
        value=60,
        step=5
    )
    
    df_with_target = create_target_from_scores(data, threshold)
    
    # Show sample of the data with target
    with st.expander("Show data sample with target variable"):
        st.dataframe(df_with_target.sample(10))
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Exclude the scores and derived columns for feature selection
    feature_cols = [col for col in data.columns if col not in ['math score', 'reading score', 'writing score']]
    selected_features = st.multiselect(
        "Select features for classification:",
        options=feature_cols,
        default=feature_cols
    )
    
    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return
    
    # Add scores to the selected features if the user wants
    include_scores = st.checkbox("Include test scores as features", value=False)
    if include_scores:
        selected_features += ['math score', 'reading score', 'writing score']
    
    # Select the target variable
    target_variable = 'pass_fail'
    
    # Model selection
    st.subheader("Model Selection")
    
    classification_models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    selected_model = st.selectbox(
        "Select a classification model:",
        options=list(classification_models.keys())
    )
    
    # Hyperparameter tuning options
    st.subheader("Hyperparameter Tuning")
    
    perform_tuning = st.checkbox("Perform hyperparameter tuning", value=False, key="classification_tuning")
    
    if perform_tuning:
        # Define hyperparameters based on the selected model
        if selected_model == "Logistic Regression":
            C = st.slider("C (Regularization strength):", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            solver = st.selectbox("Solver:", ["liblinear", "lbfgs", "newton-cg", "sag", "saga"])
            param_grid = {'C': [C], 'solver': [solver]}
        
        elif selected_model == "Random Forest":
            n_estimators = st.slider("Number of trees:", min_value=10, max_value=200, value=100, step=10)
            max_depth = st.slider("Maximum depth:", min_value=1, max_value=30, value=10, step=1)
            param_grid = {'n_estimators': [n_estimators], 'max_depth': [max_depth]}
        
        elif selected_model == "Support Vector Machine":
            C = st.slider("C (Regularization strength):", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            kernel = st.selectbox("Kernel:", ["linear", "rbf", "poly", "sigmoid"])
            param_grid = {'C': [C], 'kernel': [kernel]}
        
        elif selected_model == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of neighbors:", min_value=1, max_value=20, value=5, step=1)
            weights = st.selectbox("Weight function:", ["uniform", "distance"])
            param_grid = {'n_neighbors': [n_neighbors], 'weights': [weights]}
        
        elif selected_model == "Decision Tree":
            max_depth = st.slider("Maximum depth:", min_value=1, max_value=30, value=10, step=1)
            criterion = st.selectbox("Criterion:", ["gini", "entropy"])
            param_grid = {'max_depth': [max_depth], 'criterion': [criterion]}
    
    # Train-Test split options
    test_size = st.slider("Test size (%):", min_value=10, max_value=50, value=20, step=5, key="classification_test_size") / 100
    random_state = st.slider("Random state:", min_value=0, max_value=100, value=42, step=1, key="classification_random_state")
    
    # Train the model
    if st.button("Train Classification Model"):
        with st.spinner("Training model..."):
            # Get data with target variable
            model_data = df_with_target[selected_features + [target_variable]]
            
            # Preprocess data
            preprocessed = preprocess_data(
                model_data,
                target_variable=target_variable,
                categorical_encoding='label',
                test_size=test_size,
                random_state=random_state
            )
            
            X_train = preprocessed['X_train']
            X_test = preprocessed['X_test']
            y_train = preprocessed['y_train']
            y_test = preprocessed['y_test']
            
            # Select and train model
            model = classification_models[selected_model]
            
            if perform_tuning:
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.success(f"Best hyperparameters: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_classification_metrics(y_test, y_pred)
            
            # Display results
            st.subheader("Model Performance")
            
            # Metrics table
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(metrics_df)
            
            with col2:
                # Create a bar chart of metrics
                fig = px.bar(
                    metrics_df, 
                    x='Metric', 
                    y='Value', 
                    title='Classification Metrics',
                    text='Value',
                    color='Metric',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            
            importance = get_feature_importance(model, X_train.columns)
            
            if importance:
                importance_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")
            
            # Confusion Matrix
            from sklearn.metrics import confusion_matrix
            
            st.subheader("Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            if len(np.unique(y_test)) == 2:
                ax.set_xticklabels(['Fail', 'Pass'])
                ax.set_yticklabels(['Fail', 'Pass'])
            
            st.pyplot(fig)
            
            # Class distribution
            st.subheader("Class Distribution")
            
            fig = px.pie(
                values=[np.sum(y_test == 0), np.sum(y_test == 1)],
                names=['Fail', 'Pass'],
                title='Distribution of Classes in Test Set',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability Distribution (if available)
            if hasattr(model, 'predict_proba'):
                st.subheader("Probability Distribution")
                
                y_prob = model.predict_proba(X_test)[:, 1]
                
                fig = px.histogram(
                    y_prob,
                    nbins=30,
                    title='Probability Distribution',
                    labels={'value': 'Probability of Passing', 'count': 'Count'},
                    color_discrete_sequence=['#3366CC']
                )
                st.plotly_chart(fig, use_container_width=True)

def run_regression(data):
    """Run regression models on the data."""
    st.subheader("Regression Models")
    st.write("""
    Regression models predict continuous numerical values. Here, we'll predict 
    students' test scores based on other features.
    """)
    
    # Select target variable
    target_options = ['math score', 'reading score', 'writing score']
    target_variable = st.selectbox(
        "Select the target variable to predict:",
        options=target_options
    )
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Exclude the scores for feature selection
    feature_cols = [col for col in data.columns if col not in target_options]
    selected_features = st.multiselect(
        "Select features for regression:",
        options=feature_cols,
        default=feature_cols
    )
    
    if not selected_features:
        st.warning("Please select at least one feature to proceed.")
        return
    
    # Include other scores as features
    include_other_scores = st.checkbox("Include other test scores as features", value=True)
    if include_other_scores:
        other_scores = [score for score in target_options if score != target_variable]
        selected_features += other_scores
    
    # Model selection
    st.subheader("Model Selection")
    
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Machine": SVR(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor()
    }
    
    selected_model = st.selectbox(
        "Select a regression model:",
        options=list(regression_models.keys())
    )
    
    # Hyperparameter tuning options
    st.subheader("Hyperparameter Tuning")
    
    perform_tuning = st.checkbox("Perform hyperparameter tuning", value=False, key="regression_tuning")
    
    if perform_tuning:
        # Define hyperparameters based on the selected model
        if selected_model == "Linear Regression":
            st.info("Linear Regression has no hyperparameters to tune.")
            param_grid = {}
        
        elif selected_model == "Random Forest":
            n_estimators = st.slider("Number of trees:", min_value=10, max_value=200, value=100, step=10)
            max_depth = st.slider("Maximum depth:", min_value=1, max_value=30, value=10, step=1)
            param_grid = {'n_estimators': [n_estimators], 'max_depth': [max_depth]}
        
        elif selected_model == "Support Vector Machine":
            C = st.slider("C (Regularization strength):", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            kernel = st.selectbox("Kernel:", ["linear", "rbf", "poly", "sigmoid"])
            param_grid = {'C': [C], 'kernel': [kernel]}
        
        elif selected_model == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of neighbors:", min_value=1, max_value=20, value=5, step=1)
            weights = st.selectbox("Weight function:", ["uniform", "distance"])
            param_grid = {'n_neighbors': [n_neighbors], 'weights': [weights]}
        
        elif selected_model == "Decision Tree":
            max_depth = st.slider("Maximum depth:", min_value=1, max_value=30, value=10, step=1)
            criterion = st.selectbox("Criterion:", ["mse", "mae"])
            param_grid = {'max_depth': [max_depth], 'criterion': [criterion]}
    
    # Train-Test split options
    test_size = st.slider("Test size (%):", min_value=10, max_value=50, value=20, step=5, key="regression_test_size") / 100
    random_state = st.slider("Random state:", min_value=0, max_value=100, value=42, step=1, key="regression_random_state")
    
    # Train the model
    if st.button("Train Regression Model"):
        with st.spinner("Training model..."):
            # Get data for model
            model_data = data[selected_features + [target_variable]]
            
            # Preprocess data
            preprocessed = preprocess_data(
                model_data,
                target_variable=target_variable,
                categorical_encoding='label',
                test_size=test_size,
                random_state=random_state
            )
            
            X_train = preprocessed['X_train']
            X_test = preprocessed['X_test']
            y_train = preprocessed['y_train']
            y_test = preprocessed['y_test']
            
            # Select and train model
            model = regression_models[selected_model]
            
            if perform_tuning and selected_model != "Linear Regression":
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.success(f"Best hyperparameters: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = calculate_regression_metrics(y_test, y_pred)
            
            # Display results
            st.subheader("Model Performance")
            
            # Metrics table
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(metrics_df)
            
            with col2:
                # Create a bar chart of metrics
                fig = px.bar(
                    metrics_df[metrics_df['Metric'] != 'R²'],  # Exclude R² for scale
                    x='Metric', 
                    y='Value', 
                    title='Regression Metrics (Error)',
                    text='Value',
                    color='Metric',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # Display R² separately
                r2_value = metrics['R²']
                st.metric("R² Score", f"{r2_value:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            
            importance = get_feature_importance(model, X_train.columns)
            
            if importance:
                importance_df = pd.DataFrame({
                    'Feature': list(importance.keys()),
                    'Importance': list(importance.values())
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Values")
            
            pred_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred,
                'Error': np.abs(y_test - y_pred)
            })
            
            fig = px.scatter(
                pred_df,
                x='Actual',
                y='Predicted',
                color='Error',
                title=f'Actual vs Predicted {target_variable}',
                color_continuous_scale=px.colors.sequential.Viridis
            )
            
            # Add perfect prediction line
            min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
            max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual Plot
            st.subheader("Residual Plot")
            
            pred_df['Residual'] = y_test - y_pred
            
            fig = px.scatter(
                pred_df,
                x='Predicted',
                y='Residual',
                title='Residual Plot',
                color='Residual',
                color_continuous_scale=px.colors.sequential.RdBu_r
            )
            
            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution of Errors
            st.subheader("Distribution of Errors")
            
            fig = px.histogram(
                pred_df['Residual'],
                nbins=30,
                title='Distribution of Residuals',
                labels={'value': 'Residual', 'count': 'Frequency'},
                color_discrete_sequence=['#3366CC']
            )
            st.plotly_chart(fig, use_container_width=True)

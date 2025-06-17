


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from PIL import Image

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    # Try to load from URL first, then local if that fails
    try:
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        dataset = pd.read_csv(url)
    except:
        try:
            dataset = pd.read_csv("BostonHousing.csv")
        except:
            st.error("Failed to load dataset. Please ensure the file is available.")
            return None
    
    dataset['rm'] = dataset['rm'].fillna(dataset['rm'].mean())
    dataset = dataset.rename(columns={'medv': 'Price'})
    return dataset

dataset = load_data()

if dataset is not None:
    # Sidebar for user inputs
    st.sidebar.header("User Input Features")
    
    # Model selection
    model_type = st.sidebar.selectbox("Select Regression Model", 
                                     ["Linear Regression", "Ridge Regression", "Lasso Regression"])
    
    # Get user inputs - using consistent column names with the dataset
    def get_user_input():
        crim = st.sidebar.slider('Crime rate (crim)', float(dataset['crim'].min()), float(dataset['crim'].max()), float(dataset['crim'].mean()))
        zn = st.sidebar.slider('Residential land zoned (zn)', float(dataset['zn'].min()), float(dataset['zn'].max()), float(dataset['zn'].mean()))
        indus = st.sidebar.slider('Non-retail business acres (indus)', float(dataset['indus'].min()), float(dataset['indus'].max()), float(dataset['indus'].mean()))
        chas = st.sidebar.selectbox('Charles River dummy variable (chas)', [0, 1], index=0)
        nox = st.sidebar.slider('Nitric oxides concentration (nox)', float(dataset['nox'].min()), float(dataset['nox'].max()), float(dataset['nox'].mean()))
        rm = st.sidebar.slider('Average rooms per dwelling (rm)', float(dataset['rm'].min()), float(dataset['rm'].max()), float(dataset['rm'].mean()))
        age = st.sidebar.slider('Owner-occupied units built prior to 1940 (age)', float(dataset['age'].min()), float(dataset['age'].max()), float(dataset['age'].mean()))
        dis = st.sidebar.slider('Weighted distances to employment centers (dis)', float(dataset['dis'].min()), float(dataset['dis'].max()), float(dataset['dis'].mean()))
        rad = st.sidebar.slider('Accessibility to radial highways (rad)', float(dataset['rad'].min()), float(dataset['rad'].max()), float(dataset['rad'].mean()))
        tax = st.sidebar.slider('Property tax rate (tax)', float(dataset['tax'].min()), float(dataset['tax'].max()), float(dataset['tax'].mean()))
        ptratio = st.sidebar.slider('Pupil-teacher ratio (ptratio)', float(dataset['ptratio'].min()), float(dataset['ptratio'].max()), float(dataset['ptratio'].mean()))
        b = st.sidebar.slider('Proportion of blacks by town (b)', float(dataset['b'].min()), float(dataset['b'].max()), float(dataset['b'].mean()))
        lstat = st.sidebar.slider('% lower status of population (lstat)', float(dataset['lstat'].min()), float(dataset['lstat'].max()), float(dataset['lstat'].mean()))
        
        user_data = {
            'crim': crim,
            'zn': zn,
            'indus': indus,
            'chas': chas,
            'nox': nox,
            'rm': rm,
            'age': age,
            'dis': dis,
            'rad': rad,
            'tax': tax,
            'ptratio': ptratio,
            'b': b,
            'lstat': lstat
        }
        
        features = pd.DataFrame(user_data, index=[0])
        return features

    user_input = get_user_input()
    
    # Main panel
    st.title("🏠 Boston House Price Prediction")
    st.markdown("""
    This app predicts **Boston House Prices** using different regression models.
    - Adjust the input parameters using the sliders in the sidebar
    - Select the regression model type
    - View the prediction and model performance metrics
    """)
    
    # Show user inputs
    st.subheader('User Input Features')
    st.write(user_input)
    
    # Prepare data
    X = dataset.drop('Price', axis=1)
    y = dataset['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # Train models
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Cross-validation
        mse = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
        mean_mse = np.mean(mse)
        
    elif model_type == "Ridge Regression":
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 45, 50, 55, 100]}
        ridge = Ridge()
        model = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
        model.fit(X_train, y_train)
        mean_mse = model.best_score_
        
    elif model_type == "Lasso Regression":
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 50, 55, 100]}
        lasso = Lasso()
        model = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)
        model.fit(X_train, y_train)
        mean_mse = model.best_score_
    
    # Make prediction - ensure column order matches training data
    user_input = user_input[X_train.columns]
    
    if model_type == "Linear Regression":
        prediction = model.predict(user_input)
    else:
        prediction = model.best_estimator_.predict(user_input)
    
    st.subheader('Prediction')
    st.markdown(f"### Predicted House Price: **${prediction[0]*1000:,.2f}**")
    
    # Model evaluation
    st.subheader('Model Performance')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Metrics")
        if model_type == "Linear Regression":
            train_pred = model.predict(X_train)
            st.write(f"R-squared: {r2_score(y_train, train_pred):.4f}")
            st.write(f"MSE: {mean_squared_error(y_train, train_pred):.4f}")
        else:
            st.write(f"Best Alpha: {model.best_params_['alpha']}")
            st.write(f"Cross-validated Negative MSE: {mean_mse:.4f}")
    
    with col2:
        st.markdown("#### Test Metrics")
        if model_type == "Linear Regression":
            test_pred = model.predict(X_test)
            st.write(f"R-squared: {r2_score(y_test, test_pred):.4f}")
            st.write(f"MSE: {mean_squared_error(y_test, test_pred):.4f}")
        else:
            test_pred = model.best_estimator_.predict(X_test)
            st.write(f"R-squared: {r2_score(y_test, test_pred):.4f}")
            st.write(f"MSE: {mean_squared_error(y_test, test_pred):.4f}")
    
    # Feature importance
    st.subheader('Feature Importance')
    
    if model_type == "Linear Regression":
        coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    else:
        coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.best_estimator_.coef_})
    
    coefficients = coefficients.sort_values(by='Coefficient', key=abs, ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis', ax=ax)
    ax.set_title('Feature Coefficients (Impact on Price)')
    st.pyplot(fig)
    
    # Actual vs Predicted plot
    st.subheader('Actual vs Predicted Prices')
    
    if model_type == "Linear Regression":
        preds = model.predict(X_test)
    else:
        preds = model.best_estimator_.predict(X_test)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(y_test, preds, alpha=0.6)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel('Actual Prices')
    ax2.set_ylabel('Predicted Prices')
    ax2.set_title('Actual vs Predicted House Prices')
    st.pyplot(fig2)
    
    # Data exploration
    st.subheader('Data Exploration')
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Price Distribution")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.histplot(dataset['Price'], kde=True, ax=ax3)
        st.pyplot(fig3)
    
    with col4:
        st.markdown("#### Correlation with Price")
        corr = dataset.corr()['Price'].sort_values(ascending=False)[1:]
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=corr.values, y=corr.index, ax=ax4)
        st.pyplot(fig4)
    
    # Show raw data option
    if st.checkbox('Show Raw Data'):
        st.subheader('Raw Data')
        st.write(dataset)
    
    # Add some styling
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSlider>div>div>div>div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.error("Please ensure the dataset is available to run the application.")
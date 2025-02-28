import streamlit as st
import streamlit as st
import numpy as np  
import pandas as pd  
from datetime import datetime  
import matplotlib.pyplot as plt  
import seaborn as sns  
import plotly.express as px  
import plotly.graph_objects as go  
import plotly.figure_factory as ff  
from plotly.offline import init_notebook_mode, iplot  
from IPython.display import display, HTML
from sklearn.preprocessing import MinMaxScaler  
from scipy import stats  
from sklearn import (manifold, decomposition, ensemble, discriminant_analysis,)  
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import mean_squared_error  
from time import time  

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
# from scipy.sparse import csr_matrix
from surprise import Dataset, Reader, SVD, BaselineOnly, CoClustering
from surprise.model_selection import train_test_split as surprise_train_test_split
from math import sqrt  
import warnings  
warnings.filterwarnings('ignore') 

def main():
    """Anime Recommender App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Anime Recommender")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
options =  ["Project Introduction", "Importing Packages", "Data loading and Inspection", "Data Cleaning", "Exploratory Data Analysis (EDA)","Data Preprocessing","Model Development", "Model Evaluation", "Model Deployment", "Conclusion and Recomendations"]
section = st.sidebar.selectbox("Options", options)
    
    #### Introduction section     
if section == "Project Introduction":
    
        st.title("Project Introduction")

        st.subheader("1.1. Introduction")
        st.write("""
        ***still to add***
        """)

        # Objectives of the Project
        st.subheader("1.2. Objectives of the Project")
        st.write("""
        Develop an unsupervised learning model to recommend anime based on user preferences.
        Perform data cleaning, preprocessing, and exploratory data analysis (EDA) to understand patterns in the dataset.
        Implement clustering algorithms (e.g., K-Means, DBSCAN) for grouping similar anime.
        Evaluate the performance of the recommendation system.
        Submit predictions to the Kaggle Challenge Leaderboard for evaluation.
        """)

        # Data Source Section
        st.subheader("1.3. Data Source")
        st.write("The data can be accessed at: [Kaggle](https://www.kaggle.com/t/dec21d5abc8c4d33bae9a25fbc3cfb7b)")

        # Problem Statement Section
        st.subheader("1.4. Problem Statement")
        st.write("""
        With the rapid growth of anime streaming platforms, users often struggle to find new anime that align with their preferences. A recommender system can help solve this problem by analyzing user-anime interactions and suggesting relevant content.

        In this project, we will develop an unsupervised learning-based anime recommender system using clustering techniques. The goal is to provide personalized recommendations based on user behavior and anime attributes.
        """)

if section == "Importing Packages":
        st.header("2. Importing Packages")
        
        # Code for importing packages (displayed as text)
        code = """
        import numpy as np  
        import pandas as pd  
        from datetime import datetime  
        import matplotlib.pyplot as plt  
        import seaborn as sns  
        import plotly.express as px  
        import plotly.graph_objects as go  
        import plotly.figure_factory as ff  
        from plotly.offline import init_notebook_mode, iplot  
        from IPython.display import display, HTML
        from sklearn.preprocessing import MinMaxScaler  
        from scipy import stats  
        from sklearn import (manifold, decomposition, ensemble, discriminant_analysis,)  
        from sklearn.metrics.pairwise import cosine_similarity  
        from sklearn.metrics import mean_squared_error  
        from time import time  

        from sklearn.preprocessing import MinMaxScaler, LabelEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import sigmoid_kernel
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.decomposition import TruncatedSVD
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
        # from scipy.sparse import csr_matrix
        from surprise import Dataset, Reader, SVD, BaselineOnly, CoClustering
        from surprise.model_selection import train_test_split as surprise_train_test_split
        from math import sqrt  
        import warnings  
        warnings.filterwarnings('ignore')
        """
        st.code(code, language='python')

        # Add button to run the code
        run_button = st.button("Run")
        
        if run_button:
            try:
                st.success("Importing Packages completed successfully!")
            except Exception as e:
                st.error(f"Error executing code: {e}")
    # --- Loading Data Section ---
if section == "Loading Data and Inspection":
        st.header("3. Loading Data and Inspection")
        st.write("The data used is assigned to anime_df, ratings_df, test df and submission_df. To better manipulate and analyse the data, it is loaded into a Pandas DataFrame using the Pandas function and .read_csv() and then referred to as df.")
        
        # Code for loading the data
        data_loading_code = r"""
        # Load the data
        st.session_state.anime_df = pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\anime.csv")
        st.session_state.ratings_df = pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\train.csv")
        st.session_state.test_df= pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\test.csv")
        st.session_state.submission_df = pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\submission.csv")        
    
        # Display the DataFrames
        st.subheader("Preview of the DataFrames")
        st.write("Anime DataFrame:")
        st.dataframe(st.session_state.anime_df)
                
        st.write("Ratings DataFrame:")
        st.dataframe(st.session_state.ratings_df)

        st.write("Test DataFrame:")
        st.dataframe(st.session_state.test_df)                

        st.write("Submission DataFrame:")
        st.dataframe(st.session_state.submission_df)
        """
        
        st.code(data_loading_code, language='python')


# Button to run the data loading process
if st.button("Run"):
    try:
        # Load the data into session state
        st.session_state.anime_df = pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\anime.csv")
        st.session_state.ratings_df = pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\train.csv")
        st.session_state.test_df= pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\test.csv")
        st.session_state.submission_df = pd.read_csv(r"C:\Users\bonas\streamlit-anime-recommender\Streamlit_Anime_Recommendations\Anime data\submission.csv")

        # Store a flag in session state
        st.session_state.data_loaded = True

        st.success("✅ Data loaded successfully!")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Please check the file paths or ensure that the data files exist.")

# Display the DataFrames only if data is loaded
if st.session_state.get("data_loaded", False):
    st.subheader("Preview of the DataFrames")

    st.write("**Anime DataFrame:**")
    st.dataframe(st.session_state.anime_df)

    st.write("**Ratings DataFrame:**")
    st.dataframe(st.session_state.ratings_df)

    st.write("**Test DataFrame:**")
    st.dataframe(st.session_state.test_df)

    st.write("**Submission DataFrame:**")
    st.dataframe(st.session_state.submission_df)


         


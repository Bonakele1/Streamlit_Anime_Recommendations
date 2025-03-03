import streamlit as st
import io
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

        st.subheader("Introduction")
        st.write("""
        ***still to add***
        """)

        # Objectives of the Project
        st.subheader("Objectives of the Project")
        st.write("""
        Develop an unsupervised learning model to recommend anime based on user preferences.
        Perform data cleaning, preprocessing, and exploratory data analysis (EDA) to understand patterns in the dataset.
        Implement clustering algorithms (e.g., K-Means, DBSCAN) for grouping similar anime.
        Evaluate the performance of the recommendation system.
        Submit predictions to the Kaggle Challenge Leaderboard for evaluation.
        """)

        # Data Source Section
        st.subheader("Data Source")
        st.write("The data can be accessed at: [Kaggle](https://www.kaggle.com/t/dec21d5abc8c4d33bae9a25fbc3cfb7b)")

        # Problem Statement Section
        st.subheader("1.4. Problem Statement")
        st.write("""
        With the rapid growth of anime streaming platforms, users often struggle to find new anime that align with their preferences. A recommender system can help solve this problem by analyzing user-anime interactions and suggesting relevant content.

        In this project, we will develop an unsupervised learning-based anime recommender system using clustering techniques. The goal is to provide personalized recommendations based on user behavior and anime attributes.
        """)

if section == "Importing Packages":
        st.header("Importing Packages")
        
        # Code for importing packages (displayed as text)
        code = """
        import io
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
        run_button = st.button("Run imports")
        
        if run_button:
            try:
                st.success("Importing Packages completed successfully!")
            except Exception as e:
                st.error(f"Error executing code: {e}")
    # --- Loading Data Section ---
if section == "Data loading and Inspection":
        st.header("Loading Data and Inspection")
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
        if st.button("Run Data"):
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

# --- Data Cleaning ---
if section == "Data Cleaning":
            st.header("Data Cleaning")
            
            # --- Missing Values Check ---
            st.subheader("Check for Missing Values")
            st.write("""
            In this step, we will check for any missing values in the dataset and handle them appropriately (e.g., by imputation or removal) to ensure the dataset is complete and ready for analysis.
            """)

            def display_missing_values(df, name):
                        
                        #Displays missing values in a DataFrame in a clean, readable format.
                        missing = df.isnull().sum().to_frame(name="Missing Values")

                        st.subheader(f"🔍 Missing Values in {name} Dataset")
                        st.dataframe(missing)
                    

                    # Button to run missing values check
            check_null_values_button = st.button("Check Missing Values", key="check_null")

            if check_null_values_button:
                    try:
                        # Ensure datasets exist in session_state
                        required_keys = ["anime_df", "ratings_df", "test_df"]
                        if not all(key in st.session_state for key in required_keys):
                            st.error("❌ Data is not loaded. Please load the dataset first.")
                        else:
                            # Assign datasets correctly
                            datasets = {
                                "Anime": st.session_state.anime_df,
                                "Ratings": st.session_state.ratings_df,
                                "Test": st.session_state.test_df
                            }
                            # Loop through all datasets and display missing values
                            for name, df in datasets.items():
                                display_missing_values(df, name)

                        st.success("Missing values displayed successfully!")

                    except Exception as e:
                        st.error(f"Error checking missing values: {e}")
                        st.write("Please check the data files.")

                
            st.subheader("Handling Missing Values")
            st.write("""
                    Fill in missing values in the anime dataset:
                    -Fill missing rating with mean for corresponding type.
                    -Fill genre and name missing values with unknown.
            """)

                    # Button to apply missing value handling
            fill_data_button = st.button("Fill Data", key="fill_data")

            if fill_data_button:
                    try:
                        if "anime_df" in st.session_state:
                            # Fill missing ratings with mean per 'type'
                            st.session_state.anime_df['rating'] = st.session_state.anime_df['rating'].fillna(
                                st.session_state.anime_df.groupby('type')['rating'].transform('mean')
                            )

                            # Fill missing genre and name with 'Unknown'
                            st.session_state.anime_df['genre'] = st.session_state.anime_df['genre'].fillna('Unknown')
                            st.session_state.anime_df['name'] = st.session_state.anime_df['name'].fillna('Unknown')

                            st.success("✅ Missing values handled successfully!")

                            # Show updated missing values
                            datasets = {
                            "Anime": st.session_state.anime_df,
                            "Ratings": st.session_state.ratings_df,
                            "Test": st.session_state.test_df
                            }
                        # Loop through all datasets and display missing values
                        for name, df in datasets.items():
                            display_missing_values(df, name)
                    
                    except Exception as e:
                        st.error(f"⚠️ Error cleaning data: {e}")

            # --- Handle Null Values ---
            st.subheader("Handling Null Values")
            st.write("""
            We apply the following Null value replacements:
            - type: Replace Null-type entries.
            - genre: Replace Null-genre entries
            - rating: Replace Null-ratings entries.
            """)

            # Button to clean data
            Null_data_button = st.button("Null Data", key="null_data")

            if Null_data_button:
                try:
                    if "anime_df" in st.session_state:
                        df = st.session_state.anime_df 
                        
                        # --- Replace Null 'type' entries ---
                        df['type'] = df['type'].replace('', 'TV').fillna('TV')

                        # --- Replace Null 'genre' entries based on 'type' ---
                        def assign_default_genre(row):
                            if pd.isna(row['genre']) or row['genre'] == '':
                                default_genres = {
                                    'Movie': 'Comedy',
                                    'TV': 'Comedy',
                                    'OVA': 'Hentai',
                                    'Special': 'Comedy',
                                    'Music': 'Music',
                                    'ONA': 'Comedy'
                                }
                                return default_genres.get(row['type'], 'Unknown')
                            return row['genre']

                        df['genre'] = df.apply(assign_default_genre, axis=1)

                        # --- Replace Null 'rating' entries with type-wise mean ---
                        df['rating'] = df['rating'].fillna(df.groupby('type')['rating'].transform('mean'))

                        # Save back to session state
                        st.session_state.anime_df = df  

                        st.success("✅ Null values handled successfully!")

                    # Show updated missing values
                    for name, df in datasets.items():
                        display_missing_values(df, name)

                except Exception as e:
                    st.error(f"⚠️ Error cleaning data: {e}")    

# --- Exploratory Data Analysis ---
if section == "Exploratory Data Analysis (EDA)":
            st.header("Exploratory Data Analysis")            
            if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
                st.error("⚠️ Data is not loaded yet! Please load the data in the 'Loading Data and Inspection' section first.")
            
            else:

                # Assign session state data to variables
                anime_df = st.session_state.anime_df
                ratings_df = st.session_state.ratings_df
                test_df = st.session_state.test_df
                submission_df = st.session_state.submission_df

                # Summary statistics
                st.title("Summary Statistics")

                st.write("### Anime Dataset")
                st.dataframe(anime_df.describe())

                st.write("### User Ratings (Train) Dataset")
                st.dataframe(ratings_df.describe())

                st.write("### Test Dataset")
                st.dataframe(test_df.describe())

                st.write("### Submission Dataset")
                st.dataframe(submission_df.describe())

                # Dataset info
                st.title("Dataset Information")

                def capture_info(df):
                    """Capture df.info() output as a string."""
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    return buffer.getvalue()

                st.write("#### Anime Dataset Info")
                st.text(capture_info(anime_df))

                st.write("#### User Ratings (Train) Dataset Info")
                st.text(capture_info(ratings_df))

                st.write("#### Test Dataset Info")
                st.text(capture_info(test_df))

                st.write("#### Submission Dataset Info")
                st.text(capture_info(submission_df))

                # Unique Value Counts
                st.title("Unique Value Counts")

                st.write("### Anime Dataset Unique Counts")
                st.dataframe(anime_df.nunique().to_frame(name="Unique Values"))

                st.write("### User Ratings (Train) Dataset Unique Counts")
                st.dataframe(ratings_df.nunique().to_frame(name="Unique Values"))

                st.write("### Test Dataset Unique Counts")
                st.dataframe(test_df.nunique().to_frame(name="Unique Values"))

                st.write("### Submission Dataset Unique Counts")
                st.dataframe(submission_df.nunique().to_frame(name="Unique Values"))

            st.title("User-related: User Behavior Analysis")    
            # Ensure data is loaded
            if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
                st.error("⚠️ Data is not loaded yet! Please load the data in the 'Loading Data and Inspection' section first.")
            else:
                st.subheader("User Behavior Analysis")

                # Get the ratings dataset from session state
                ratings_df = st.session_state.ratings_df

                # 📊 **Distribution of User Ratings**
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(ratings_df['rating'], bins=20, kde=False, color='skyblue', ax=ax)
                ax.set_title('Distribution of User Ratings')
                ax.set_xlabel('User Rating')
                ax.set_ylabel('Frequency')
                ax.grid(True)
                st.pyplot(fig)  # Display plot in Streamlit

            # 📊 **Number of Ratings Per User**
            st.subheader("Distribution of Ratings per User")

            # Calculate number of ratings per user
            user_ratings_count = ratings_df.groupby('user_id')['rating'].count()

            # Display descriptive statistics
            st.write("#### Descriptive Statistics for Number of Ratings per User")
            st.dataframe(user_ratings_count.describe().to_frame())

            # Plot distribution of ratings per user
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(user_ratings_count, bins=50, kde=True, color='skyblue', ax=ax)
            ax.set_title('Distribution of Ratings per User')
            ax.set_xlabel('Number of Ratings')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            st.pyplot(fig)   

            # 📊 **Number of Ratings Per User**
            st.subheader("Distribution of Ratings per User")
            st.write("#### Descriptive Statistics for Number of Ratings per User")

            user_ratings_count = ratings_df.groupby('user_id')['rating'].count()
            st.dataframe(user_ratings_count.describe().to_frame())

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(user_ratings_count, bins=50, kde=True, color='skyblue', ax=ax)
            ax.set_title('Distribution of Ratings per User')
            ax.set_xlabel('Number of Ratings')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            st.pyplot(fig)

            # 🕵️ **Identify Active and Inactive Users**
            active_threshold = user_ratings_count.quantile(0.75)  # 75th percentile
            inactive_threshold = user_ratings_count.quantile(0.25)  # 25th percentile
            active_users = user_ratings_count[user_ratings_count > active_threshold].index
            inactive_users = user_ratings_count[user_ratings_count < inactive_threshold].index

            st.subheader("Active and Inactive Users")
            st.write(f"🔥 **Number of active users:** {len(active_users)}")
            st.write(f"❄️ **Number of inactive users:** {len(inactive_users)}")

            # 📊 **Categorizing Users Based on Activity Levels**
            st.subheader("User Activity Levels")
            
            very_active_threshold = user_ratings_count.quantile(0.9)  # Top 10%
            active_threshold = user_ratings_count.quantile(0.75)      # Top 25%
            inactive_threshold = user_ratings_count.quantile(0.25)    # Bottom 25%
            very_inactive_threshold = user_ratings_count.quantile(0.1)  # Bottom 10%

            user_activity = pd.DataFrame({'user_id': user_ratings_count.index})
            user_activity['user_type'] = np.where(
                user_ratings_count > very_active_threshold, 'very_active',
                np.where(user_ratings_count > active_threshold, 'active',
                        np.where(user_ratings_count < inactive_threshold, 'inactive', 'low_activity'))
            )

            st.write("#### User Type Distribution")
            st.dataframe(user_activity['user_type'].value_counts().to_frame())

            # 📊 **Visualizing User Activity Levels**
            fig, ax = plt.subplots(figsize=(10, 6))
            user_type_order = ['low_activity', 'inactive', 'active', 'very_active']
            sns.countplot(x='user_type', data=user_activity, order=user_type_order, palette='pastel', ax=ax)
            ax.set_title('Distribution of User Activity Levels')
            ax.set_xlabel('User Type')
            ax.set_ylabel('Count of Users')
            ax.grid(True)
            st.pyplot(fig)

            # 📊 **Comparing Number of Ratings and User Type**
            st.subheader("Comparing Number of Ratings and User Type")
            
            user_activity['rating_count'] = user_ratings_count.values
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                x='user_type',
                y='rating_count',
                data=user_activity,
                order=user_type_order,
                showmeans=True,
                palette='pastel',
                ax=ax
            )
            ax.set_title('Comparing Number of Ratings and User Type')
            ax.set_xlabel('User Type')
            ax.set_ylabel('Number of Ratings')
            ax.grid(True)
            st.pyplot(fig)

            st.title("Analyzing Anime Content Behavior: Insights into Genres, Ratings, and Types")
            st.write("""
            Understanding anime content behavior helps us uncover patterns in genres, ratings, and types of anime that users engage with. 
            By analyzing these factors, we can determine which genres are most popular, how ratings vary across different categories,
            and what types of anime (e.g., TV series, movies, OVAs) receive the most interaction. These insights
            will guide the development of a more personalized and effective recommender system.
            """)

           # Ensure data is loaded
            if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
                st.error("⚠️ Data is not loaded yet! Please load the data in the 'Loading Data and Inspection' section first.")
            else:
                st.subheader("Anime Data Analysis")

                # Get the anime dataset from session state
                anime_df = st.session_state.anime_df
                ratings_df = st.session_state.ratings_df

                # 📊 **Distribution of Average Anime Ratings**
                st.subheader("Distribution of Average Anime Ratings")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=anime_df, x='rating', bins=20, kde=False, color='salmon', ax=ax)
                ax.set_title('Distribution of Average Anime Ratings')
                ax.set_xlabel('Average Anime Rating')
                ax.set_ylabel('Frequency')
                ax.grid(True)
                st.pyplot(fig)

                # 📊 **Distribution of Anime Types**
                st.subheader("Distribution of Anime Types")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.countplot(data=anime_df, x='type', order=anime_df['type'].value_counts().index, palette='pastel', ax=ax)
                ax.set_title('Distribution of Anime Types')
                ax.set_xlabel('Anime Type')
                ax.set_ylabel('Count of Anime')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)

                # 🎭 **Analysis of Genres**
                st.subheader("Distribution of Anime Genres")
                
                genre_count = anime_df['genre'].str.get_dummies(sep=', ').sum()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                genre_count.sort_values(ascending=False).plot(kind='bar', color='lightblue', ax=ax)
                ax.set_title('Distribution of Anime Genres')
                ax.set_xlabel('Genre')
                ax.set_ylabel('Count of Anime')
                st.pyplot(fig)

                # 🏆 **Top 10 Genres**
                st.subheader("Top 10 Anime Genres")

                top_genres = genre_count.sort_values(ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(12, 6))
                top_genres.plot(kind='bar', color='skyblue', ax=ax)
                ax.set_title('Top 10 Anime Genres')
                ax.set_xlabel('Genre')
                ax.set_ylabel('Count of Anime')
                st.pyplot(fig)

                # 🥇 **Top 10 Anime by Number of User Ratings**
                st.subheader("Top 10 Anime by Number of User Ratings")

                top_anime_by_ratings = ratings_df['anime_id'].value_counts().head(10)
                top_anime_titles = anime_df.set_index('anime_id').loc[top_anime_by_ratings.index]['name']
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(y=top_anime_titles, x=top_anime_by_ratings.values, orient='h', palette='coolwarm', ax=ax)
                ax.set_title('Top 10 Anime by Number of User Ratings')
                ax.set_xlabel('Number of Ratings')
                ax.set_ylabel('Anime Title')
                st.pyplot(fig)

                # ⭐ **Top 10 Anime by Average Rating**
                st.subheader("Top 10 Anime by Average Rating")

                top_anime_by_avg_rating = anime_df[['anime_id', 'name', 'rating']].drop_duplicates('anime_id') \
                    .sort_values(by='rating', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='rating', y='name', data=top_anime_by_avg_rating, orient='h', palette='coolwarm', ax=ax)
                ax.set_title('Top 10 Anime by Average Rating')
                ax.set_xlabel('Average Rating')
                ax.set_ylabel('Anime Title')
                st.pyplot(fig)

                # Display the top 10 list for reference
                st.write("#### Top 10 Anime by Average Rating")
                st.dataframe(top_anime_by_avg_rating[['name', 'rating']])

                # 🌟 **Top 10 Unique Anime by User Rating**
                st.subheader("Top 10 Unique Anime by User Rating")

                top_rated_unique_animes = ratings_df.groupby('anime_id')['rating'].mean()

                subset = pd.DataFrame(anime_df[['anime_id', 'name']])
                subset['rating'] = subset['anime_id'].map(top_rated_unique_animes)
                subset = subset.nlargest(10, 'rating')

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.barh(subset['name'], subset['rating'], color='skyblue')
                ax.set_title('Top 10 Unique Anime by User Rating')
                ax.set_xlabel('User Rating')
                ax.set_ylabel('Anime Title')
                st.pyplot(fig)

                # 🎬 **Top 10 Unique Genres by User Rating**
                st.subheader("Top 10 Unique Genres by Anime Rating")

                top_genres = anime_df.nlargest(10, 'rating')
                
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.barh(top_genres['genre'], top_genres['rating'], color='lightblue')
                ax.set_title('Top 10 Unique Genres by Anime Rating')
                ax.set_xlabel('Average Anime Rating')
                ax.set_ylabel('Genre')
                ax.invert_yaxis()
                st.pyplot(fig)

                # 🔗 **Correlation Matrix**

                st.subheader("Correlation Matrix")
                merged_df = pd.merge(ratings_df, anime_df[['anime_id', 'rating', 'episodes', 'members']], on='anime_id', how='left')
                merged_df = merged_df.rename(columns={
                    'rating_x': 'user_rating',
                    'rating_y': 'average_anime_rating'
                })
                numeric_cols = ['user_rating', 'average_anime_rating', 'episodes', 'members']
                merged_df['episodes'] = pd.to_numeric(merged_df['episodes'], errors='coerce')
                correlation_matrix = merged_df[numeric_cols].corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax)
                ax.set_title('Correlation Matrix')
                st.pyplot(fig)

                # Display the correlation matrix
                st.write("#### Correlation Matrix")
                st.dataframe(correlation_matrix) 

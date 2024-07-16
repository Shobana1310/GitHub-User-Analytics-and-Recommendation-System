
import pymongo
import re
import time
import joblib
import io
import nltk
import requests
import json
import time
from bs4 import BeautifulSoup
from io import BytesIO
import pandas as pd
import datetime
from pathlib import Path
import streamlit as st
import hydralit_components as hc
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import hstack
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from streamlit_extras.mention import mention
from sqlalchemy import create_engine
from sqlalchemy import text
from nltk.data import find
import spacy
import urllib.parse
import joblib
import pyodbc
import en_core_web_sm
nlp = en_core_web_sm.load()

def download_nltk_data():
    packages = ['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            if package == 'punkt':
                find('tokenizers/punkt')
            else:
                find(package)
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_data()

api_key =st.secrets["github_api_key_6"]
headers = {"Authorization": f"token {api_key}"}


mongo_atlas_user_name =st.secrets["MONGO_ATLAS_USER_NAME"]
mongo_atlas_password = st.secrets["MONGO_ATLAS_PASSWORD"]
client=pymongo.MongoClient(f"mongodb+srv://{mongo_atlas_user_name}:{mongo_atlas_password}@cluster0.ehfepgy.mongodb.net/?retryWrites=true&w=majority")
db = client["github"]
collection=db["github_user_details"] 

server = st.secrets["SERVER"]
database = st.secrets["DATABASE"]
username = st.secrets["USERNAME"]
password = st.secrets["AZURE_PASSWORD"]
driver = 'ODBC Driver 17 for SQL Server'

params = urllib.parse.quote_plus(f"""
Driver={driver};
Server=tcp:{server},1433;
Database={database};
Uid={username};
Pwd={password};
Encrypt=yes;
TrustServerCertificate=no;
Connection Timeout=30;
""")

conn_str = f'mssql+pyodbc:///?autocommit=true&odbc_connect={params}'

engine = create_engine(conn_str, echo=True)

# mysql_password = st.secrets["MYSQL_PASSWORD"] 
# engine=create_engine(f"mysql+pymysql://root:{mysql_password}@localhost:3306/github")

def get_user_details(user_name):
    user_url = f'https://api.github.com/users/{user_name}'
    response = requests.get(user_url, headers=headers)
    user_json= response.json()
    starred_url = user_json['starred_url'].replace('{/owner}{/repo}', '')
    response = requests.get(starred_url)
    star_list=[]
    if response.status_code == 200:
        starred_repos = response.json()
        for repo in starred_repos:
            star_list.append(repo['full_name'])
    else:
        print("Failed to fetch starred repositories. Status code:", response.status_code)
    if not star_list:
        star_list = None
    data_list=[{
            'login':user_json['login'],
            'name':user_json['name'],
            'bio':user_json['bio'],
            'company':user_json['company'],
            'location':user_json['location'],
            'company':user_json['company'],
            'email':user_json['email'],
            'public_repos':user_json['public_repos'],
            'following_count':user_json['following'],
            'followers_count':user_json['followers'],
            'created_at':user_json['created_at'],
            'avatar_url':user_json['avatar_url'],
            'profile_url':user_json['html_url'],
    }]
    if star_list is not None:
        data_list[0]['user_starred_repo'] = ','.join(star_list)
    else:
        data_list[0]['user_starred_repo'] = None
    return data_list


def scrape_repository_readme(repo_full_name):
        repo_url = f"https://github.com/{repo_full_name}"
        response = requests.get(repo_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            readme_section = soup.find('article', class_='markdown-body')
            if readme_section:
                text = readme_section.get_text(separator=' ')
                
                
                text = re.sub(r'http\S+', '', text)
                text = re.sub(r'#+|\*{1,3}', '', text)
                text = re.sub(r'[^\w\s]', ' ', text)
                text = re.sub(r'\d', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                text = re.sub(r'\[.*?\]', '', text)
                
                lines = text.split(' ')
                cleaned_lines = []
                for line in lines:
                    if not re.search(r'\([^\)]+\)\s*[-—]\s*\w+', line):
                        cleaned_lines.append(line)
                cleaned_text = ' '.join(cleaned_lines).strip()
                
                if len(cleaned_text) > 3000:
                    cleaned_text = cleaned_text[:3000]
                
                return cleaned_text
            else:
                print(f"README not found for repository: {repo_full_name}")
                return None
        else:
            print(f"Failed to fetch repository: {repo_full_name}")
            return None
def get_all_repos(user_name):
        user_repos_url = f"https://api.github.com/users/{user_name}/repos"
        repos = []
        page = 1
        per_page = 100 

        while True:
            params = {'page': page, 'per_page': per_page}
            response = requests.get(user_repos_url, params=params)
            
            if response.status_code == 403:  # Rate limit error
                reset_time = int(response.headers.get('X-RateLimit-Reset'))
                sleep_time = reset_time - int(time.time()) + 5  # Adding a buffer of 5 seconds
                time.sleep(sleep_time)
                continue
            
            response_json = response.json()
            
            if not response_json:
                break
            
            repos.extend(response_json)
            page += 1
        
        return repos

def get_all_readme_details(user_name):
        user_repos = get_all_repos(user_name)
        readme_data = []

        for repo in user_repos:
            repo_full_name = repo['full_name']
            readme = scrape_repository_readme(repo_full_name)

            if readme is not None:
                readme_data.append({
                    'login': user_name,
                    'repos_name': repo["name"],
                    'repo_url': repo["html_url"],
                    'readme': readme
                })
        
        return readme_data



def fetch_repository_details(repo):
        commits_count = 0
        commits_url = repo['commits_url'].replace('{/sha}', '') 
        response = requests.get(commits_url, headers=headers)
        if response.status_code == 200:
            commits_count = len(response.json())

        language_url = repo['languages_url']
        response = requests.get(language_url, headers=headers)
        languages_used = ""
        languages_list = []  
        if response.status_code == 200:
            languages_data = response.json()
            languages_list = list(languages_data.keys())
            languages_used = ','.join([f"{lang} ({languages_data[lang]})" for lang in languages_list])

        return commits_count, languages_used, languages_list

def get_all_repository_details(user_name):
    
    user_url = f'https://api.github.com/users/{user_name}'
    response = requests.get(user_url, headers=headers)
    user_json = response.json()
    repos_url = user_json['repos_url']
    
    repository_data = []
    page = 1
    per_page = 100  

    while True:
    
        params = {'page': page, 'per_page': per_page}
        response = requests.get(repos_url, headers=headers, params=params)
        response_json = response.json()
        
        if not response_json:
            break
        
        for repo in response_json:
            commits_count, languages_used, languages_list = fetch_repository_details(repo)
            
            repository_data.append({
                'login': repo['owner']['login'],
                'repo_id': repo['id'],
                'repos_name': repo["name"],
                'Language_used': repo['language'],
                'repo_url': repo["html_url"],
                'pushed_at': repo['pushed_at'],
                'size': repo['size'],
                'repos_description': repo["description"],
                'repo_created_at': repo["created_at"],
                'languages_with_count': languages_used,
                'languages_list': ' '.join(languages_list),
                'forks_count': repo["forks_count"],
                'open_issues_count': repo["open_issues"],
                'updated_at': repo['updated_at'],
                'Stargazers': repo['stargazers_count'],
                'Watchers_Counts': repo['watchers_count'],
                'commit_count': commits_count
            })
        
        page += 1
    
    return repository_data

def get_all_detailsof_user(user_name):
    try:
        user=get_user_details(user_name)
        repository=get_all_repository_details(user_name)
        readme=get_all_readme_details(user_name)
        collection.insert_one({
                'user_data': user,
                'repository_data': repository,
                'readme_data':readme})
        print(f"Data inserted for user: {user_name}")
    except Exception as e:
        print(f"Failed to process user {user_name}. Error: {e}")

def update_user_data(user_name):
    try:
        user = get_user_details(user_name)
        repository = get_all_repository_details(user_name)
        readme = get_all_readme_details(user_name)
        
        existing_user = collection.find_one({'user_data.login':user_name})
        
        if existing_user:
            collection.update_one(
           {'user_data.login': user_name}, 
           {'$set': {'user_data': user, 'repository_data': repository,'readme_data':readme}})
            
            print(f"Data updated for user: {user_name}")
    except Exception as e:
        print(f"Failed to process user {user_name}. Error: {e}")


def create_and_insert_data(engine, table_name, create_table_query,file_path):
    try:
        create_table_query = text("""
                              CREATE TABLE IF NOT EXISTS repositories (
                                   login VARCHAR(255),
                                   repos_name VARCHAR(255),
                                   repo_url varchar(255) PRIMARY KEY,
                                   languages_list TEXT,
                                   repos_description TEXT,
                                   readme MEDIUMTEXT,
                                   primary_language VARCHAR(255))""")
        with engine.connect() as conn:
            conn.execute(create_table_query)

        df = joblib.load(file_path)

        df.to_sql(table_name, con=engine, if_exists='append', index=False)

        print(f"Data inserted into '{table_name}' successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

file_path = 'https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/vectorizer/df.joblib'

create_and_insert_data(engine, 'repositories', file_path)

def new_data_updation(new_user_df):
    try:
     query = text('select * from repositories')
     df = pd.read_sql_query(query, engine)
     
     updated_data = pd.concat([df, new_user_df], axis=0).reset_index(drop=True)

     delete_query = text('TRUNCATE TABLE repositories')
     with engine.connect() as conn:
          conn.execute(delete_query)

     updated_data.to_sql('repositories', con=engine, if_exists='append', index=False)

     print("Data updated successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")


def exist_data_updation(user_name, new_user_df):
    try:
        delete_query = text('DELETE FROM repositories WHERE login = :user_name')
        
        with engine.connect() as conn:
            conn.execute(delete_query, {'user_name': user_name})
            conn.commit()  
            
            new_user_df.to_sql('repositories', con=conn, if_exists='append', index=False)
            
            print("Data updated successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def generate_description_from_readme(row):
    if pd.isnull(row['repos_description']):
        return row['readme'][:255]  
    return row['repos_description']

def word_tokanize(text):
    words = nltk.word_tokenize(text) 
    return ' '.join(words)

def clean_the_text(text):
     text = text.lower()
     text = re.sub(r'\bimg\b', '', text)
     text = re.sub(r'http\S+', '', text)
     text = re.sub(r'<.*?>', '', text)
     text = re.sub(r'[-_]', ' ', text)
     text = re.sub(r'[^a-zA-Z\s]', '', text)
     text = re.sub(r'\s+', ' ', text).strip()
     return ''.join(text)

def remove_stopwords(text):
    doc=nlp(text)
    filtered_words=[i.text for i in doc if not i.is_stop and not i.is_punct and len(i.text) > 2 and i.is_alpha]
    return ' '.join(filtered_words)


def lemmatization(text):
    doc=nlp(text)
    words=[]
    for i in doc:
        words.append(i.lemma_)
    return ' '.join(words)

def preprocess_data(df_column):
    df_column=df_column.apply(word_tokanize)
    df_column=df_column.apply(clean_the_text)
    df_column=df_column.apply(remove_stopwords)
    df_column=df_column.apply(lemmatization)
    return df_column





st.set_page_config(layout="wide")
menu_data = [
    {'icon': "fa fa-home", 'label': "HOME"},
    {'icon': "fas fa-chart-line", 'label': "ANALYSIS"},
    {'icon': "far fa-lightbulb", 'label': "RECOMMENDATION"},
    {'icon': "fa fa-info-circle", 'label': "ABOUT"}]

over_theme = {'txc_inactive': '#FFFFFF', 'bg': '#07bff'}  
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=False, 
    sticky_nav=True,
    sticky_mode='pinned')


if menu_id=='HOME':

    text_to_center = "GITHUB RECOMMENDATION APP"
    logo_url="https://raw.githubusercontent.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/main/images/github-mark-white.png"
    st.markdown(f"""
    <div style='text-align: center;'>
        <img src="{logo_url}" alt="GitHub Logo" style="width: 75px; height: auto; margin-bottom: 20px;"> 
        <span style='background-image: linear-gradient(to right, white, #948C8C);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 50px; font-weight: bold; font-family: Arial, sans-serif;'>{text_to_center}</span>
    </div>
    """, unsafe_allow_html=True)
    st.write(' ')
    st.write(' ')
    with st.container():
        left_column,right_column=st.columns([2,3])
        with left_column:
            video_url = "https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/design%201.mp4"
            video_html = f"""
            <video width="85%" autoplay loop muted playsinline>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        with right_column:
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.subheader('This app helps developers and tech enthusiasts discover and analyze GitHub repositories using Readme files, repository names, descriptions, and programming languages. It offers precise and relevant recommendations tailored to your interests and needs')

    st.write(' ')
    st.write(' ')
    st.markdown(f"<h1 style='color: #A118E6;'>Features</h1>", unsafe_allow_html=True)

    st.write(' ')
    col1,col2=st.columns(2)
    with col1:
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/search-engine.png'
        st.image(image_path,width=120)
        st.subheader('Content-Based Recommendation Algorithm')
    
    with col2:
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/book.png'
        st.image(image_path,width=120)
        st.subheader('The app parses and analyzes Readme files to understand the project purpose, usage, and key features, providing deeper insights into each repository')

    st.write(' ')
    st.write(' ')
    st.write(' ')
    col1,col2=st.columns(2)
    with col1:
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/file-and-folder.png'
        st.image(image_path,width=120)
        st.subheader('By considering the repository names, the app identifies relevant projects that match specific keywords and themes')
    
    with col2:
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/code.png'
        st.image(image_path,width=120)
        st.subheader('The programming languages used in repositories are considered to match users with projects that align with their coding skills and preferences')

    st.write(' ')
    st.write(' ')
    
    st.markdown(f"<h1 style='color: #A118E6;'>Technical Deep Dives-Backend Architecture</h1>", unsafe_allow_html=True)

    st.write(' ')
    st.write(' ')
    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.write(' ')
        st.write(' ')
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/python.png'
        st.image(image_path,width=120)
    with col2:
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/mongodb.png'
        st.image(image_path,width=200)
    with col3:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/beautiful-soup.png'
        st.image(image_path,width=220)
    with col4:
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/github%20api.png'
        st.image(image_path,width=220)

    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.write(' ')
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/scikit-learn.png'
        st.image(image_path,width=220)
    with col2:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/plotly.png'
        st.image(image_path,width=250)
    with col3:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/panda.png'
        st.image(image_path,width=160)
    with col4:
        image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/azure%20logo.png'
        st.image(image_path,width=280)
    
    st.header(':violet[See The Github Recent Events,Click Here 👇]')

    st.markdown("<a style='font-size: 30px;' href='https://github.com/events' target='_blank'>GitHub Events</a>", unsafe_allow_html=True)




    
if menu_id=='ABOUT':
    st.header(':green[OVERVIEW]')
    st.subheader('Our project aims to suggest GitHub repositories to users based on their interests and activity, using content-based filtering.')

    st.write(' ')
    st.write(' ')
    st.header(':green[HOW IT WORKS]')
    st.write(' ')
    st.write(' ')
    col1,col2=st.columns(2)
    with col1:
        st.header(':violet[DATA COLLECTION]')
        video_url = "https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/data%20collection.mp4"
        video_html = f"""
            <video width="50%" autoplay loop muted playsinline>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
        st.markdown(video_html, unsafe_allow_html=True)
        st.subheader('We gather data from the users repositories, including names, descriptions, README files, and programming languages.')
    col1,col2=st.columns(2)
    with col2:
        st.header(':violet[FEATURE EXTRACTION]')
        video_url = "https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/feature.mp4"
        video_html = f"""
            <video width="50%" autoplay loop muted playsinline>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
        st.markdown(video_html, unsafe_allow_html=True)
        st.subheader('Processes this data to extract meaningful features that represent the content of each repository.')
    col1,col2=st.columns(2)
    with col1:
        st.header(':violet[SIMILARITY CALCULATION]')
        video_url = "https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/similarity.mp4"
        video_html = f"""
            <video width="50%" autoplay loop muted playsinline>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
        st.markdown(video_html, unsafe_allow_html=True)
        st.subheader('Computes similarity scores between the users repositories and other repositories on GitHub using techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity.')
    col1,col2=st.columns(2)
    with col2:
        st.header(':violet[RECOMMENDATION GENERATION]')
        video_url = "https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/engine.mp4"
        video_html = f"""
            <video width="50%" autoplay loop muted playsinline>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
        st.markdown(video_html, unsafe_allow_html=True)
        st.subheader('Rank Repositories based on their similarity scores and suggest the top ones.')

    st.write(' ')
    st.write(' ')
    underline = "__________" 
    st.markdown(underline)
    st.header('About Me')
    st.subheader('👋 Hi there! I am Shobana, A Passionate into Data Science And Business Solutions. With a Strong Foundation In Data Analysis And Machine Learning, I Thrive On Uncovering Actionable Insights From Complex Datasets to Drive Strategic Decision-making And Enhance Business Performance. I Am Dedicated To Continuous Learning, Always Staying On Latest Trends And Technologies In The Field')
    
    st.write(':blue[**LinkedIn**]')  
    linkedin_logo="https://img.icons8.com/fluent/48/000000/linkedin.png"      
    linkedin_url="https://www.linkedin.com/in/shobana-v-534b472a2"
    st.markdown(f"[![LinkedIn]({linkedin_logo})]({linkedin_url})")








if menu_id =='ANALYSIS':
    col1,col2=st.columns(2)
    with col1:
        video_url = "https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/analystics.mp4"
        video_html = f"""
            <video width="75%" autoplay loop muted playsinline>
                <source src="{video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
        st.markdown(video_html, unsafe_allow_html=True)
    with col2:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        font_family = "Georgia"
        st.markdown(f"<h1 style='font-size: 55px; font-family: {font_family};'>You can review and analyze your GitHub activity here 🌱</h1>", unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        user_name = st.text_input(label="GitHub Username", placeholder="Enter GitHub Username", label_visibility='hidden')

    if user_name:
        login_names = []
        user_data_cursor = list(collection.find({}, {"_id": 0, "user_data": 1}))
        total_documents = len(user_data_cursor)

        progress_bar = st.progress(0)
        progress_text = st.empty()  # Create an empty container for the progress text

        start_time = time.time()

        for idx, i in enumerate(user_data_cursor):
            for j in range(len(i["user_data"])):
                login_names.append(i["user_data"][j]["login"])

            # Update progress bar
            progress = (idx + 1) / total_documents
            progress_bar.progress(progress)

            # Calculate ETR and display
            elapsed_time = time.time() - start_time
            if progress > 0:  # Avoid division by zero
                etr = (elapsed_time / progress) * (1 - progress)
                progress_text.text(f"Progress: {progress:.2%}, Estimated Time Remaining: {etr:.2f} seconds")

            # Simulate delay for demonstration
            time.sleep(0.1)

        if user_name in login_names:
            with st.spinner("User name is Already Exist In The Database, It's Updating Now..."):
                update_user_data(user_name)
        else:
            with st.spinner("Collecting Your Data..."):
                get_all_detailsof_user(user_name)


    if user_name:
        def get_repo_df():
            data1=[]                                                
            for i in collection.find({"user_data.login": user_name}, {"_id": 0, "repository_data": 1}):
                for j in range(len(i["repository_data"])):
                    data1.append(i["repository_data"][j])
            repo_df=pd.DataFrame(data1)
            return repo_df
        repo_df=get_repo_df()

        def get_user_df():
            data2=[]
            for i in collection.find({"user_data.login": user_name}, {"_id": 0, "user_data": 1}):
                for j in range(len(i["user_data"])):
                    data2.append(i["user_data"][j])
            user_df=pd.DataFrame(data2) 
            return user_df
        user_df=get_user_df()
        
        def get_readme_df():
            data3=[]
            for i in collection.find({"user_data.login": user_name}, {"_id": 0, "readme_data": 1}):   
                for j in range(len(i["readme_data"])):
                    data3.append(i["readme_data"][j])
            readme_df=pd.DataFrame(data3)
            return readme_df
        readme_df=get_readme_df()

        repo_df['pushed_at']=pd.to_datetime(repo_df['pushed_at'])
        repo_df['repo_created_at']=pd.to_datetime(repo_df['repo_created_at'])
        repo_df['updated_at']=pd.to_datetime(repo_df['updated_at'])
        result = pd.merge(repo_df, readme_df, on=['login','repos_name','repo_url'], how='inner')
        new_user_df=result[['login','repos_name','repo_url','languages_list','repos_description','readme']]
        new_user_df['primary_language'] = new_user_df['languages_list'].apply(lambda x: x.split(' ')[0])
        new_user_df['repos_description'] = new_user_df.apply(generate_description_from_readme, axis=1)
        new_user_df= new_user_df.drop_duplicates(subset=['repo_url'])

        @st.cache_data
        def prepare_the_data():
            new_user_df['repos_name'] = preprocess_data(new_user_df['repos_name'])
            new_user_df['repos_description'] = preprocess_data(new_user_df['repos_description'])
            new_user_df['readme'] = preprocess_data(new_user_df['readme'])
            new_user_df['languages_list']=preprocess_data(new_user_df['languages_list'])
        prepare_the_data()

    
        st.write(' ')
        st.write(' ')
        underline = "__________" 
        st.markdown(f"{underline}")
        col1,col2=st.columns(2)
        with col1:
            st.write(' ')
            image_path =user_df['avatar_url'][0]
            st.markdown(f"""
                    <style>
                    .round-image {{
                        width: 250px;
                        height: 250px;
                        border-radius: 50%;
                        object-fit: cover;
                    }}
                    </style>
                    <img src="{image_path}" class="round-image">
                    """,
                    unsafe_allow_html=True)
            st.write(' ')
            st.write(' ')
            user_name = user_df['name'][0]
            st.subheader(user_name)
            bio = user_df['bio'][0]
            if bio is not None:
                st.subheader(bio)
        
        with col2:
            st.write(' ')
            st.write(' ')
            public_repos=user_df['public_repos'][0]
            followers_count = user_df['followers_count'][0]
            following_count = user_df['following_count'][0]
            company = user_df['company'][0]
            location = user_df['location'][0]
            timestamp = pd.Timestamp(user_df['created_at'][0])
            formatted_date = timestamp.strftime('%b %d,%Y')


            st.markdown(f"<p style='font-size:25px;'>🗳️ <strong>{public_repos} Public Repos</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:25px;'>👥 <strong>{followers_count} Followers</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:25px;'>👥 <strong>{following_count} Following</strong></p>", unsafe_allow_html=True)


            if company:
                st.markdown(f"<p style='font-size:25px;'>🏢 <strong>Works At {company}</strong></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:25px;'>⏳ <strong>Joined Github At {formatted_date}</strong></p>", unsafe_allow_html=True)
            if location:
                st.markdown(f"<h2 style='font-size:25px;'>📍{location}</h2>", unsafe_allow_html=True)


        underline = "__________" 
        st.markdown(f"{underline}")
        if public_repos > 5:
            def repo_trend():
                repo_df['repo_created_at'] = pd.to_datetime(repo_df['repo_created_at'], errors='coerce')
                repo_df['YearMonth'] = repo_df['repo_created_at'].dt.to_period('M').dt.strftime('%Y-%m')
                df = repo_df.groupby('YearMonth').size().reset_index(name='repo_count')
                fig = px.line(df, x='YearMonth', y='repo_count', title='Repository Creation Trend', height=400)
                fig.update_traces(line=dict(color='orange'))
                st.plotly_chart(fig, use_container_width=True)
            repo_trend()
        
        def commit_trend():
            repo_df['repo_created_at'] = pd.to_datetime(repo_df['repo_created_at'], errors='coerce')
            repo_df['YearMonth'] = repo_df['repo_created_at'].dt.to_period('M').dt.strftime('%Y-%m')
            monthly_commits = repo_df.groupby('YearMonth')['commit_count'].sum().reset_index()
            fig = px.area(monthly_commits, x='YearMonth', y='commit_count', title='Commit Trend Over The Time', markers=True)
            fig.update_traces(line=dict(color='orange'))
            st.plotly_chart(fig, use_container_width=True)
        commit_trend()



        def languages_chart():
            def extract_languages(row):
                languages = row.split(',')
                language_count = {}
                for lang in languages:
                    parts = lang.split('(')
                    if len(parts) == 2: 
                        lang_name = parts[0].strip()
                        try:
                            count = int(parts[1].strip(')'))
                            language_count[lang_name] = count
                        except ValueError:
                            print(f"Ignoring invalid count for language: {lang}")
                    else:
                        print(f"Ignoring invalid language format: {lang}")
                return language_count
            
            repo_df['language_counts'] = repo_df['languages_with_count'].apply(extract_languages)
            
            total_counts = {}
            for counts in repo_df['language_counts']:
                for lang, count in counts.items():
                    total_counts[lang] = total_counts.get(lang, 0) + count
            
            total_df = pd.DataFrame(total_counts.items(), columns=['Language', 'Count'])
            total_df = total_df.nlargest(5, 'Count')  
            fig = px.pie(total_df, values='Count', names='Language', title='Top 5 Languages Using in Repositories', hole=0.5,
                        color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
            st.plotly_chart(fig, use_container_width=True)


        def watchers_chart():
            filter_df = repo_df.groupby('repos_name')['Watchers_Counts'].sum().nlargest(5).reset_index()
            fig = px.pie(filter_df, 
                        values='Watchers_Counts',
                        names='repos_name',
                        title='Top Repositories Has Highest Watchers',
                        hole=0.5,
                        labels={'Watchers_Counts': 'Watcher Count', 'repos_name': 'Repository Name'},
                        template='plotly_white',color_discrete_sequence=px.colors.sequential.Aggrnyl_r)

            st.plotly_chart(fig, use_container_width=True)
        def show_language_and_watchers_chart():
            sum=0
            for i in repo_df['Watchers_Counts']:
                sum+=i
            if sum > 5:
                col1,col2=st.columns(2)
                with col1:
                    languages_chart()
                with col2:
                    watchers_chart()
            else:
                languages_chart()
        show_language_and_watchers_chart()
        
        def forkscount_chart():
            sum=0
            for i in repo_df['forks_count']:
                sum+=i
            if sum >5 and sum <=100:
                filter_df = repo_df.loc[repo_df['forks_count'] >= 1]
                sorted_df = filter_df.sort_values(by='forks_count', ascending=False)
                fig = px.bar(sorted_df, 
                            x='repos_name', 
                            y='forks_count', 
                            title='Repositories by Forks Count',
                            labels={'forks_count': 'Forks Count', 'repos_name': 'Repository Name'},
                            template='plotly_white',color_discrete_sequence=px.colors.sequential.Oranges_r)
                st.plotly_chart(fig, use_container_width=True)
            elif sum >100:
                filter_df = repo_df.loc[repo_df['forks_count'] >= 1]
                sorted_df = filter_df.sort_values(by='forks_count', ascending=False)
                top_10_df = sorted_df.head(10)
                fig = px.bar(top_10_df, 
                    x='repos_name', 
                    y='forks_count', 
                    title='Top 10 Repositories by Forks Count',
                    color='forks_count',
                    labels={'forks_count': 'Forks Count', 'repos_name': 'Repository Name'},
                    template='plotly_white',
                    color_discrete_sequence=px.colors.sequential.Oranges_r)

                st.plotly_chart(fig, use_container_width=True)
        forkscount_chart()

       

        st.subheader(':red[**Mandatory ⚠️  Add Your Data In The Recommendation System, Don’t Go Recommendation Without Click Here**]')
        if st.button("Click Here", type="primary"):
            query = text('select * from repositories')
            df = pd.read_sql_query(query, engine)
            existing_usernames = df['login'].unique()
            new_usernames = new_user_df['login'].unique()
            total_usernames = len(new_usernames)
            progress_bar = st.progress(0)
            progress_text = st.empty()

            start_time = time.time()

            for idx, user_name in enumerate(new_usernames):
                if user_name in existing_usernames:
                    with st.spinner("Updating Your Data..."):
                        exist_data_updation(user_name, new_user_df[new_user_df['login'] == user_name])
                else:
                    with st.spinner("Adding Your Data..."):
                        new_data_updation(new_user_df[new_user_df['login'] == user_name])

                progress = (idx + 1) / total_usernames
                progress_bar.progress(progress)

                elapsed_time = time.time() - start_time
                if progress > 0:
                    etr = (elapsed_time / progress) * (1 - progress)
                    progress_text.text(f"Progress: {progress:.2%}, Estimated Time Remaining: {etr:.2f} seconds")

                time.sleep(0.1)

            st.success('Added Successfully ✅')
        
    
        st.write(' ')
        st.write(' ')


        def github_icon(repo_url):
            return f'<a href="{repo_url}" target="_blank"><img src="https://img.icons8.com/ios-filled/50/ffffff/github.png" alt="GitHub" style="width:20px;height:20px;margin-right:10px;"> {repo_url}</a>'

        starred_repos = user_df['user_starred_repo'][0]
        if starred_repos is not None:
            starred_repos = user_df['user_starred_repo'][0].split(',')
            count=len(starred_repos)
            with st.expander(f"**⭐ Your Starred Repositories ({count})**"):
                for idx, repo in enumerate(starred_repos, 1):
                    repo_url = f"https://github.com/{repo}"
                    st.markdown(github_icon(repo_url), unsafe_allow_html=True)
                
                

if menu_id=='RECOMMENDATION':
    st.write(' ')
    st.write(' ')
    st.write(' ')
    col1,col2=st.columns([3,2])
    with col1:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        font_family = "Georgia"
        st.markdown(f"<h1 style='font-size: 55px; font-family: {font_family};'>GET YOUR RECOMMENDATION</h1>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
        st.subheader(':green[Get Recommendation of high-quality projects that perfectly match your interests based on your Readme files and descriptions and your programming languages]')
    with col2:
         st.write(' ')
         st.write(' ')
         st.write(' ')
         st.write(' ')
         st.write(' ')
         image_path ='https://github.com/Shobana1310/GitHub-User-Analytics-and-Recommendation-System/raw/main/images/DataScientist1-ezgif.com-video-to-gif-converter.gif'
         st.image(image_path)
    col1,col2=st.columns(2)   
    with col1:
      st.write('')
      st.write('')
      user_name = st.text_input(label="GitHub Username", placeholder="Enter GitHub Username", label_visibility='hidden')
      st.write(' ')
      st.write(' ')
      st.write(' ')
      st.write(' ')
      st.write(' ')
      st.write(' ')
      st.write(' ')
      st.write(' ')

    query=text('select * from repositories')
    df= pd.read_sql_query(query, engine)
    def model_setup():
        name_vectorizer = TfidfVectorizer(stop_words='english')
        langugage_vectorizer = TfidfVectorizer(stop_words='english')
        readme_vectorizer = TfidfVectorizer(stop_words='english')
        description_vectorizer = TfidfVectorizer(stop_words='english')


        name_tfidf_matrix = name_vectorizer.fit_transform(df['repos_name'])
        langugage_tfidf_matrix = langugage_vectorizer.fit_transform(df['languages_list'])
        description_tfidf_matrix = description_vectorizer.fit_transform(df['repos_description'])
        readme_tfidf_matrix = readme_vectorizer.fit_transform(df['readme'])


        combined_tfidf_matrix = hstack([name_tfidf_matrix,langugage_tfidf_matrix,description_tfidf_matrix,readme_tfidf_matrix])
        cosine_sim = cosine_similarity(combined_tfidf_matrix, combined_tfidf_matrix)

        repo_indices = pd.Series(df.index, index=df['repo_url']).drop_duplicates()
        user_repos = df.groupby('login')['repo_url'].apply(list)
        return cosine_sim,repo_indices,user_repos
    cosine_sim,repo_indices,user_repos=model_setup()

    def get_recommendations_for_user(user_name, cosine_sim=cosine_sim, repo_indices=repo_indices, df=df, user_repos=user_repos):
        if user_name not in user_repos:
            return []

        user_repo_links = user_repos[user_name]
        if isinstance(repo_indices, pd.Series):
             repo_indices = repo_indices.to_dict()
        repo_indices_list = [repo_indices[repo_link] for repo_link in user_repo_links]

        sim_scores = cosine_sim[repo_indices_list].mean(axis=0)
        
        sim_scores = list(enumerate(sim_scores))
        
        user_repo_indices_set = set(repo_indices_list)
        sim_scores = [score for score in sim_scores if score[0] not in user_repo_indices_set]
        recommended_users = set()

        filtered_sim_scores = []
        for score in sim_scores:
            repo_url = df['repo_url'].iloc[score[0]]
            user = extract_user_from_repo_url(repo_url)
            if user not in recommended_users:
                filtered_sim_scores.append(score)
                recommended_users.add(user)

        filtered_sim_scores = sorted(filtered_sim_scores, key=lambda x: x[1], reverse=True)

        filtered_sim_scores = filtered_sim_scores[:15]

        similar_repo_indices = [i[0] for i in filtered_sim_scores]

        return df['repo_url'].iloc[similar_repo_indices].tolist()


    def extract_user_from_repo_url(repo_url):
       return repo_url.split('/')[3]

    def get_user_details(username):
        data = []
        for i in collection.find({"user_data.login": username}, {"_id": 0, "user_data": 1}):
            for j in range(len(i["user_data"])):
                data.append(i["user_data"][j])
        if not data:
            return None, None  
        user_df = pd.DataFrame(data)
        avatar_url = user_df['avatar_url'].iloc[0]
        bio = user_df['bio'].apply(lambda x: x if pd.notnull(x) else "Bio not available").iloc[0]
        return avatar_url, bio


    recommendations = get_recommendations_for_user(user_name)



    rows = [recommendations[i:i+3] for i in range(0, len(recommendations), 3)]


    for row in rows:
        cols = st.columns(3)
        for col, repo_url in zip(cols, row):
            with col:
                username = extract_user_from_repo_url(repo_url)
                avatar_url, bio = get_user_details(username)

                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{avatar_url}" style="border-radius: 50%; width: 230px; height: 230px;">
                    </div>
                    """,unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style='text-align: center;'>
                        <span style='margin-left: 15px;'><strong>{username}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)

                html_code = f"""
                    <style>
                        .info-container {{
                            margin: 10px 0;
                            padding: 5px;
                            border: 1px solid #ddd;
                            border-radius: 4px;
                            background-color: transparent;
                            color: white; /* Set text color to white */
                            transition: box-shadow 0.3s;
                            position: relative;
                        }}

                        .info-container:hover {{
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                        }}
                        
                        .repo-url {{
                            max-height: 0;
                            overflow: hidden;
                            transition: max-height 0.5s ease-out, opacity 0.5s ease-out;
                            opacity: 0;
                            background-color: transparent;
                        }}
                        
                        .repo-url-content {{
                            overflow-y: auto;
                            transition: max-height 0.5s ease-out;
                            max-height: 0;
                            background-color: transparent;
                            color: #D4D214; /* Set text color to #D4D214 */
                        }}
                        
                        .info-container:hover .repo-url {{
                            max-height: 100px;  /* Adjust based on content */
                            opacity: 1;
                        }}
                        
                        .info-container:hover .repo-url-content {{
                            max-height: 300px;  /* Adjust based on content */
                        }}
                        
                        .info-container a {{
                            color: #C7C7C0; /* Set link color to #C7C7C0 */
                        }}
                    </style>
                    <div class="info-container">
                        <p>{bio}</p>
                        <div class="repo-url">
                            <div class="repo-url-content">
                                <p><strong>Repo URL:</strong> <a href="{repo_url}" target="_blank">{repo_url}</a></p>
                            </div>
                        </div>
                    </div>
                """

                st.components.v1.html(html_code, height=240)
                st.markdown("<br><br>", unsafe_allow_html=True)

      














































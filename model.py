import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

movies = movies[['id','title','overview','genres','keywords']]

movies.dropna(inplace=True)

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']

new_df = movies[['id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

recommend("Avatar")

print(new_df.head())
print(similarity.shape)

import pickle

pickle.dump(new_df.to_dict(), open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))
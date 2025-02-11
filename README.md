# Netflix Data Analysis 📊

## Overview
This project performs an in-depth analysis of Netflix's content dataset, exploring trends, distributions, and key insights about the platform's shows and movies.

## Dataset 📂
- The dataset contains information about Netflix titles, including type, release year, director, cast, country, duration, genre, and rating.
- File: `Netflix-Data-Analysis.csv`

## Features & Analysis 🔍
### 1️⃣ Data Overview & Missing Values
```python
missing_values = content_df.isnull().sum()
sns.barplot(x=missing_values.index, y=missing_values.values)
plt.xticks(rotation=90)
plt.title('Missing Values in Each Column')
plt.show()
```

### 2️⃣ Content Distribution
```python
sns.countplot(x='type', data=content_df)
plt.title('Distribution of Content Types')
plt.show()
```

### 3️⃣ Top Directors & Cast
```python
# Top Directors
top_directors = content_df['director'].value_counts().head(10)
sns.barplot(y=top_directors.index, x=top_directors.values)
plt.title('Top 10 Directors by Number of Titles')
plt.show()

# Top Cast
cast_members = content_df['cast'].dropna().str.split(', ').explode()
top_cast = cast_members.value_counts().head(10)
sns.barplot(y=top_cast.index, x=top_cast.values)
plt.title('Top 10 Actors/Actresses')
plt.show()
```

### 4️⃣ Content by Country
```python
top_countries = content_df['country'].value_counts().head(10)
sns.barplot(y=top_countries.index, x=top_countries.values)
plt.title('Top 10 Countries by Number of Titles')
plt.show()
```

### 5️⃣ Release Year Analysis
```python
sns.histplot(content_df['release_year'], bins=20, kde=True)
plt.title('Release Year Distribution')
plt.show()
```

### 6️⃣ Ratings & Genres
```python
sns.countplot(y='rating', data=content_df, order=content_df['rating'].value_counts().index)
plt.title('Content Rating Distribution')
plt.show()
```

### 7️⃣ Duration Analysis
```python
# Movies
movies = content_df[content_df['type'] == 'Movie']
movies['duration'] = movies['duration'].str.replace(' min', '').astype(int)
sns.histplot(movies['duration'], bins=20)
plt.title('Movie Duration Distribution')
plt.show()

# TV Shows
tv_shows = content_df[content_df['type'] == 'TV Show']
tv_shows['seasons'] = tv_shows['duration'].str.replace(' Season', '').str.replace('s', '').astype(int)
sns.histplot(tv_shows['seasons'], bins=10)
plt.title('Number of Seasons Distribution for TV Shows')
plt.show()
```

### 8️⃣ Word Cloud of Titles
```python
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(content_df['title']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Titles')
plt.show()
```

### 9️⃣ Clustering: Movies by Duration & Release Year
```python
from sklearn.cluster import KMeans
import numpy as np

movies_filtered = movies.dropna(subset=['duration', 'release_year'])
X = movies_filtered[['duration', 'release_year']]
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
movies_filtered['cluster'] = kmeans.labels_

plt.figure(figsize=(12, 8))
sns.scatterplot(data=movies_filtered, x='release_year', y='duration', hue='cluster', palette='viridis')
plt.title('Clustering of Movies by Duration and Release Year')
plt.xlabel('Release Year')
plt.ylabel('Duration (minutes)')
plt.show()
```

### 🔟 Trend Analysis: Popular Categories Over Time
```python
df_exploded = content_df.assign(listed_in=content_df['listed_in'].str.split(', ')).explode('listed_in')
popular_years = df_exploded.groupby(['release_year', 'listed_in']).size().reset_index(name='count')
top_categories_by_year = popular_years.sort_values(['release_year', 'count'], ascending=[True, False]).groupby('release_year').head(3)

plt.figure(figsize=(14, 8))
sns.lineplot(data=top_categories_by_year, x='release_year', y='count', hue='listed_in', marker='o')
plt.title('Top 3 Popular Categories Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

## Getting Started 🚀
### Prerequisites
- Python 3.8+
- Jupyter Notebook or Google Colab
- Required Libraries:
  ```sh
  pip install pandas seaborn matplotlib plotly wordcloud scikit-learn
  ```

### Running the Notebook
1. Clone this repository:
   ```sh
   git clone https://github.com/your-username/Netflix-Data-Analysis.git
   ```
2. Open `Netflix_Analysis.ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells to analyze the dataset.

## Contributing 🤝
Feel free to open issues or submit pull requests to enhance the project.

## License 📝
This project is open-source and available under the MIT License.


import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
import random
import warnings
import pickle
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import hashlib
import locale

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['CACHE_DIR'] = 'cache'
app.config['MAX_CACHE_AGE_DAYS'] = 7  # Rebuild cache if older than 7 days


# Custom Jinja2 filter for formatting numbers with commas
def format_int(value):
    """Format integer with commas"""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


app.jinja_env.filters['intcomma'] = format_int

# Ensure cache directory exists
os.makedirs(app.config['CACHE_DIR'], exist_ok=True)


class BookRecommender:
    def __init__(self, books_csv='Books.csv', users_csv='Users.csv', ratings_csv='Ratings.csv', use_cache=True):
        """
        Initialize the recommender system with caching support
        """
        self.books_csv = books_csv
        self.users_csv = users_csv
        self.ratings_csv = ratings_csv
        self.use_cache = use_cache

        # Generate cache file names based on file hashes
        self.cache_prefix = self._generate_cache_prefix()

        # Initialize empty attributes
        self.df = None
        self.users_df = None
        self.ratings_df = None
        self.explicit_ratings = None
        self.book_ratings = None
        self.user_ratings = None
        self.user_item_dict = None
        self.popular_books = None
        self.title_to_index = None
        self.author_to_books = None
        self.category_to_books = None
        self.decade_to_books = None
        self.isbn_to_index = None
        self.user_profiles = None
        self.categories = None
        self.authors = None

        self.load_data()

    def _generate_cache_prefix(self):
        """Generate cache prefix based on file modification times and sizes"""
        try:
            files_info = []
            for file_path in [self.books_csv, self.users_csv, self.ratings_csv]:
                if os.path.exists(file_path):
                    stat = os.stat(file_path)
                    files_info.append(f"{file_path}:{stat.st_mtime}:{stat.st_size}")
                else:
                    files_info.append(f"{file_path}:missing")

            # Create hash from file info
            hash_input = "_".join(files_info)
            return hashlib.md5(hash_input.encode()).hexdigest()[:12]
        except:
            return "default"

    def _cache_file(self, name):
        """Get full cache file path"""
        return os.path.join(app.config['CACHE_DIR'], f"{self.cache_prefix}_{name}.pkl")

    def _is_cache_valid(self, cache_file):
        """Check if cache file exists and is not too old"""
        if not os.path.exists(cache_file):
            return False

        # Check age
        file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
        return file_age <= app.config['MAX_CACHE_AGE_DAYS']

    def _save_to_cache(self, data_dict):
        """Save data to cache files"""
        try:
            for name, data in data_dict.items():
                cache_file = self._cache_file(name)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Cache saved successfully")
            return True
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False

    def _load_from_cache(self, names):
        """Load data from cache files"""
        results = {}
        try:
            for name in names:
                cache_file = self._cache_file(name)
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        results[name] = pickle.load(f)
                else:
                    return None
            print("Cache loaded successfully")
            return results
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None

    def load_data(self):
        """Load and process data with caching"""
        print("=" * 60)
        print("Initializing Book Recommender System")
        print("=" * 60)

        # Try to load from cache first
        if self.use_cache:
            cache_data = self._load_from_cache([
                'metadata', 'indices', 'book_ratings',
                'user_ratings', 'user_item_dict', 'user_profiles'
            ])

            if cache_data:
                print("✓ Loaded from cache")
                self.df = cache_data['metadata']['df']
                self.users_df = cache_data['metadata']['users_df']
                self.ratings_df = cache_data['metadata']['ratings_df']
                self.explicit_ratings = cache_data['metadata']['explicit_ratings']
                self.popular_books = cache_data['metadata']['popular_books']
                self.categories = cache_data['metadata']['categories']
                self.authors = cache_data['metadata']['authors']

                self.title_to_index = cache_data['indices']['title_to_index']
                self.author_to_books = cache_data['indices']['author_to_books']
                self.category_to_books = cache_data['indices']['category_to_books']
                self.decade_to_books = cache_data['indices']['decade_to_books']
                self.isbn_to_index = cache_data['indices']['isbn_to_index']

                self.book_ratings = cache_data['book_ratings']
                self.user_ratings = cache_data['user_ratings']
                self.user_item_dict = cache_data['user_item_dict']
                self.user_profiles = cache_data['user_profiles']

                self.display_stats()
                return

        print("Loading from CSV files...")

        # Load raw data
        print("Loading book data...")
        self.df = pd.read_csv(self.books_csv, encoding='utf-8', low_memory=False)
        print(f"Loaded {len(self.df)} books")

        print("Loading user data...")
        self.users_df = pd.read_csv(self.users_csv, encoding='utf-8', low_memory=False)
        print(f"Loaded {len(self.users_df)} users")

        print("Loading ratings data...")
        self.ratings_df = pd.read_csv(self.ratings_csv, encoding='utf-8', low_memory=False)
        print(f"Loaded {len(self.ratings_df)} ratings")

        # Process data
        self._clean_data()
        self._prepare_indices()
        self._prepare_ratings_data()
        self._prepare_user_profiles()

        # Extract categories and authors for web interface
        self.categories = sorted(self.category_to_books.keys())
        self.authors = sorted(self.author_to_books.keys())[:100]  # Top 100 authors

        # Save to cache
        if self.use_cache:
            cache_dict = {
                'metadata': {
                    'df': self.df,
                    'users_df': self.users_df,
                    'ratings_df': self.ratings_df,
                    'explicit_ratings': self.explicit_ratings,
                    'popular_books': self.popular_books,
                    'categories': self.categories,
                    'authors': self.authors
                },
                'indices': {
                    'title_to_index': self.title_to_index,
                    'author_to_books': self.author_to_books,
                    'category_to_books': self.category_to_books,
                    'decade_to_books': self.decade_to_books,
                    'isbn_to_index': self.isbn_to_index
                },
                'book_ratings': self.book_ratings,
                'user_ratings': self.user_ratings,
                'user_item_dict': self.user_item_dict,
                'user_profiles': self.user_profiles
            }
            self._save_to_cache(cache_dict)

        self.display_stats()

    def _clean_data(self):
        """Clean and preprocess the book data"""
        print("Cleaning data...")

        # Basic cleaning
        self.df = self.df.dropna(subset=['Book-Title', 'Book-Author'])
        self.df['Book-Title'] = self.df['Book-Title'].fillna('').astype(str)
        self.df['Book-Author'] = self.df['Book-Author'].fillna('').astype(str)
        self.df['Publisher'] = self.df['Publisher'].fillna('').astype(str)

        # Handle year of publication
        self.df['Year-Of-Publication'] = pd.to_numeric(self.df['Year-Of-Publication'], errors='coerce')
        self.df['Decade'] = (self.df['Year-Of-Publication'] // 10) * 10

        # Lowercase for searching
        self.df['Title_Lower'] = self.df['Book-Title'].str.lower()
        self.df['Author_Lower'] = self.df['Book-Author'].str.lower()

        # Categorization
        print("Categorizing books...")
        self.df['Category'] = self.df.apply(self._categorize_book, axis=1)
        print(f"Cleaned {len(self.df)} books")

    def _categorize_book(self, row):
        """Simple keyword-based categorization"""
        title = str(row['Book-Title']).lower()
        author = str(row['Book-Author']).lower()

        categories = {
            'Mystery/Crime': ['mystery', 'murder', 'crime', 'detective', 'sherlock'],
            'Romance': ['love', 'romance', 'heart', 'kiss', 'wedding'],
            'Fantasy': ['fantasy', 'magic', 'dragon', 'wizard', 'witch'],
            'Science Fiction': ['science', 'tech', 'space', 'planet', 'robot', 'sci-fi'],
            'Historical': ['history', 'war', 'battle', 'historical', 'century'],
            'Horror': ['horror', 'ghost', 'haunted', 'fear', 'terror'],
            'Biography': ['biography', 'memoir', 'autobiography', 'life of'],
            'Self-Help': ['self-help', 'motivation', 'success', 'habits', 'mindset']
        }

        for category, keywords in categories.items():
            if any(keyword in title for keyword in keywords):
                return category

        # Author-based categorization
        author_categories = {
            'grisham': 'Legal Thriller',
            'christie': 'Mystery/Crime',
            'clancy': 'Techno-Thriller',
            'king': 'Horror/Thriller',
            'tolkien': 'Fantasy',
            'asimov': 'Science Fiction',
            'austen': 'Classic Romance',
            'hemingway': 'Classic Literature'
        }

        for auth_keyword, category in author_categories.items():
            if auth_keyword in author:
                return category

        return 'General Fiction'

    def _prepare_indices(self):
        """Prepare efficient search indices"""
        print("Building search indices...")

        self.title_to_index = {}
        self.author_to_books = defaultdict(list)
        self.category_to_books = defaultdict(list)
        self.decade_to_books = defaultdict(list)
        self.isbn_to_index = {}

        for idx, row in self.df.iterrows():
            title = str(row['Book-Title']).strip().lower()
            author = str(row['Book-Author']).strip().lower()
            category = row['Category']
            decade = row['Decade']
            isbn = str(row['ISBN']).strip()

            if title:
                self.title_to_index[title] = idx
            if author:
                self.author_to_books[author].append(idx)
            if pd.notna(category):
                self.category_to_books[category].append(idx)
            if pd.notna(decade):
                self.decade_to_books[decade].append(idx)
            if isbn:
                self.isbn_to_index[isbn] = idx

        print(f"Built indices for {len(self.title_to_index)} titles, {len(self.author_to_books)} authors")

    def _prepare_ratings_data(self):
        """Prepare ratings data for collaborative filtering"""
        print("Processing ratings data...")

        # Filter out zero ratings
        self.explicit_ratings = self.ratings_df[self.ratings_df['Book-Rating'] > 0].copy()

        # Calculate average ratings for books
        self.book_ratings = self.explicit_ratings.groupby('ISBN').agg({
            'Book-Rating': ['mean', 'count']
        }).reset_index()
        self.book_ratings.columns = ['ISBN', 'Avg_Rating', 'Num_Ratings']

        # Calculate user rating statistics
        self.user_ratings = self.explicit_ratings.groupby('User-ID').agg({
            'Book-Rating': ['mean', 'count']
        }).reset_index()
        self.user_ratings.columns = ['User-ID', 'Avg_User_Rating', 'Num_User_Ratings']

        # Create sparse user-item dictionary (more memory efficient)
        self.user_item_dict = {}
        print("Building user-item matrix...")
        for _, row in self.explicit_ratings.iterrows():
            user_id = row['User-ID']
            isbn = row['ISBN']
            rating = row['Book-Rating']

            if user_id not in self.user_item_dict:
                self.user_item_dict[user_id] = {}
            self.user_item_dict[user_id][isbn] = rating

        # Find popular books
        self.popular_books = self.book_ratings.sort_values(
            by=['Num_Ratings', 'Avg_Rating'],
            ascending=[False, False]
        ).head(200)['ISBN'].tolist()

        print(f"✓ Processed {len(self.explicit_ratings)} explicit ratings")
        print(f"✓ Found {len(self.book_ratings)} rated books")
        print(f"✓ Found {len(self.user_ratings)} active users")

    def _prepare_user_profiles(self):
        """Prepare user profiles for personalization"""
        print("Preparing user profiles...")

        # Clean user data
        self.users_df['Age'] = pd.to_numeric(self.users_df['Age'], errors='coerce')
        self.users_df['AgeGroup'] = pd.cut(
            self.users_df['Age'],
            bins=[0, 12, 18, 30, 50, 100],
            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
        )
        self.users_df['Location'] = self.users_df['Location'].fillna('').astype(str).str.lower()

        # Create user profiles
        self.user_profiles = {}
        for _, row in self.users_df.iterrows():
            user_id = row['User-ID']
            self.user_profiles[user_id] = {
                'AgeGroup': row['AgeGroup'],
                'Location': row['Location']
            }

            # Add rating history if available
            if user_id in self.user_ratings['User-ID'].values:
                user_stats = self.user_ratings[self.user_ratings['User-ID'] == user_id].iloc[0]
                self.user_profiles[user_id]['Avg_Rating'] = float(user_stats['Avg_User_Rating'])
                self.user_profiles[user_id]['Num_Ratings'] = int(user_stats['Num_User_Ratings'])

        print(f"✓ Prepared {len(self.user_profiles)} user profiles")

    # ============== RECOMMENDATION METHODS ==============

    def search_books(self, query, limit=20):
        """Search books by title, author, or category"""
        query = query.lower().strip()
        results = []

        # Search in titles
        for title, idx in self.title_to_index.items():
            if query in title:
                book = self.df.iloc[idx]
                results.append(self._book_to_dict(book))
                if len(results) >= limit:
                    break

        # Search in authors
        if len(results) < limit:
            for author, indices in self.author_to_books.items():
                if query in author:
                    for idx in indices[:5]:  # Limit per author
                        book = self.df.iloc[idx]
                        results.append(self._book_to_dict(book))
                        if len(results) >= limit:
                            break

        # Search in categories
        if len(results) < limit:
            for category, indices in self.category_to_books.items():
                if query in category.lower():
                    for idx in indices[:3]:  # Limit per category
                        book = self.df.iloc[idx]
                        results.append(self._book_to_dict(book))
                        if len(results) >= limit:
                            break

        return results[:limit]

    def get_recommendations_by_category(self, category, n=10):
        """Get recommendations by category"""
        category_lower = category.lower()
        matching_categories = [cat for cat in self.category_to_books.keys()
                               if category_lower in cat.lower()]

        if not matching_categories:
            return []

        all_books = []
        for cat in matching_categories:
            indices = self.category_to_books[cat]
            sample_size = min(10, len(indices))
            sampled_indices = random.sample(indices, sample_size) if indices else []

            for idx in sampled_indices:
                book = self.df.iloc[idx]
                all_books.append(self._book_to_dict(book))

        # Remove duplicates
        unique_books = {}
        for book in all_books:
            if book['title'] not in unique_books:
                unique_books[book['title']] = book

        return list(unique_books.values())[:n]

    def get_popular_recommendations(self, n=10):
        """Get popular books based on ratings"""
        recommendations = []
        for isbn in self.popular_books[:n * 2]:
            if isbn in self.isbn_to_index:
                idx = self.isbn_to_index[isbn]
                book = self.df.iloc[idx]
                book_dict = self._book_to_dict(book)

                # Add rating info
                rating_info = self.book_ratings[self.book_ratings['ISBN'] == isbn]
                if not rating_info.empty:
                    book_dict['avg_rating'] = float(rating_info['Avg_Rating'].values[0])
                    book_dict['num_ratings'] = int(rating_info['Num_Ratings'].values[0])

                recommendations.append(book_dict)

        return recommendations[:n]

    def get_collaborative_recommendations(self, user_id, n=10):
        """Simple collaborative filtering"""
        if user_id not in self.user_item_dict:
            return []

        user_ratings = self.user_item_dict[user_id]

        # Find similar users (simplified)
        similar_users = []
        for other_user, other_ratings in self.user_item_dict.items():
            if other_user != user_id:
                common_books = set(user_ratings.keys()) & set(other_ratings.keys())
                if common_books:
                    similarity = len(common_books)
                    similar_users.append((other_user, similarity))

        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)

        # Get recommendations
        recommendations = []
        for other_user, _ in similar_users[:5]:  # Top 5 similar users
            other_ratings = self.user_item_dict[other_user]
            for isbn, rating in other_ratings.items():
                if rating >= 7 and isbn not in user_ratings and isbn in self.isbn_to_index:
                    idx = self.isbn_to_index[isbn]
                    book = self.df.iloc[idx]
                    recommendations.append(self._book_to_dict(book))
                    if len(recommendations) >= n * 2:
                        break

        return recommendations[:n]

    def get_personalized_recommendations(self, user_id, n=10):
        """Get personalized recommendations"""
        if user_id not in self.user_profiles:
            return self.get_popular_recommendations(n)

        profile = self.user_profiles[user_id]
        recommendations = []

        # Collaborative filtering if user has ratings
        if 'Num_Ratings' in profile and profile['Num_Ratings'] > 0:
            cf_recs = self.get_collaborative_recommendations(user_id, n // 2)
            recommendations.extend(cf_recs)

        # Age-based recommendations
        age_group = profile['AgeGroup']
        age_categories = {
            'Child': ['Fantasy', 'Children'],
            'Teen': ['Young Adult', 'Romance', 'Fantasy'],
            'Young Adult': ['Science Fiction', 'Romance', 'Mystery/Crime'],
            'Adult': ['Mystery/Crime', 'Historical', 'Biography'],
            'Senior': ['Historical', 'Biography', 'Classic Literature']
        }

        if age_group in age_categories:
            for category in age_categories[age_group]:
                cat_recs = self.get_recommendations_by_category(category, 2)
                recommendations.extend(cat_recs)

        # Add popular books as fallback
        if len(recommendations) < n:
            popular_recs = self.get_popular_recommendations(n - len(recommendations))
            recommendations.extend(popular_recs)

        # Remove duplicates
        unique_recs = {}
        for rec in recommendations:
            if rec['title'] not in unique_recs:
                unique_recs[rec['title']] = rec

        return list(unique_recs.values())[:n]

    def get_similar_books(self, book_title, n=10):
        """Find similar books based on author and category"""
        book_title_lower = book_title.lower()

        # Find the book
        book_idx = None
        for title, idx in self.title_to_index.items():
            if book_title_lower in title:
                book_idx = idx
                break

        if book_idx is None:
            return []

        book = self.df.iloc[book_idx]
        author = book['Book-Author'].lower()
        category = book['Category']

        # Find books by same author
        similar_books = []
        if author in self.author_to_books:
            for idx in self.author_to_books[author][:5]:
                if idx != book_idx:
                    similar_book = self.df.iloc[idx]
                    similar_books.append(self._book_to_dict(similar_book))

        # Find books in same category
        if category in self.category_to_books:
            for idx in self.category_to_books[category][:5]:
                if idx != book_idx and self.df.iloc[idx]['Book-Title'] not in [b['title'] for b in similar_books]:
                    similar_book = self.df.iloc[idx]
                    similar_books.append(self._book_to_dict(similar_book))

        return similar_books[:n]

    def _book_to_dict(self, book):
        """Convert book row to dictionary"""
        isbn = book['ISBN']
        rating_info = self.book_ratings[self.book_ratings['ISBN'] == isbn]

        book_dict = {
            'isbn': str(isbn),
            'title': str(book['Book-Title']),
            'author': str(book['Book-Author']),
            'publisher': str(book['Publisher']),
            'category': str(book['Category'])
        }

        # Add year if available
        if not pd.isna(book['Year-Of-Publication']):
            book_dict['year'] = int(book['Year-Of-Publication'])

        # Add rating info if available
        if not rating_info.empty:
            book_dict['avg_rating'] = float(rating_info['Avg_Rating'].values[0])
            book_dict['num_ratings'] = int(rating_info['Num_Ratings'].values[0])
        else:
            book_dict['avg_rating'] = None
            book_dict['num_ratings'] = 0

        return book_dict

    def get_stats(self):
        """Get system statistics"""
        stats = {
            'total_books': len(self.df),
            'total_authors': len(self.author_to_books),
            'total_users': len(self.users_df),
            'total_ratings': len(self.ratings_df),
            'explicit_ratings': len(self.explicit_ratings),
            'categories': len(self.category_to_books),
            'cache_status': 'Enabled' if self.use_cache else 'Disabled',
            'cache_prefix': self.cache_prefix
        }

        # Publication years
        valid_years = self.df['Year-Of-Publication'].dropna()
        if len(valid_years) > 0:
            stats['min_year'] = int(valid_years.min())
            stats['max_year'] = int(valid_years.max())

        # Top categories
        top_categories = self.df['Category'].value_counts().head(10).to_dict()
        stats['top_categories'] = {str(k): int(v) for k, v in top_categories.items()}

        # Top authors
        top_authors = self.df['Book-Author'].value_counts().head(10).to_dict()
        stats['top_authors'] = {str(k): int(v) for k, v in top_authors.items()}

        # Rating distribution
        if len(self.explicit_ratings) > 0:
            rating_dist = self.explicit_ratings['Book-Rating'].value_counts().sort_index().to_dict()
            stats['rating_distribution'] = {int(k): int(v) for k, v in rating_dist.items()}
            stats['avg_rating'] = float(self.explicit_ratings['Book-Rating'].mean())

        return stats

    def display_stats(self):
        """Display statistics in console"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("SYSTEM STATISTICS")
        print("=" * 60)
        print(f"Total Books: {stats['total_books']:,}")
        print(f"Total Authors: {stats['total_authors']:,}")
        print(f"Total Users: {stats['total_users']:,}")
        print(f"Total Ratings: {stats['total_ratings']:,}")
        print(f"Explicit Ratings: {stats['explicit_ratings']:,}")

        if 'min_year' in stats:
            print(f"Publication Years: {stats['min_year']} - {stats['max_year']}")

        if 'avg_rating' in stats:
            print(f"Average Rating: {stats['avg_rating']:.2f}")

        print(f"\nCache: {stats['cache_status']}")
        if self.use_cache:
            print(f"Cache Prefix: {stats['cache_prefix']}")

        print("\nTop Categories:")
        for category, count in stats['top_categories'].items():
            print(f"  {category}: {count:,}")

        print("\nSystem ready!")


# Initialize recommender
recommender = BookRecommender(use_cache=True)


# ============== FLASK ROUTES ==============

@app.route('/')
def index():
    """Home page"""
    stats = recommender.get_stats()
    popular_books = recommender.get_popular_recommendations(n=6)
    return render_template('index.html',
                           categories=recommender.categories[:50],
                           authors=recommender.authors[:50],
                           stats=stats,
                           popular_books=popular_books)


@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search books"""
    if request.method == 'POST':
        query = request.form.get('query', '')
        results = recommender.search_books(query, limit=50)
        return render_template('results.html',
                               results=results,
                               query=query,
                               result_type=f"Search results for '{query}'")

    query = request.args.get('q', '')
    if query:
        results = recommender.search_books(query, limit=50)
        return render_template('results.html',
                               results=results,
                               query=query,
                               result_type=f"Search results for '{query}'")

    return render_template('results.html', results=[], query='')


@app.route('/recommend/category/<category>')
def recommend_by_category(category):
    """Recommend by category"""
    results = recommender.get_recommendations_by_category(category, n=50)
    return render_template('results.html',
                           results=results,
                           query=category,
                           result_type=f"Recommendations in '{category}'")


@app.route('/recommend/popular')
def recommend_popular():
    """Popular recommendations"""
    results = recommender.get_popular_recommendations(n=50)
    return render_template('results.html',
                           results=results,
                           query='',
                           result_type="Popular Books")


@app.route('/recommend/user/<int:user_id>')
def recommend_for_user(user_id):
    """Personalized recommendations for user"""
    results = recommender.get_personalized_recommendations(user_id, n=50)
    return render_template('results.html',
                           results=results,
                           query=str(user_id),
                           result_type=f"Personalized recommendations for User {user_id}")


@app.route('/recommend/similar')
def recommend_similar():
    """Find similar books"""
    book_title = request.args.get('book', '')
    if book_title:
        results = recommender.get_similar_books(book_title, n=50)
        return render_template('results.html',
                               results=results,
                               query=book_title,
                               result_type=f"Books similar to '{book_title}'")
    return render_template('results.html', results=[], query='')


@app.route('/api/search')
def api_search():
    """API endpoint for search"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    results = recommender.search_books(query, limit=limit)
    return jsonify({'results': results, 'count': len(results)})


@app.route('/api/recommend/category')
def api_recommend_category():
    """API endpoint for category recommendations"""
    category = request.args.get('category', '')
    limit = int(request.args.get('limit', 10))

    if not category:
        return jsonify({'error': 'No category provided'}), 400

    results = recommender.get_recommendations_by_category(category, n=limit)
    return jsonify({'results': results, 'count': len(results)})


@app.route('/api/recommend/popular')
def api_recommend_popular():
    """API endpoint for popular recommendations"""
    limit = int(request.args.get('limit', 10))
    results = recommender.get_popular_recommendations(n=limit)
    return jsonify({'results': results, 'count': len(results)})


@app.route('/api/stats')
def api_stats():
    """API endpoint for system statistics"""
    stats = recommender.get_stats()
    return jsonify(stats)


@app.route('/stats')
def stats_page():
    """Statistics page"""
    stats = recommender.get_stats()
    return render_template('stats.html', stats=stats)


@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear cache files"""
    try:
        cache_files = [f for f in os.listdir(app.config['CACHE_DIR'])
                       if f.endswith('.pkl')]

        cleared_count = 0
        for file in cache_files:
            try:
                os.remove(os.path.join(app.config['CACHE_DIR'], file))
                cleared_count += 1
            except:
                pass

        return jsonify({'success': True, 'message': f'Cleared {cleared_count} cache files'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting Flask Web Interface...")
    print("=" * 60)
    print("Open your browser and navigate to: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
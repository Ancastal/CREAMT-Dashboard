import psutil
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class KNNProcessor(RetrievalSystem):
    def load_model(self, model_path=None):
        # KNN doesn't require a pre-trained model, so we can skip loading a model
        self.vectorizer = TfidfVectorizer()
        self.knn = None
        print("KNN Processor initialized successfully.")

    def process_data(self):
        # Check if data is loaded
        if self.data is None or self.titles_data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        # Fit the vectorizer on the combined data
        combined_data = self.data + self.titles_data
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_data)

        # Split the tfidf matrix back into data and titles
        self.data_tfidf = self.tfidf_matrix[:len(self.data)]
        self.titles_tfidf = self.tfidf_matrix[len(self.data):]

        # Initialize and fit the KNN model on the general domain data
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.knn.fit(self.data_tfidf)
        print("Data processed and KNN model fitted successfully.")

    def compare_data(self, batch_size=1000):
        # Check if the KNN model is fitted
        if self.knn is None or self.titles_tfidf is None:
            raise ValueError("Model is not fitted or data is not processed. Call process_data() first.")

        titles_dict = {}
        num_titles = self.titles_tfidf.shape[0]

        for start_idx in range(0, num_titles, batch_size):
            end_idx = min(start_idx + batch_size, num_titles)
            batch_titles_tfidf = self.titles_tfidf[start_idx:end_idx]

            # Monitor memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"Memory usage: {memory_info.rss / (1024 * 1024)} MB")

            # Find the k-nearest neighbors for the batch of titles
            distances, indices = self.knn.kneighbors(batch_titles_tfidf, n_neighbors=5)
            
            for i, idx in enumerate(range(start_idx, end_idx)):
                title = self.titles_data[idx]
                titles_dict[title] = [(self.data[j], 1 - distances[i][j]) for j in indices[i]]
        
        print("Data compared using KNN successfully.")
        return titles_dict

    def get_topk_absolute(self, titles_dict, k, similarity_threshold=0.8):
        topk = {}
        for title, similar_titles in titles_dict.items():
            topk[title] = similar_titles[:k]
        for title, similar_titles in topk.items():
            topk[title] = [
                (similar_title, similarity)
                for similar_title, similarity in similar_titles
                if similarity > similarity_threshold
            ]
        topk = {title: similar_titles for title, similar_titles in topk.items() if similar_titles}
        if not topk:
            print("No similar titles found.")
        return topk

# CLI Script remains unchanged

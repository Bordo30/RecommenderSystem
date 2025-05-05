import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load MovieLens dataset
dataset = "ratings.csv"
df = pd.read_csv(dataset)

# Keep relevant columns
df = df[['userId', 'movieId', 'rating']]

# Pivot the table to create a user-item matrix
user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# Fill NaN values with 0 (for SVD computation)
user_item_matrix = user_item_matrix.fillna(0)

# Convert to NumPy array
R = user_item_matrix.values

# Number of users and items
num_users, num_items = R.shape

# Normalize by subtracting each user's mean rating (optional but often improves performance)
user_ratings_mean = np.mean(R, axis=1)
R_normalized = R - user_ratings_mean.reshape(-1, 1)

def svd_predict(R, k=50):
    """
    Perform SVD and return predicted ratings
    Args:
        R (numpy.ndarray): User-item matrix
        k (int): Number of latent factors to keep
    Returns:
        numpy.ndarray: Predicted ratings matrix
    """
    # Perform SVD
    U, sigma, Vt = svds(R_normalized, k=k)
    
    # Reconstruct the diagonal matrix of singular values
    sigma = np.diag(sigma)
    
    # Reconstruct the predicted matrix
    predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    
    return predicted_ratings

# Predict ratings using SVD with different latent factor counts
k_values = [10, 50, 100]  # Test different numbers of latent factors
results = []

for k in k_values:
    # Get predictions
    pred_ratings = svd_predict(R, k=k)
    
    # Extract known ratings for evaluation
    mask = R > 0
    true_ratings = R[mask]
    pred_ratings_values = pred_ratings[mask]
    
    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings_values))
    mae = mean_absolute_error(true_ratings, pred_ratings_values)
    
    results.append((k, rmse, mae))

# Print results
print("Model-Based Collaborative Filtering (SVD) Results:")
print("-----------------------------------------------")
for k, rmse, mae in results:
    print(f"Latent Factors (k={k}) - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Optional: Compare with baseline (average rating)
baseline_pred = np.full(R.shape, np.mean(R[R > 0]))
baseline_rmse = np.sqrt(mean_squared_error(R[R > 0], baseline_pred[R > 0]))
baseline_mae = mean_absolute_error(R[R > 0], baseline_pred[R > 0])
print(f"\nBaseline (Average Rating) - RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")
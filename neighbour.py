import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load MovieLens dataset
dataset = "ratings.csv"
df = pd.read_csv(dataset)

# Keep relevant columns
df = df[['userId', 'movieId', 'rating']]

# Pivot the table to create a user-item matrix
user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# Fill NaN values with 0
user_item_matrix = user_item_matrix.fillna(0).values # Convert to NumPy array

# Number of users and items
num_users, num_items = user_item_matrix.shape

print(user_item_matrix.shape)


def euclidean_similarity(matrix):
    distances = cdist(matrix, matrix, metric='euclidean')
    return 1 / (1 + distances)  # Convert distance to similarity


def cosine_similarity_custom(matrix):
    return cosine_similarity(matrix)


def modified_pearson_correlation(matrix):
    mask = matrix > 0  # Mask for rated items
    mean_ratings = np.where(mask, matrix, 0).sum(axis=1) / mask.sum(axis=1)  # Mean ratings
    mean_ratings = mean_ratings.reshape(-1, 1)

    # Compute centered ratings
    centered_matrix = np.where(mask, matrix - mean_ratings, 0)

    # Compute similarity using dot product
    numerator = centered_matrix @ centered_matrix.T
    denominator = np.sqrt((centered_matrix ** 2).sum(axis=1, keepdims=True)) @ np.sqrt(
        (centered_matrix ** 2).sum(axis=1, keepdims=True).T)

    similarity = np.divide(numerator, denominator, where=denominator > 0)
    return similarity


def predict_ratings(matrix, similarity, top_k=5):
    np.fill_diagonal(similarity, 0)  # Set diagonal to 0 (self-similarity)

    # Sort similarities and select top-k neighbors
    top_k_indices = np.argsort(similarity, axis=1)[:, -top_k:]

    # Compute weighted sum of ratings
    weighted_ratings = np.zeros_like(matrix)
    for i in range(num_users):
        neighbors = top_k_indices[i]
        sim_scores = similarity[i, neighbors]
        ratings = matrix[neighbors, :]

        # Weighted sum
        weighted_ratings[i, :] = np.dot(sim_scores, ratings) / (np.sum(np.abs(sim_scores)) + 1e-8)

    return weighted_ratings


# Compute similarity matrices
sim_euclidean = euclidean_similarity(user_item_matrix)
sim_cosine = cosine_similarity_custom(user_item_matrix)
sim_pcc = modified_pearson_correlation(user_item_matrix)

# Predict ratings
pred_euclidean = predict_ratings(user_item_matrix, sim_euclidean)
pred_cosine = predict_ratings(user_item_matrix, sim_cosine)
pred_pcc = predict_ratings(user_item_matrix, sim_pcc)

# Extract known ratings for evaluation
mask = user_item_matrix > 0
true_ratings = user_item_matrix[mask]
pred_euclidean_values = pred_euclidean[mask]
pred_cosine_values = pred_cosine[mask]
pred_pcc_values = pred_pcc[mask]


def evaluate(true, predicted):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    return rmse, mae


rmse_euclidean, mae_euclidean = evaluate(true_ratings, pred_euclidean_values)
rmse_cosine, mae_cosine = evaluate(true_ratings, pred_cosine_values)
rmse_pcc, mae_pcc = evaluate(true_ratings, pred_pcc_values)

# Print results
print(f"Euclidean Distance - RMSE: {rmse_euclidean:.4f}, MAE: {mae_euclidean:.4f}")
print(f"Cosine Similarity - RMSE: {rmse_cosine:.4f}, MAE: {mae_cosine:.4f}")
print(f"Pearson Correlation - RMSE: {rmse_pcc:.4f}, MAE: {mae_pcc:.4f}")
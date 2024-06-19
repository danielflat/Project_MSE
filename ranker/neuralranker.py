import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

# Example data
data = {
    'query': ['Tübingen', 'Tübingen', 'Tübingen'],
    'document': ['doc1', 'doc2', 'doc3'],
    'features': [[0.1, 0.2, 0.3], [0.4, 0.1, 0.0], [0.3, 0.3, 0.1]],
    'relevance': [3, 1, 2]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['features'].tolist(), df['relevance'], test_size=0.2, random_state=42
)

# Convert to LightGBM dataset format
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# Set parameters for LambdaMART
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the model
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)

class RankNet(nn.Module):
    def __init__(self, input_size):
        super(RankNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Convert features and targets to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = RankNet(input_size=len(X_train[0]))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


def rank_documents(query, documents, feature_extractor, lambdamart_model, neural_net_model):
    # Extract features for each document
    document_features = [feature_extractor.transform([doc]).toarray()[0] for doc in documents]

    # LambdaMART predictions
    lambdamart_scores = lambdamart_model.predict(document_features)

    # Convert document features to tensor for neural network prediction
    document_features_tensor = torch.tensor(document_features, dtype=torch.float32)

    # Neural network predictions
    neural_net_model.eval()  # Set to evaluation mode
    with torch.no_grad():
        neural_net_scores = neural_net_model(document_features_tensor).numpy().flatten()

    # Combine the scores
    combined_scores = lambdamart_scores + neural_net_scores

    # Rank documents by combined scores
    ranked_docs = sorted(zip(documents, combined_scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked_docs]





query = "Tübingen"
documents = ["doc1 content", "doc2 content", "doc3 content"]

# Dummy feature extractor for demonstration
vectorizer = TfidfVectorizer()
vectorizer.fit(documents)  # Fit the vectorizer on the document corpus

ranked_documents = rank_documents(query, documents, vectorizer, gbm, model)
print(ranked_documents)


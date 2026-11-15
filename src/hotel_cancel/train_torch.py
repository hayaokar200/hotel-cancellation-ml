"""Train a PyTorch model (logistic regression or small NN) on the same split and features as train.py."""
import torch
from torch import nn
from datetime import datetime

from hotel_cancel import config, data, evaluate, features



def to_tensors(X, y):
    """Dense float32 X (n, 224) and y (n, 1)."""
    X= X.toarray()
    y = y.to_numpy()
    X = torch.tensor(X,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    y = y.reshape(-1,1)
    return X,y



def build_torch_model(n_features: int, hidden: int | None) -> nn.Module:
    """hidden=None → logistic regression; hidden=64 → small NN with ReLU."""

    if hidden is None :
        model =  nn.Linear(n_features, 1)
    else:   
        model = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    return model




def train_loop(model, X_train_t, y_train_t, epochs: int, lr: float) -> nn.Module:
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()                                # training mode
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

       
    return model


def main():
    torch.manual_seed(config.RANDOM_STATE)   # same weights every run → reproducible

       # Step 1: Load the raw data
    df = data.load_raw(config.DATA_PATH)

    # Step 2: Clean the data
    df = data.drop_unused_columns(df)
    df = data.add_dates(df)

    # Step 3: Split the data into train, validation, and test sets
    train_df, val_df, test_df = data.temporal_split(df)

    # Step 4: Split the data into features and target
    X_train, y_train = data.split_xy(train_df)
    X_val, y_val = data.split_xy(val_df)
    X_test, y_test = data.split_xy(test_df)

    # Step 5: Build the preprocessor
    preprocessor = features.build_preprocessor(X_train)

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    X_train_t,y_train_t =  to_tensors(X_train,y_train)
    X_val_t,y_val_t =  to_tensors(X_val,y_val)
    X_test_t,y_test_t =  to_tensors(X_test,y_test)

    hidden = config.TORCH_PARAMS['hidden'] #Set to None for torch_logreg

    model = build_torch_model(X_train.shape[1],hidden)
    model = train_loop(model,X_train_t,y_train_t,config.TORCH_PARAMS['epochs'],config.TORCH_PARAMS['lr'])

    model.eval()

    with torch.no_grad():                    # no gradients needed: faster, uses less memory
        val_logits = model(X_val_t)
        val_prob = torch.sigmoid(val_logits).numpy().ravel()   # scores → probabilities → flat NumPy array

    threshold = evaluate.get_best_threshold(y_val, val_prob)            # chosen on validation, like for LightGBM
    val_metrics = evaluate.get_eval_metrics(y_val, val_prob, threshold)


    # Step 10: Save the model
    models_dir = config.MODEL_PATH.parent
    models_dir.mkdir(parents=True, exist_ok=True)




    with open(config.EXPERIMENTS_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## Experiment - {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"- **Model:** {'torch_logreg' if hidden is None else 'torch_nn'}\n")
        f.write(f"- **Parameters:** `{config.TORCH_PARAMS}`\n")
        for metric, value in val_metrics.items():
            f.write(f"- **{metric}:** {value:.4f}\n")

        f.write("\n---\n")

if __name__ == "__main__":
    main()
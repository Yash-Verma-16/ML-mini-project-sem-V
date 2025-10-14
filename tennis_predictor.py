# tennis_predictor.py
#
# This script builds an advanced machine learning model to predict professional
# tennis match outcomes and simulates a betting strategy based on the model's
# predictions. This implementation is based on the concepts outlined in the
# project report "Machine Learning for Professional Tennis Match Prediction and Betting".
#
# Improvements over the original report's implementation:
# 1.  Model: Uses a GradientBoostingClassifier, which is often more powerful
#     than a Random Forest and directly addresses the "Future Work" suggestion
#     of trying more advanced models.
# 2.  Hyperparameter Tuning: Implements GridSearchCV to systematically find the
#     best parameters for the model, ensuring optimal performance.
# 3.  Betting Strategy: Implements a more sophisticated betting strategy that
#     scales the bet size based on the model's confidence, another point
#     from the "Future Work" section.
# 4.  Code Structure: The code is organized into functions for clarity,
#     maintainability, and easy experimentation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import calibration_curve

def load_and_prepare_data(num_samples=10000):
    """
    Creates and prepares a synthetic dataset mimicking real tennis match data.
    
    In your actual project, you would replace this function with your own
    data loading and merging logic for the datasets from Jeff Sackmann and
    tennis-data.co.uk.

    Args:
        num_samples (int): The number of synthetic matches to generate.

    Returns:
        pandas.DataFrame: A DataFrame with features and the match outcome.
    """
    print("Loading synthetic data... (Replace with your actual data loading)")
    # Generate synthetic data
    np.random.seed(42)
    data = {
        'p1_rank': np.random.randint(1, 200, num_samples),
        'p2_rank': np.random.randint(1, 200, num_samples),
        'p1_age': np.random.uniform(18, 38, num_samples),
        'p2_age': np.random.uniform(18, 38, num_samples),
        'p1_wins_last_20': np.random.randint(5, 21, num_samples),
        'p2_wins_last_20': np.random.randint(5, 21, num_samples),
        'p1_odds': np.random.uniform(1.1, 4.0, num_samples),
    }
    df = pd.DataFrame(data)

    # Player 2 odds are often inversely related to player 1's odds
    df['p2_odds'] = 1 / (df['p1_odds'] - 1)
    df['p2_odds'] = df['p2_odds'].clip(1.1, 4.0)
    
    # Create a synthetic target variable based on features
    # A lower rank and higher recent wins are good indicators of winning
    p1_win_prob = 0.5 + (df['p2_rank'] - df['p1_rank']) / 400 + \
                  (df['p1_wins_last_20'] - df['p2_wins_last_20']) / 40
    p1_win_prob = np.clip(p1_win_prob, 0.1, 0.9)
    
    df['p1_wins'] = (np.random.rand(num_samples) < p1_win_prob).astype(int)

    return df

def feature_engineering(df):
    """
    Creates symmetrical difference-based features from the raw data.
    This ensures the model is not biased by the arbitrary assignment of
    'Player 1' and 'Player 2'.

    Args:
        df (pandas.DataFrame): The input DataFrame with raw player stats.

    Returns:
        pandas.DataFrame: A DataFrame with engineered features.
    """
    print("Performing feature engineering...")
    features = pd.DataFrame()
    features['diff_rank'] = df['p1_rank'] - df['p2_rank']
    features['diff_age'] = df['p1_age'] - df['p2_age']
    features['diff_wins_last_20'] = df['p1_wins_last_20'] - df['p2_wins_last_20']
    
    # Add betting odds as features, as they are powerful predictors
    features['p1_odds'] = df['p1_odds']
    features['p2_odds'] = df['p2_odds']

    return features

def plot_calibration_curve(y_true, y_prob, model_name):
    """
    Plots a calibration curve to evaluate how well-calibrated the model's
    predicted probabilities are.
    """
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated')
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'{model_name}')
    plt.title('Calibration Curve', fontsize=16)
    plt.xlabel('Mean Predicted Value', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importances from the trained model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances', fontsize=16)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance', fontsize=12)
    plt.grid(True)
    plt.show()

def simulate_betting_strategy(model, X_test, df_test):
    """
    Simulates an improved betting strategy on the test set.

    This strategy scales the bet size based on the perceived "edge" or
    "value" in the bet, where value exists if the model's predicted probability
    is higher than the probability implied by the betting odds.

    Args:
        model: The trained classifier.
        X_test (pandas.DataFrame): The test set features.
        df_test (pandas.DataFrame): The original test DataFrame containing odds and results.

    Returns:
        None. Prints the simulation results.
    """
    print("\n--- Simulating Advanced Betting Strategy ---")
    predictions = model.predict_proba(X_test)[:, 1] # Probability of Player 1 winning
    
    # Implied probabilities from odds
    p1_implied_prob = 1 / df_test['p1_odds']
    p2_implied_prob = 1 / df_test['p2_odds']
    
    bankroll = 1000
    winnings = []
    
    for i, p1_win_prob in enumerate(predictions):
        p2_win_prob = 1 - p1_win_prob
        match_result = df_test['p1_wins'].iloc[i]
        
        p1_edge = p1_win_prob - p1_implied_prob.iloc[i]
        p2_edge = p2_win_prob - p2_implied_prob.iloc[i]

        bet_amount = 0
        bet_on = None
        profit = 0

        # Decide who to bet on based on the largest perceived edge
        if p1_edge > 0.05 and p1_edge > p2_edge: # Bet on P1 if edge is > 5%
            bet_on = 1
            bet_amount = bankroll * 0.01 * (p1_edge * 10) # Scale bet by edge
            odds = df_test['p1_odds'].iloc[i]
            if match_result == 1:
                profit = bet_amount * (odds - 1)
            else:
                profit = -bet_amount
        elif p2_edge > 0.05: # Bet on P2 if edge is > 5%
            bet_on = 2
            bet_amount = bankroll * 0.01 * (p2_edge * 10) # Scale bet by edge
            odds = df_test['p2_odds'].iloc[i]
            if match_result == 0:
                profit = bet_amount * (odds - 1)
            else:
                profit = -bet_amount
        
        winnings.append(profit)

    # --- Print Results ---
    total_profit = np.sum(winnings)
    num_bets = len([w for w in winnings if w != 0])
    
    print(f"Total Matches in Test Set: {len(X_test)}")
    print(f"Number of Bets Placed: {num_bets} ({num_bets/len(X_test):.2%})")
    print(f"Total Profit: ${total_profit:.2f}")
    if num_bets > 0:
      avg_return_per_bet = (total_profit / num_bets)
      print(f"Average Return per Bet: {avg_return_per_bet:.2%}")
      sharpe_ratio = np.mean(winnings) / np.std(winnings) if np.std(winnings) > 0 else 0
      print(f"Strategy Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Plot cumulative winnings
    cumulative_winnings = np.cumsum(winnings)
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_winnings)
    plt.title('Cumulative Winnings Over Time', fontsize=16)
    plt.xlabel('Match Number in Test Set', fontsize=12)
    plt.ylabel('Cumulative Profit ($)', fontsize=12)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # --- 1. Data Loading and Preparation ---
    df_raw = load_and_prepare_data()
    X = feature_engineering(df_raw)
    y = df_raw['p1_wins']
    
    # --- 2. Train-Test Split ---
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df_raw, test_size=0.1, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # --- 3. Model Training with Hyperparameter Tuning ---
    print("\n--- Training Gradient Boosting Model with GridSearchCV ---")
    
    # Using a smaller grid for faster execution. You can expand this for better results.
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.7, 0.8]
    }
    
    gb_model = GradientBoostingClassifier(random_state=42)
    
    # Using cv=3 for faster cross-validation
    grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Parameters found: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # --- 4. Model Evaluation ---
    print("\n--- Evaluating Model on Test Set ---")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # --- 5. Visualization ---
    plot_calibration_curve(y_test, y_prob, "Gradient Boosting")
    plot_feature_importance(best_model, X.columns)
    
    # --- 6. Betting Simulation ---
    simulate_betting_strategy(best_model, X_test, df_test)

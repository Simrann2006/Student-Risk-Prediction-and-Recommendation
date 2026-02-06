# STUDENT RISK PREDICTION SYSTEM - MODEL TRAINING

# Required imports for model training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, make_scorer, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

FEATURES = ['G1', 'G2', 'studytime', 'failures', 'absences', 
            'schoolsup', 'famsup', 'paid', 'internet', 'higher', 
            'health', 'goout']


def load_dataset(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} students, {df.shape[1]} features")
    return df


def split_data(X, y):
    # Split: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION - HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Define hyperparameters to tune
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    print("\nTesting hyperparameters:")
    print(f"  C values: {param_grid['C']}")
    print(f"  Penalty: {param_grid['penalty']}")
    
    # Grid Search with Cross-Validation
    lr_base = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    grid_search = GridSearchCV(lr_base, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_*100:.2f}%")
    
    # Use best model
    lr_model = grid_search.best_estimator_
    
    # Cross-validation score
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5)
    print(f"\nCross-Validation Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"Mean CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    val_accuracy = accuracy_score(y_val, lr_model.predict(X_val))
    test_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
    
    print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, lr_model.predict(X_test), target_names=['Not At Risk', 'At Risk']))
    
    return lr_model, val_accuracy, test_accuracy


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n" + "=" * 50)
    print("RANDOM FOREST - HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    
    print("\nTesting hyperparameters:")
    print(f"  n_estimators: {param_grid['n_estimators']}")
    print(f"  max_depth: {param_grid['max_depth']}")
    print(f"  min_samples_split: {param_grid['min_samples_split']}")
    print(f"  class_weight: {param_grid['class_weight']}")
    
    # F2-score prioritizes recall (important for catching at-risk students)
    f2_scorer = make_scorer(fbeta_score, beta=2)
    
    # Grid Search with Cross-Validation
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
    grid_search = GridSearchCV(rf_base, param_grid, cv=5, scoring=f2_scorer, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F2-Score: {grid_search.best_score_*100:.2f}%")
    
    # Use best model
    rf_model = grid_search.best_estimator_
    
    # Cross-validation scores
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring=f2_scorer)
    print(f"\nCross-Validation F2-Scores: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"Mean CV F2-Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    val_accuracy = accuracy_score(y_val, rf_model.predict(X_val))
    test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    
    print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_model.predict(X_test), target_names=['Not At Risk', 'At Risk']))
    
    return rf_model, val_accuracy, test_accuracy


def create_knn_model(X_train_scaled, n_neighbors=7):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X_train_scaled)
    return knn


def get_recommendations(student_data, knn, scaler, X_train, y_train):
    student_scaled = scaler.transform(student_data)
    distances, indices = knn.kneighbors(student_scaled)
    similar_outcomes = y_train.iloc[indices[0][1:]]
    success_rate = (similar_outcomes == 0).mean() * 100
    return success_rate, indices[0][1:]


def plot_scaling_comparison(X_train, X_train_scaled, features):
    """Visualize the effect of StandardScaler on feature distributions"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Select 6 features with different scales to show the effect
    display_features = ['G2', 'absences', 'studytime', 'failures', 'health', 'goout']
    display_indices = [features.index(f) for f in display_features if f in features]
    
    for idx, (ax, feat_idx) in enumerate(zip(axes.flatten(), display_indices)):
        feat_name = features[feat_idx]
        
        # Before scaling (original values)
        before = X_train.iloc[:, feat_idx] if hasattr(X_train, 'iloc') else X_train[:, feat_idx]
        # After scaling
        after = X_train_scaled[:, feat_idx]
        
        # Plot both distributions
        ax.hist(before, bins=15, alpha=0.7, label=f'Before (μ={np.mean(before):.1f}, σ={np.std(before):.1f})', color='coral')
        ax.hist(after, bins=15, alpha=0.7, label=f'After (μ={np.mean(after):.1f}, σ={np.std(after):.1f})', color='steelblue')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title(f'{feat_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
    plt.suptitle('Feature Scaling: Before vs After StandardScaler\n(After scaling: Mean=0, Std=1)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('scaling_comparison.png', dpi=150)
    plt.show()
    
    # Print summary table
    print("\n" + "=" * 60)
    print("STANDARD SCALER TRANSFORMATION SUMMARY")
    print("=" * 60)
    print(f"{'Feature':<12} {'Before Mean':>12} {'Before Std':>12} {'After Mean':>12} {'After Std':>10}")
    print("-" * 60)
    for feat_idx, feat_name in enumerate(features):
        before = X_train.iloc[:, feat_idx] if hasattr(X_train, 'iloc') else X_train[:, feat_idx]
        after = X_train_scaled[:, feat_idx]
        print(f"{feat_name:<12} {np.mean(before):>12.2f} {np.std(before):>12.2f} {np.mean(after):>12.2f} {np.std(after):>10.2f}")


def plot_confusion_matrix(y_test, y_pred, model_name, cmap='Blues'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=['Not At Risk', 'At Risk'],
                yticklabels=['Not At Risk', 'At Risk'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    print(f"\n{model_name} Confusion Matrix:")
    print(cm)


def plot_both_confusion_matrices(y_test, lr_pred, rf_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Logistic Regression
    cm_lr = confusion_matrix(y_test, lr_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Not At Risk', 'At Risk'],
                yticklabels=['Not At Risk', 'At Risk'])
    axes[0].set_title('Logistic Regression\nConfusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Random Forest
    cm_rf = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Not At Risk', 'At Risk'],
                yticklabels=['Not At Risk', 'At Risk'])
    axes[1].set_title('Random Forest\nConfusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

def plot_feature_importance(rf_model, features):
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    plt.barh(importance['Feature'], importance['Importance'], color=colors)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    
    print("\nTop 5 Most Important Features:")
    for i, row in importance.tail(5).iloc[::-1].iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

def plot_knn_accuracy(X_train_scaled, y_train, X_val_scaled, y_val):
    print("\n" + "=" * 50)
    print("KNN - FINDING BEST K VALUE")
    print("=" * 50)
    
    k_values = range(1, 12, 2)
    accuracy_list = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_val_pred = knn.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)
        accuracy_list.append(acc)
        print(f"K={k}, Validation Accuracy={acc:.4f}")
    
    best_k = list(k_values)[accuracy_list.index(max(accuracy_list))]
    print(f"\nBest K value: {best_k} (Accuracy: {max(accuracy_list):.4f})")
    
    # Plot K vs Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_list, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K = {best_k}')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Validation Accuracy')
    plt.title('KNN: K Value vs Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('knn_k_accuracy.png')
    plt.show()
    
    return best_k, accuracy_list

def plot_model_comparison(lr_acc, rf_acc, lr_val_acc, rf_val_acc):
    models = ['Logistic Regression', 'Random Forest']
    test_acc = [lr_acc * 100, rf_acc * 100]
    val_acc = [lr_val_acc * 100, rf_val_acc * 100]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, val_acc, width, label='Validation', color='steelblue')
    bars2 = ax.bar(x + width/2, test_acc, width, label='Test', color='coral')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison: Validation vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()


def plot_roc_curves(lr_model, rf_model, X_test_scaled, y_test):
    """Plot ROC curves for both models to compare their classification performance"""
    print("\n" + "=" * 50)
    print("ROC CURVE ANALYSIS")
    print("=" * 50)
    
    # Get probability predictions for positive class (At Risk = 1)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate ROC curve points
    lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_proba)
    rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_proba)
    
    # Calculate AUC (Area Under Curve)
    lr_auc = auc(lr_fpr, lr_tpr)
    rf_auc = auc(rf_fpr, rf_tpr)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves
    plt.plot(lr_fpr, lr_tpr, color='steelblue', linewidth=2, 
             label=f'Logistic Regression (AUC = {lr_auc:.3f})')
    plt.plot(rf_fpr, rf_tpr, color='coral', linewidth=2, 
             label=f'Random Forest (AUC = {rf_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, 
             label='Random Classifier (AUC = 0.500)')
    
    # Highlight the ideal point (0, 1) - perfect classifier
    plt.scatter([0], [1], color='green', s=100, zorder=5, marker='*', 
                label='Perfect Classifier')
    
    # Customize plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)\n(Students incorrectly flagged as At Risk)', fontsize=11)
    plt.ylabel('True Positive Rate (TPR) / Recall\n(At Risk students correctly identified)', fontsize=11)
    plt.title('ROC Curve Comparison: Logistic Regression vs Random Forest\nStudent Risk Prediction', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add AUC interpretation zone shading
    plt.fill_between(lr_fpr, lr_tpr, alpha=0.1, color='steelblue')
    plt.fill_between(rf_fpr, rf_tpr, alpha=0.1, color='coral')
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()
    
    # Print detailed analysis
    print(f"\nLogistic Regression AUC: {lr_auc:.3f}")
    print(f"Random Forest AUC: {rf_auc:.3f}")
    print(f"\nAUC Interpretation:")
    print(f"  0.90-1.00 = Excellent")
    print(f"  0.80-0.90 = Good")
    print(f"  0.70-0.80 = Fair")
    print(f"  0.60-0.70 = Poor")
    print(f"  0.50-0.60 = Fail (no better than random)")
    
    # Determine winner
    if lr_auc > rf_auc:
        print(f"\n► Logistic Regression has better discrimination ability (higher AUC)")
    elif rf_auc > lr_auc:
        print(f"\n► Random Forest has better discrimination ability (higher AUC)")
    else:
        print(f"\n► Both models have equal discrimination ability")
    
    return lr_auc, rf_auc


def train_all_models(df, silent=True):
    X = df[FEATURES].copy()
    y = df['Risk'].copy()
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Train Logistic Regression with GridSearchCV
    lr_param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    lr_grid = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), 
                           lr_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    lr_model = lr_grid.best_estimator_
    lr_val_accuracy = accuracy_score(y_val, lr_model.predict(X_val_scaled))
    lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test_scaled))
    lr_cm = confusion_matrix(y_test, lr_model.predict(X_test_scaled))
    
    # Train Random Forest with GridSearchCV
    rf_param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15], 
                     'min_samples_split': [2, 5], 'class_weight': ['balanced', None]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), 
                           rf_param_grid, cv=5, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    rf_model = rf_grid.best_estimator_
    rf_val_accuracy = accuracy_score(y_val, rf_model.predict(X_val_scaled))
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test_scaled))
    rf_cm = confusion_matrix(y_test, rf_model.predict(X_test_scaled))
    
    # KNN for recommendations
    knn = create_knn_model(X_train_scaled)
    
    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'features': FEATURES,
        'lr_accuracy': lr_accuracy,
        'rf_accuracy': rf_accuracy,
        'lr_val_accuracy': lr_val_accuracy,
        'rf_val_accuracy': rf_val_accuracy,
        'lr_cm': lr_cm,
        'rf_cm': rf_cm,
        'knn': knn,
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'df': df
    }


def main():
    print("=" * 50)
    print("STUDENT RISK PREDICTION SYSTEM")
    print("=" * 50)
    
    # Load cleaned data
    df = load_dataset('dataset/Portuguese_cleaned.csv')
    
    X = df[FEATURES].copy()
    y = df['Risk'].copy()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"\nData Split:")
    print(f"  Training:   {len(X_train)} samples (60%)")
    print(f"  Validation: {len(X_val)} samples (20%)")
    print(f"  Testing:    {len(X_test)} samples (20%)")
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Visualize Scaling Effect
    print("\n" + "=" * 50)
    print("FEATURE SCALING VISUALIZATION")
    print("=" * 50)
    plot_scaling_comparison(X_train, X_train_scaled, FEATURES)
    
    # Train models
    lr_model, lr_val_acc, lr_test_acc = train_logistic_regression(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    )
    
    rf_model, rf_val_acc, rf_test_acc = train_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
    )
    
    # Model Comparison
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"Logistic Regression Test Accuracy: {lr_test_acc*100:.2f}%")
    print(f"Random Forest Test Accuracy: {rf_test_acc*100:.2f}%")
    
    # Get predictions for plotting
    lr_pred = lr_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Plot Confusion Matrices
    plot_both_confusion_matrices(y_test, lr_pred, rf_pred)
    
    # Plot Feature Importance
    plot_feature_importance(rf_model, FEATURES)
    
    # Plot Model Comparison
    plot_model_comparison(lr_test_acc, rf_test_acc, lr_val_acc, rf_val_acc)
    
    # Plot ROC Curves
    plot_roc_curves(lr_model, rf_model, X_test_scaled, y_test)
    
    # KNN - Find Best K and Plot
    best_k, accuracy_list = plot_knn_accuracy(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # KNN Recommendations using Best K
    print("\n" + "=" * 50)
    print(f"KNN RECOMMENDATION SYSTEM (K={best_k})")
    print("=" * 50)
    knn = create_knn_model(X_train_scaled, n_neighbors=best_k)
    
    at_risk_students = df[df['Risk'] == 1].index[:3]
    for idx in at_risk_students:
        student_data = df[FEATURES].iloc[[idx]]
        success_rate, similar_idx = get_recommendations(student_data, knn, scaler, X_train, y_train)
        print(f"Student {idx}: Similar students success rate = {success_rate:.1f}%")
    
    return df, lr_model, rf_model, knn, scaler


if __name__ == "__main__":
    main()
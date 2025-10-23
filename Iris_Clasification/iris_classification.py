
import warnings
warnings.filterwarnings("ignore")  # keep output clean for the demo

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_and_prepare_iris():
    """Load Iris dataset and return a pandas DataFrame with species names."""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)

    # Map target to species names
    target_names = iris.target_names
    df['species'] = pd.Categorical.from_codes(y, target_names)

    return df


def explore_data(df):
    """Print basic exploration info: shape, head, and description."""
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())


def visualize_data(df, show_plots=True):
    """Create pairplot and correlation heatmap. If show_plots is False, save figures instead of showing."""
    sns.set(style="ticks", palette="pastel")

    # Pairplot
    pairplt = sns.pairplot(df, hue='species', corner=False)
    pairplt.fig.suptitle('Iris Pairplot', y=1.02)
    if show_plots:
        plt.show()
    else:
        pairplt.savefig('iris_pairplot.png')
        plt.close(pairplt.fig)

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df.select_dtypes(include=[np.number]).corr()
    hm = sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    if show_plots:
        plt.show()
    else:
        plt.savefig('iris_correlation_heatmap.png')
        plt.close()


def prepare_features(df):
    """Split into X, y and then into train/test sets. Return scaled X_train, X_test and y's."""
    X = df.iloc[:, :-1].values  # all numeric feature columns
    y = df['species'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for algorithms that benefit from scaling (LogisticRegression, KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test


def train_and_evaluate(X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test):
    """Train three classifiers and evaluate them. Return dictionary of accuracies."""
    results = {}

    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")

        # Choose scaled input for models that need scaling
        if name in ['Logistic Regression', 'K-Nearest Neighbors']:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        else:
            # Decision Tree can work with unscaled features
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"{name} Accuracy: {acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} - Confusion Matrix')
        plt.show()

        # Classification report
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))

    return results


def print_comparison(results):
    """Print a comparison table of model accuracies."""
    print("\nModel Accuracies:")
    for name, acc in results.items():
        print(f" - {name}: {acc:.4f}")

    best = max(results, key=results.get)
    print(f"\nBest model: {best} with accuracy {results[best]:.4f}")


def main(show_plots=True):
    df = load_and_prepare_iris()

    explore_data(df)

    visualize_data(df, show_plots=show_plots)

    X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test = prepare_features(df)

    results = train_and_evaluate(X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test)

    print_comparison(results)


if __name__ == '__main__':
    # When running in a headless or CI environment, set show_plots=False to save figures
    main(show_plots=True)

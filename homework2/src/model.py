"""
Model stencil for Homework 2: Ensemble Methods with Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Union

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Set plotting style
sns.set_style("whitegrid")


class GradientBoostingModel:
    """Gradient Boosting model implementation with comprehensive evaluation and analysis tools"""

    def __init__(
        self,
        task: str = "classification",
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 50,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
        use_scaler: bool = False,
    ):
        """
        Initialize Gradient Boosting model with customizable parameters

        Args:
            task: 'classification' or 'regression'
            max_depth: Maximum depth of a tree (controls pruning)
            learning_rate: Step size shrinkage to prevent overfitting
            n_estimators: Number of boosting rounds/trees
            subsample: Subsample ratio of training instances
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random seed for reproducibility
            use_scaler: Whether to apply StandardScaler before training/prediction
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")


        self.model = None
        self.feature_names = None
        self.task = task
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Split data into training and testing sets

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        # TODO: Implement train/test split and track feature names
        self.feature_names = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True):
        """
        Train the Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training targets
            verbose: Whether to print training progress

        Returns:
            self: Trained model instance
        """
        # TODO: Create classifier/regressor based on task and fit it
        if self.scaler is not None: 
            X_t = self.scaler.fit_transform(X_train)
        else: 
            X_t = X_train

        learning_rate = self.params['learning_rate']
        n_estimators = self.params['n_estimators']
        subsample = self.params['subsample']
        max_depth = self.params['max_depth']
        min_samp_split = self.params['min_samples_split']
        min_samp_leaf = self.params['min_samples_leaf']
        max_features = self.params['max_features']
        random_state = self.params['random_state']

        if self.task == "regression":
           self.model = GradientBoostingRegressor(learning_rate=learning_rate, 
                                                  n_estimators=n_estimators, 
                                                  subsample=subsample,
                                                  min_samples_split=min_samp_split, 
                                                  min_samples_leaf=min_samp_leaf, 
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  max_features=max_features, 
                                                  verbose=verbose)
        else: 
            self.model = GradientBoostingClassifier(learning_rate=learning_rate, 
                                                  n_estimators=n_estimators, 
                                                  subsample=subsample,
                                                  min_samples_split=min_samp_split, 
                                                  min_samples_leaf=min_samp_leaf, 
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  max_features=max_features, 
                                                  verbose=verbose, 
                                                  tol=0.01)
            
        self.model.fit(X_t, y_train)

        return self.model

    def predict(
        self, X: pd.DataFrame, return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the trained model

        Args:
            X: Feature data for prediction
            return_proba: If True and model is a classifier, return probability estimates

        Returns:
            Predictions or probability estimates
        """
        # TODO: Apply scaler when enabled, then predict
        if self.scaler is not None: 
            X_test = self.scaler.fit_transform(X)
        else: 
            X_test = X

        if return_proba and self.task == "classification":
            y_proba = self.model.predict_proba(X_test)
            return y_proba
        else: 
            y_pred = self.model.predict(X_test)
            return y_pred

        

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """
        # TODO: Compute metrics (classification vs regression)
        y = self.predict(X_test)
        y_proba = self.predict(X_test, return_proba=True)

        if self.task == "classification":
            metrics = {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
            }
            num_class = len(np.unique(y_test))
            average = 'binary' if num_class == 2 else 'multiclass'

            metrics["accuracy"] = accuracy_score(y_test, y)
            metrics["precision"] = precision_score(y_test, y, average=average)
            metrics['recall'] = recall_score(y_test, y, average=average)
            metrics['f1'] = f1_score(y_test, y, average=average)

            if num_class == 2: 
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])

            else: 
                average = 'weighted'
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba, average=average, multi_class='ovr')
   
        else:
            metrics = {"rmse": None, "mae": None, "r2": None}
            metrics['rmse'] = mean_squared_error(y_test, y)
            metrics['mae'] = mean_absolute_error(y_test, y)
            metrics['r2'] = r2_score(y_test, y)

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation results using sklearn cross_val_score
        """
        # TODO: Use Pipeline when scaling, and choose classifier/regressor based on task
        # TODO: Choose scoring metrics based on classification vs regression
        learning_rate = self.params['learning_rate']
        n_estimators = self.params['n_estimators']
        subsample = self.params['subsample']
        max_depth = self.params['max_depth']
        min_samp_split = self.params['min_samples_split']
        min_samp_leaf = self.params['min_samples_leaf']
        max_features = self.params['max_features']
        random_state = self.params['random_state']

        if self.task == "classification":

            if len(np.unique(y)) == 2:
                scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            else: scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr']

            model_type = GradientBoostingClassifier(learning_rate=learning_rate, 
                                                  n_estimators=n_estimators, 
                                                  subsample=subsample,
                                                  min_samples_split=min_samp_split, 
                                                  min_samples_leaf=min_samp_leaf, 
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  max_features=max_features)

        else:
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
            model_type = GradientBoostingRegressor(learning_rate=learning_rate, 
                                                  n_estimators=n_estimators, 
                                                  subsample=subsample,
                                                  min_samples_split=min_samp_split, 
                                                  min_samples_leaf=min_samp_leaf, 
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  max_features=max_features)

        results = {}
        summary = {}

        if self.scaler:
            model = Pipeline([('scaler', StandardScaler()), ('model_type', model_type)])
        else: 
            model = model_type

        for metric in scoring: 
            # print(metric)
            score = cross_val_score(estimator=model, X=X, y=y, scoring=metric, cv=cv, verbose=2, n_jobs=-1)
            mean_score = np.mean(score)
            std_score = np.std(score)
            results[metric] = score 
            summary[metric] = [mean_score, std_score]

        # return results std and mean dict
        return results, summary

    def get_feature_importance(
        self, plot: bool = False, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importances

        Returns:
            DataFrame with feature importances
        """
        # TODO: Optionally plot a bar chart of top_n feature importances
        feature_importance = self.model.feature_importances_
        feature_importances = pd.DataFrame({"feature": self.feature_names, 
                                            "feature_importance": feature_importance})
        
        importance_sorted = feature_importances.sort_values(by='feature_importance', ascending=False)
        top_n_features = importance_sorted.head(top_n)

        if plot: 
            plt.bar(top_n_features['feature'], top_n_features['feature_importance'])
            plt.xlabel("Features")
            plt.ylabel("Importance")
            plt.title(f"Feature Importance (top {top_n})")
            # plt.savefig('../results/top_n_features')
            # plt.show()

        return importance_sorted

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "roc_auc",
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning

        Args:
            X: Feature data
            y: Target data
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric to evaluate

        Returns:
            Dictionary with best parameters and results
        """
        # TODO: Choose classifier or regressor based on task
        learning_rate = self.params['learning_rate']
        n_estimators = self.params['n_estimators']
        subsample = self.params['subsample']
        max_depth = self.params['max_depth']
        min_samp_split = self.params['min_samples_split']
        min_samp_leaf = self.params['min_samples_leaf']
        max_features = self.params['max_features']
        random_state = self.params['random_state']

        if self.task == 'regression':
            model = GradientBoostingRegressor(learning_rate=learning_rate, 
                                                  n_estimators=n_estimators, 
                                                  subsample=subsample,
                                                  min_samples_split=min_samp_split, 
                                                  min_samples_leaf=min_samp_leaf, 
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  max_features=max_features)
        else: 
            model = GradientBoostingClassifier(learning_rate=learning_rate, 
                                                  n_estimators=n_estimators, 
                                                  subsample=subsample,
                                                  min_samples_split=min_samp_split, 
                                                  min_samples_leaf=min_samp_leaf, 
                                                  max_depth=max_depth,
                                                  random_state=random_state,
                                                  max_features=max_features)

        # TODO: Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model, 
                                   param_grid=param_grid, 
                                   scoring=scoring, 
                                   cv=cv, 
                                   refit=True, 
                                   return_train_score=True, 
                                   verbose=2)

        # TODO: Perform grid search for hyperparameter tuning
        grid_search.fit(X, y)
    
        best_params = grid_search.best_params_
        results = grid_search.cv_results_
        score = grid_search.best_score_

        return {'best_params': best_params, 'best_score':score, 'results':results}

    def plot_tree(
        self, tree_index: int = 0, figsize: Tuple[int, int] = (20, 15)
    ) -> None:
        """
        Plot a specific tree from the ensemble

        Args:
            tree_index: Index of the tree to plot
            figsize: Figure size for the plot
        """
        n_est = self.params['n_estimators']
        if tree_index > n_est:
            raise ValueError(f'Tree index out of range > {n_est}')
        
        else:
            est = self.model.estimators_
            plt.figure(figsize=figsize)
            plot_tree(est[tree_index][0], feature_names=self.feature_names)
            # plt.savefig('../results/plot_tree')
            plt.show()

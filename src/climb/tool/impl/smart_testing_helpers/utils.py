# Imports
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar


def chi_square_test_for_accuracy(df, model, query):
    """
    Performs a chi-square test for accuracy within a specified subgroup.

    Parameters:
        df (pd.DataFrame): The dataset containing features and target.
        model: The trained predictive model.
        query (str): The pandas query string to define the subgroup.

    Returns:
        float: The p-value from the chi-square test. Returns np.nan in case of errors.
    """
    try:
        # Subgroup defined by the query
        subgroup = df.query(query)
        if subgroup.empty:
            print(f"No data found for the query: {query}")
            return np.nan

        # Drop 'y' and generate predictions
        subgroup_X = subgroup.drop("y", axis=1)
        subgroup_predictions = model.predict(subgroup_X)

        # Convert predictions to a pandas Series with the same index as subgroup['y']
        subgroup_predictions = pd.Series(subgroup_predictions, index=subgroup.index)

        # Counting correct and incorrect predictions in the subgroup
        correct_subgroup = np.sum(subgroup["y"] == subgroup_predictions)
        incorrect_subgroup = np.sum(subgroup["y"] != subgroup_predictions)

        # Complementary group (not in the subgroup)
        complementary_group = df.query(f"not ({query})")
        if complementary_group.empty:
            print(f"No data found for the complementary group of the query: {query}")
            return np.nan

        complementary_X = complementary_group.drop("y", axis=1)
        complementary_predictions = model.predict(complementary_X)

        # Convert complementary_predictions to a pandas Series with the same index as complementary_group['y']
        complementary_predictions = pd.Series(complementary_predictions, index=complementary_group.index)

        # Counting correct and incorrect predictions in the complementary group
        correct_complementary = np.sum(complementary_group["y"] == complementary_predictions)
        incorrect_complementary = np.sum(complementary_group["y"] != complementary_predictions)

        # Constructing the contingency table
        table = [[correct_subgroup, correct_complementary], [incorrect_subgroup, incorrect_complementary]]

        # Perform Chi-Square test
        chi2, p, dof, expected = chi2_contingency(table)

        return p

    except Exception as e:
        print(f"Error in chi_square_test_for_accuracy: {e}")
        return np.nan


def bootstrapping_test_for_accuracy_string(df, model, subgroup, num_bootstrap_samples=200):
    """
    Performs a bootstrapping test for accuracy within a specified subgroup using string queries.

    Parameters:
        df (pd.DataFrame): The dataset containing features and target.
        model: The trained predictive model.
        subgroup (pd.DataFrame): The subgroup DataFrame.
        num_bootstrap_samples (int): Number of bootstrap samples.

    Returns:
        float: The p-value from the bootstrapping test.
    """
    try:
        # Define the complementary group based on the subgroup's index
        remainder = df.loc[~df.index.isin(subgroup.index)].copy()

        if remainder.empty:
            print("No complementary group found.")
            return np.nan

        # Generate predictions
        subgroup_X = subgroup.drop("y", axis=1)
        subgroup_predictions = model.predict(subgroup_X)
        subgroup_predictions_series = pd.Series(subgroup_predictions, index=subgroup.index)

        remainder_X = remainder.drop("y", axis=1)
        remainder_predictions = model.predict(remainder_X)
        remainder_predictions_series = pd.Series(remainder_predictions, index=remainder.index)

        # Combine accuracies from both subgroup and remainder
        pooled_accuracies = np.concatenate(
            [
                (subgroup["y"] == subgroup_predictions_series).astype(int),
                (remainder["y"] == remainder_predictions_series).astype(int),
            ]
        )

        # Observed accuracy difference
        observed_diff = np.mean((subgroup["y"] == subgroup_predictions_series).astype(int)) - np.mean(
            (remainder["y"] == remainder_predictions_series).astype(int)
        )

        bootstrap_diffs = []

        # Bootstrapping under the null hypothesis
        for _ in range(num_bootstrap_samples):
            # Resampling with replacement from the pooled accuracies
            resampled_indices = np.random.choice(len(pooled_accuracies), size=len(pooled_accuracies), replace=True)
            resampled_accuracies = pooled_accuracies[resampled_indices]

            # Splitting the resampled accuracies into "subgroup" and "remainder"
            resampled_subgroup_acc = resampled_accuracies[: len(subgroup)]
            resampled_remainder_acc = resampled_accuracies[len(subgroup) : len(subgroup) + len(remainder)]

            # Difference in accuracies for the resampled data
            resampled_diff = np.mean(resampled_subgroup_acc) - np.mean(resampled_remainder_acc)
            bootstrap_diffs.append(resampled_diff)

        # Calculating p-value
        p_value = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff)) / num_bootstrap_samples

        return p_value

    except Exception as e:
        print(f"Error in bootstrapping_test_for_accuracy_string: {e}")
        return np.nan


def bootstrapping_test_for_accuracy(df, model, query, num_bootstrap_samples=200):
    """
    Performs a bootstrapping test for accuracy within a specified subgroup.

    Parameters:
        df (pd.DataFrame): The dataset containing features and target.
        model: The trained predictive model.
        query (str): The pandas query string to define the subgroup.
        num_bootstrap_samples (int): Number of bootstrap samples.

    Returns:
        float: The p-value from the bootstrapping test.
    """
    try:
        # Preprocess the query to ensure it's single-line
        clean_query = query.replace("\n", " ").replace("\r", " ").strip()

        # Subgroup defined by the query
        subgroup = df.query(clean_query)
        remainder = df.query(f"not ({clean_query})")

        if subgroup.empty or remainder.empty:
            print(f"Empty subgroup or complementary group for query: {clean_query}")
            return np.nan

        # Generate predictions
        subgroup_X = subgroup.drop("y", axis=1)
        subgroup_predictions = model.predict(subgroup_X)
        subgroup_predictions_series = pd.Series(subgroup_predictions, index=subgroup.index)

        remainder_X = remainder.drop("y", axis=1)
        remainder_predictions = model.predict(remainder_X)
        remainder_predictions_series = pd.Series(remainder_predictions, index=remainder.index)

        # Combine accuracies from both subgroup and remainder
        pooled_accuracies = np.concatenate(
            [
                (subgroup["y"] == subgroup_predictions_series).astype(int),
                (remainder["y"] == remainder_predictions_series).astype(int),
            ]
        )

        # Observed accuracy difference
        observed_diff = np.mean((subgroup["y"] == subgroup_predictions_series).astype(int)) - np.mean(
            (remainder["y"] == remainder_predictions_series).astype(int)
        )

        bootstrap_diffs = []

        # Bootstrapping under the null hypothesis
        for _ in range(num_bootstrap_samples):
            # Resampling with replacement from the pooled accuracies
            resampled_indices = np.random.choice(len(pooled_accuracies), size=len(pooled_accuracies), replace=True)
            resampled_accuracies = pooled_accuracies[resampled_indices]

            # Splitting the resampled accuracies into "subgroup" and "remainder"
            resampled_subgroup_acc = resampled_accuracies[: len(subgroup)]
            resampled_remainder_acc = resampled_accuracies[len(subgroup) : len(subgroup) + len(remainder)]

            # Difference in accuracies for the resampled data
            resampled_diff = np.mean(resampled_subgroup_acc) - np.mean(resampled_remainder_acc)
            bootstrap_diffs.append(resampled_diff)

        # Calculating p-value
        p_value = np.sum(np.abs(bootstrap_diffs) >= np.abs(observed_diff)) / num_bootstrap_samples

        return p_value

    except Exception as e:
        print(f"Error in bootstrapping_test_for_accuracy: {e}")
        return np.nan


def welchs_t_test_for_accuracy(df, model, query):
    """
    Performs Welch's t-test on the accuracies of a subgroup and its complement.

    Parameters:
        df (pd.DataFrame): The full dataset containing features and the target variable 'y'.
        model: The trained model with a predict method.
        query (str): The pandas query string defining the subgroup.

    Returns:
        float: The p-value from Welch's t-test.
    """
    try:
        # Extract subgroup
        subgroup = df.query(query)
        if "y" not in subgroup.columns:
            raise KeyError("'y' column is missing from the dataframe.")

        # Generate predictions for the subgroup
        subgroup_features = subgroup.drop("y", axis=1)
        subgroup_predictions = model.predict(subgroup_features)

        # Ensure predictions are NumPy arrays
        if isinstance(subgroup_predictions, pd.Series):
            subgroup_predictions = subgroup_predictions.values
        elif not isinstance(subgroup_predictions, np.ndarray):
            subgroup_predictions = np.array(subgroup_predictions)

        # Verify lengths match
        if len(subgroup_predictions) != len(subgroup):
            raise ValueError("Number of predictions does not match number of samples in the subgroup.")

        # Calculate accuracies for the subgroup
        y_true_subgroup = subgroup["y"].values
        subgroup_accuracies = (y_true_subgroup == subgroup_predictions).astype(int)

        # Extract complementary group
        complementary_group = df.query(f"not ({query})")
        if "y" not in complementary_group.columns:
            raise KeyError("'y' column is missing from the complementary dataframe.")

        # Generate predictions for the complementary group
        complementary_features = complementary_group.drop("y", axis=1)
        complementary_predictions = model.predict(complementary_features)

        # Ensure predictions are NumPy arrays
        if isinstance(complementary_predictions, pd.Series):
            complementary_predictions = complementary_predictions.values
        elif not isinstance(complementary_predictions, np.ndarray):
            complementary_predictions = np.array(complementary_predictions)

        # Verify lengths match
        if len(complementary_predictions) != len(complementary_group):
            raise ValueError("Number of predictions does not match number of samples in the complementary group.")

        # Calculate accuracies for the complementary group
        y_true_complementary = complementary_group["y"].values
        complementary_accuracies = (y_true_complementary == complementary_predictions).astype(int)

        # Check for empty groups
        if len(subgroup_accuracies) == 0 or len(complementary_accuracies) == 0:
            print("One of the groups has no samples. Returning p-value as NaN.")
            return np.nan

        # Perform Welch's t-test
        t_stat, p_value = ttest_ind(subgroup_accuracies, complementary_accuracies, equal_var=False)

        return p_value

    except Exception as e:
        print(f"Error in welchs_t_test_for_accuracy: {e}")
        return np.nan


def mcnemars_test(df, model, query):
    # Step 1: Calculate overall model accuracy
    overall_accuracy = accuracy_score(df["y"], model.predict(df.drop("y", axis=1)))

    # Subgroup
    subgroup = df.query(query)
    subgroup_size = len(subgroup)
    subgroup_predictions = model.predict(subgroup.drop("y", axis=1))
    subgroup_actual = subgroup["y"].to_numpy()

    # Step 2: Calculate expected proportions in subgroup
    expected_correct = round(overall_accuracy * subgroup_size)
    expected_incorrect = subgroup_size - expected_correct

    # Step 3: Observe actual predictions in subgroup
    actual_correct = np.sum(subgroup_predictions == subgroup_actual)
    actual_incorrect = subgroup_size - actual_correct

    # Step 4: Apply McNemar's Test
    # Constructing the contingency table
    table = [[actual_correct, actual_incorrect], [expected_correct, expected_incorrect]]

    # Perform McNemar's test
    result = mcnemar(table, exact=False, correction=True)

    return result.pvalue


def calculate_weighted_relative_outcomes(df, query):
    df_subgroup = df.query(query)

    support = df_subgroup.shape[0] / df.shape[0]
    diff_outcomes = df_subgroup["y"].mean() - df["y"].mean()

    return support * diff_outcomes


def calculate_weighted_relative_accuracy(df, query, model):
    df_subgroup = df.query(query)
    support = df_subgroup.shape[0] / df.shape[0]
    diff_accuracy = accuracy_score(df_subgroup["y"], model.predict(df_subgroup.drop("y", axis=1))) - accuracy_score(
        df["y"], model.predict(df.drop("y", axis=1))
    )

    return support * diff_accuracy


def calculate_odds_ratio(df, query):
    """Odds ratio: (p1 * (1-p1) / (p0 * (1-p0))),
    where p1 is the probability of the outcome in the subgroup,
    and p0 is the probability of the outcome in the rest of the dataset."""

    df_subgroup = df.query(query)
    df_rest = df.query(f"not ({query})")

    p1 = df_subgroup["y"].mean()
    p0 = df_rest["y"].mean()

    return p1 * (1 - p1) / (p0 * (1 - p0))


def calculate_odds_ratio_acc(df, query, model):
    """Odds ratio: (p1 * (1-p1) / (p0 * (1-p0))),
    where p1 is the % accuracy in the subgroup, and p0 is the % accuracy in the rest of the dataset."""

    df_subgroup = df.query(query)
    df_rest = df.query(f"not ({query})")

    p1 = accuracy_score(df_subgroup["y"], model.predict(df_subgroup.drop("y", axis=1)))
    p0 = accuracy_score(df_rest["y"], model.predict(df_rest.drop("y", axis=1)))

    return p1 * (1 - p1) / (p0 * (1 - p0))


def calculate_lift(df, query):
    """Lift: p1 / p, where p is the probability of the outcome in the entire dataset"""

    df_subgroup = df.query(query)
    p1 = df_subgroup["y"].mean()
    p = df["y"].mean()

    return p1 / p


def calculate_lift_outcome(df, query, model):
    """Lift: p1 / p, where p is the accuracy of the entire dataset, and p1 is the accuracy of the subgroup"""

    df_subgroup = df.query(query)
    p1 = accuracy_score(df_subgroup["y"], model.predict(df_subgroup.drop("y", axis=1)))
    p = accuracy_score(df["y"], model.predict(df.drop("y", axis=1)))

    return p1 / p


def calculate_group_statistics(X, y, model, query, X_tr=None, num_iterations=250):
    # Calculate the dataframe
    df = deepcopy(X)
    df["y"] = y

    # clean query
    query = query.replace("\n", " ").replace("\r", " ").strip()

    # Filter the subgroup
    if len(query) == 0 or len(df.query(query)) == 0 or len(df.query(query)) == len(df):
        return {
            "group_size": 0,
            "support": 0,
            "p_value_mc": 1,
            "p_value_t": 1,
            "p_value_chi": 1,
            "p_value_bootstrap": 1,
            "num_criteria": 0,
            "outcome_diff": 0,
            "accuracy_diff": 0,
            "odds_ratio_outcome": np.nan,
            "odds_ratio_acc": np.nan,  # odds ratio of the accuracy
            "query": query,
            "lift_outcome": np.nan,
            "lift_acc": np.nan,  # lift of the accuracy
            "weighted_relative_outcome": np.nan,
            "weighted_relative_accuracy": np.nan,
        }

    subgroup = df.query(query)
    # Calculate statistics
    group_size = len(subgroup)
    relative_size = group_size / len(df)
    num_criteria = query.count("and") + 1  # Counting 'and' and adding 1 for the first condition

    # Outcome difference
    avg_outcome_dataset = y.mean()
    avg_outcome_subgroup = subgroup["y"].mean()
    outcome_diff = abs(avg_outcome_dataset - avg_outcome_subgroup)

    # Model accuracy difference
    if X_tr is not None:
        subgroup_tr = X_tr.loc[subgroup.index]
        subgroup_y = y.loc[subgroup.index]
        accuracy_dataset = accuracy_score(y, model.predict(X_tr))
        accuracy_subgroup = accuracy_score(subgroup_y, model.predict(subgroup_tr))
        accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)

    else:
        accuracy_dataset = accuracy_score(y, model.predict(X))
        accuracy_subgroup = accuracy_score(subgroup["y"], model.predict(subgroup.drop("y", axis=1)))
        accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)

    # P-value calculation (randomization-based testing)
    p_value = mcnemars_test(df, model, query)
    p_value_t = welchs_t_test_for_accuracy(df, model, query)
    p_value_chi = chi_square_test_for_accuracy(df, model, query)

    # Get odds ratio
    odds_ratio = calculate_odds_ratio(df, query)

    # Calculate lift for outcome
    lift = calculate_lift(df, query)

    # Calculate lift for accuracy
    lift_acc = calculate_lift_outcome(df, query, model)

    # Calculate weighted relative accuracy and outcomes
    wro = calculate_weighted_relative_outcomes(df, query)
    wre = calculate_weighted_relative_accuracy(df, query, model)

    # Odds ratio of accuracy
    odds_ratio_acc = calculate_odds_ratio_acc(df, query, model)

    # Bootstrap p-value acc
    pval_bootstrap = bootstrapping_test_for_accuracy(df, model, query)

    return {
        "group_size": group_size,  # size of the subgroup
        "support": relative_size,  # support of the subgroup
        "p_value_mc": p_value,  # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        "p_value_t": p_value_t,  # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        "p_value_chi": p_value_chi,  # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        "p_value_bootstrap": pval_bootstrap,  # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        "num_criteria": num_criteria,  # number of criteria in the subgroup
        "outcome_diff": outcome_diff,  # difference in the outcome between the subgroup and the entire dataset
        "accuracy_diff": accuracy_diff,  # difference in the accuracy between the subgroup and the entire dataset
        "odds_ratio_outcome": odds_ratio,  # odds ratio of the outcome
        "odds_ratio_acc": odds_ratio_acc,  # odds ratio of the accuracy
        "query": query,
        "lift_outcome": lift,
        "lift_acc": lift_acc,  # lift of the accuracy
        "weighted_relative_outcome": wro,  # Weighted relative outcomes
        "weighted_relative_accuracy": wre,  # Weighted relative accuracy
    }


def calculate_group_statistics_string(X, y, model, query, ohe, num_iterations=250):
    # Calculate the dataframe
    df = deepcopy(X)
    df["y"] = y
    # Filter the subgroup
    if len(query) == 0 or len(df.query(query)) == 0 or len(df.query(query)) == len(df):
        return {
            "group_size": 0,
            "support": 0,
            "p_value_mc": 1,
            "p_value_t": 1,
            "p_value_chi": 1,
            "p_value_bootstrap": 1,
            "num_criteria": 0,
            "outcome_diff": 0,
            "accuracy_diff": 0,
            "odds_ratio_outcome": np.nan,
            "odds_ratio_acc": np.nan,  # odds ratio of the accuracy
            "query": query,
            "lift_outcome": np.nan,
            "lift_acc": np.nan,  # lift of the accuracy
            "weighted_relative_outcome": np.nan,
            "weighted_relative_accuracy": np.nan,
        }

    subgroup = df.query(query)
    # Calculate statistics
    group_size = len(subgroup)
    relative_size = group_size / len(df)
    num_criteria = query.count("and") + 1  # Counting 'and' and adding 1 for the first condition

    # Outcome difference
    avg_outcome_dataset = y.mean()
    avg_outcome_subgroup = subgroup["y"].mean()
    outcome_diff = abs(avg_outcome_dataset - avg_outcome_subgroup)

    # Transform the dataframe (except for y) to one-hot encoding
    df_ohe = pd.DataFrame(ohe.transform(df.drop("y", axis=1)), columns=ohe.get_feature_names_out())
    df_ohe["y"] = df["y"]

    subgroup_ohe = pd.DataFrame(ohe.transform(subgroup.drop("y", axis=1)), columns=ohe.get_feature_names_out())
    subgroup_ohe["y"] = subgroup["y"]

    X_ohe = df_ohe.drop("y", axis=1)

    accuracy_dataset = accuracy_score(y, model.predict(X_ohe))
    accuracy_subgroup = accuracy_score(subgroup["y"], model.predict(subgroup_ohe.drop("y", axis=1)))
    accuracy_diff = abs(accuracy_dataset - accuracy_subgroup)

    # Bootstrap p-value acc
    pval_bootstrap = bootstrapping_test_for_accuracy_string(df_ohe, model, subgroup_ohe)

    return {
        "group_size": group_size,  # size of the subgroup
        "support": relative_size,  # support of the subgroup
        "p_value_bootstrap": pval_bootstrap,  # p-value for evaluating whether the accuracy is different in the subgroup from average accuracy
        "num_criteria": num_criteria,  # number of criteria in the subgroup
        "outcome_diff": outcome_diff,  # difference in the outcome between the subgroup and the entire dataset
        "accuracy_diff": accuracy_diff,  # difference in the accuracy between the subgroup and the entire dataset
        "query": query,
    }


def compute_differences_metrics_two_datasets(metrics_1, metrics_2):
    """Computes the differences between many metrics between the two datasets X1 and X2 that
    might come from different populations, or be simple train-test splits.

    IMPORTANT: Differences are calculated as X2 - X1, so a positive difference means that X2 is higher than X1."""

    # Calculate the differences
    metrics_diff = {}
    for metric in metrics_1:
        # CHeck if numeric
        if isinstance(metrics_1[metric], (int, float)):
            metrics_diff[metric] = metrics_2[metric] - metrics_1[metric]
        else:
            metrics_diff[metric] = None

    return metrics_diff

import pandas as pd

# Paths to NISQA results for each model
paths = {
    'RNNoise': r'F:\data\NISQARNNoise\NISQA_results.csv',
    'Test': r'F:\data\NISQATEST\NISQA_results.csv',
    'Dense': r'F:\data\NISQADENSE\NISQA_results.csv'
}

# Function to read NISQA results and calculate the average prediction scores
def get_average_scores(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Calculate the average prediction score for each metric
        averages = {
            'mos_pred': df['mos_pred'].mean(),
            'noi_pred': df['noi_pred'].mean(),
            'dis_pred': df['dis_pred'].mean(),
            'col_pred': df['col_pred'].mean(),
            'loud_pred': df['loud_pred'].mean()
        }
        return averages
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except KeyError as e:
        print(f"A required column was not found in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None

# Dictionary to hold the average prediction scores for each model and metric
average_scores = {}

# Calculate and print the average prediction scores for each model
for model_name, file_path in paths.items():
    scores = get_average_scores(file_path)
    if scores is not None:
        average_scores[model_name] = scores
        print(f"Average prediction scores for {model_name}: {scores}")

# Function to compare models based on a particular metric
def compare_models(metric, models_scores):
    baseline_score = models_scores['RNNoise'][metric]
    print(f"nComparison based on {metric}:")
    for model_name, scores in models_scores.items():
        if model_name != 'RNNoise':
            difference = scores[metric] - baseline_score
            print(f"{model_name} is {'better' if difference > 0 else 'worse'} than RNNoise by {abs(difference):.4f} for {metric}")

# Perform comparison for each metric
metrics = ['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred']
for metric in metrics:
    compare_models(metric, average_scores)

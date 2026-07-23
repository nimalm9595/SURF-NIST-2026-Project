#%%
import json
import os
import sys
import sklearn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.scikitlearn.digits_randomforest import (
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    accuracy
)
print(type(model))


def make_json_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    
    if isinstance(value, np.integer):
        return int(value)
    
    if isinstance(value, np.floating):
        return float(value)
    
    if isinstance(value, np.bool_):
        return bool(value)
    
    if isinstance(value, tuple):
        return list(value)
    
    if isinstance(value, dict):
        return {
            key : make_json_serializable(item)
            for key, item in value.items()
        }
    
    if isinstance(value, list):
        return [
            make_json_serializable(item)
            for item in value    
        ]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)

def extract_hyperparameters(model):
    return make_json_serializable(model.get_params())

def extract_learned_attributes(model):
    learned_attributes = {}

    excluded_attributes = {
        "estimators_"
    }

    for attribute_name in dir(model):
        if not attribute_name.endswith("_"):
            continue

        if attribute_name.startswith("__"):
            continue

        if attribute_name in excluded_attributes:
            continue

        try:
            value = getattr(model, attribute_name)
        except Exception:
            continue


        if callable(value):
            continue

        learned_attributes[attribute_name] = make_json_serializable(value)

    return learned_attributes

def extract_model_metadata(model):
    metadata = {
        "framework" : {
            "name": "scikit-learn",
            "version": sklearn.__version__   
        },
        "model": {
            "class_name": type(model).__name__,
            "module": type(model).__module__,
            "task_type": "classification",
            "model_family": "ensemble"
        },
        "hyperparameters": extract_hyperparameters(model),
        "learned_attributes": extract_learned_attributes(model),
        "dataset": {
            "dataset_name": "Digits",
            "training_samples": len(X_train),
            "testing_samples": len(X_test),
            "number_of_features": X_train.shape[1],
            "number_of_classes": len(np.unique(y_train))
        },
        "evaluation": {
            "metric_name": "accuracy",
            "metric_value": float(accuracy)
        }
    }

    return metadata

def save_metadata(metadata, output_path):
    output_directory = os.path.dirname(output_path)


    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)
    

def main():
    metadata = extract_model_metadata(model)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        "scikitlearn_digits_metadata.json"
    )

    save_metadata(metadata, output_path)

    print(json.dumps(metadata, indent=4))
    print(f"\nMetadata saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()









# %%

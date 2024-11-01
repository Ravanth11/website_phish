import pandas as pd
from joblib import load  # Correct import for joblib

# Load your trained models
logistic_model = load('logistic_model.pkl')
rf_model = load('rf_model.pkl')

# Load the scaler used during training
scaler = load('scaler.pkl')

# Prepare the input data
input_data = pd.DataFrame([{
    'url_length': 37,
    'number_of_dots_in_url': 2,
    'having_repeated_digits_in_url': 0,
    'number_of_digits_in_url': 0,
    'number_of_special_char_in_url': 8,
    'number_of_hyphens_in_url': 0,
    'number_of_underline_in_url': 0,
    'number_of_slash_in_url': 5,
    'number_of_questionmark_in_url': 0,
    'number_of_equal_in_url': 0,
    'number_of_at_in_url': 0,
    'number_of_dollar_in_url': 0,
    'number_of_exclamation_in_url': 0,
    'number_of_hashtag_in_url': 0,
    'number_of_percent_in_url': 0,
    'domain_length': 12,
    'number_of_dots_in_domain': 2,
    'number_of_hyphens_in_domain': 0,
    'having_special_characters_in_domain': 0,
    'number_of_special_characters_in_domain': 0,
    'having_digits_in_domain': 0,
    'number_of_digits_in_domain': 0,
    'having_repeated_digits_in_domain': 0,
    'number_of_subdomains': 2,
    'having_dot_in_subdomain': 0,
    'having_hyphen_in_subdomain': 0,
    'average_subdomain_length': 3,
    'average_number_of_dots_in_subdomain': 0,
    'average_number_of_hyphens_in_subdomain': 0,
    'having_special_characters_in_subdomain': 1,
    'number_of_special_characters_in_subdomain': 3,
    'having_digits_in_subdomain': 0,
    'number_of_digits_in_subdomain': 0,
    'having_repeated_digits_in_subdomain': 0,
    'having_path': 1,
    'path_length': 3,
    'having_query': 0,
    'having_fragment': 0,
    'having_anchor': 0,
    'entropy_of_url': 4.010412069,
    'entropy_of_domain': 2.751629167
}])

# Apply the same preprocessing steps to the input data
input_data = scaler.transform(input_data)

# Make predictions with both models
logistic_prediction = logistic_model.predict(input_data)
rf_prediction = rf_model.predict(input_data)

print(f"Logistic Regression Prediction: {'Phishing' if logistic_prediction[0] == 1 else 'Not Phishing'}")
print(f"Random Forest Prediction: {'Phishing' if rf_prediction[0] == 1 else 'Not Phishing'}")

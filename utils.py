from sklearn.preprocessing import LabelEncoder


def convert_category_to_label(categories: list) -> list:
    label_encoder = LabelEncoder()
    encoded_categories = label_encoder.fit_transform(categories)
    original_data = label_encoder.inverse_transform(encoded_categories)
    return encoded_categories, original_data

import pandas as pd
import joblib

# Load the trained model
def load_model(model_path='model/mushroom_classifier.pkl'):
    return joblib.load(model_path)

def prepare_features(input_features):
    feature_names = [
        'cap_shape_b', 'cap_shape_c', 'cap_shape_f', 'cap_shape_k', 'cap_shape_s', 'cap_shape_x',
        'cap_surface_f', 'cap_surface_g', 'cap_surface_s', 'cap_surface_y',
        'cap_color_b', 'cap_color_c', 'cap_color_e', 'cap_color_g', 'cap_color_n', 'cap_color_p', 'cap_color_r', 'cap_color_u', 'cap_color_w', 'cap_color_y',
        'bruises_f', 'bruises_t',
        'odor_a', 'odor_c', 'odor_f', 'odor_l', 'odor_m', 'odor_n', 'odor_p', 'odor_s', 'odor_y',
        'gill_attachment_a', 'gill_attachment_d', 'gill_attachment_f', 'gill_attachment_n',
        'gill_spacing_c', 'gill_spacing_d', 'gill_spacing_w',
        'gill_size_b', 'gill_size_n',
        'gill_color_b', 'gill_color_e', 'gill_color_g', 'gill_color_h', 'gill_color_k', 'gill_color_n', 'gill_color_o', 'gill_color_p', 'gill_color_r', 'gill_color_u', 'gill_color_w', 'gill_color_y',
        'stalk_shape_e', 'stalk_shape_t',
        'stalk_root_?', 'stalk_root_b', 'stalk_root_c', 'stalk_root_e', 'stalk_root_r', 'stalk_root_u', 'stalk_root_z',
        'stalk_surface_above_ring_f', 'stalk_surface_above_ring_k', 'stalk_surface_above_ring_s', 'stalk_surface_above_ring_y',
        'stalk_surface_below_ring_f', 'stalk_surface_below_ring_k', 'stalk_surface_below_ring_s', 'stalk_surface_below_ring_y',
        'stalk_color_above_ring_b', 'stalk_color_above_ring_c', 'stalk_color_above_ring_e', 'stalk_color_above_ring_g', 'stalk_color_above_ring_n', 'stalk_color_above_ring_o', 'stalk_color_above_ring_p', 'stalk_color_above_ring_w', 'stalk_color_above_ring_y',
        'stalk_color_below_ring_b', 'stalk_color_below_ring_c', 'stalk_color_below_ring_e', 'stalk_color_below_ring_g', 'stalk_color_below_ring_n', 'stalk_color_below_ring_o', 'stalk_color_below_ring_p', 'stalk_color_below_ring_w', 'stalk_color_below_ring_y',
        'veil_type_p', 'veil_type_u',
        'veil_color_n', 'veil_color_o', 'veil_color_w', 'veil_color_y',
        'ring_number_n', 'ring_number_o', 'ring_number_t',
        'ring_type_c', 'ring_type_e', 'ring_type_f', 'ring_type_l', 'ring_type_n', 'ring_type_p', 'ring_type_s', 'ring_type_z',
        'spore_print_color_b', 'spore_print_color_h', 'spore_print_color_k', 'spore_print_color_n', 'spore_print_color_o', 'spore_print_color_r', 'spore_print_color_u', 'spore_print_color_w', 'spore_print_color_y',
        'population_a', 'population_c', 'population_n', 'population_s', 'population_v', 'population_y',
        'habitat_d', 'habitat_g', 'habitat_l', 'habitat_m', 'habitat_p', 'habitat_u', 'habitat_w'
    ]
    
    df = pd.DataFrame(columns=feature_names, index=[0]).fillna(0)
    
    for feature, value in input_features.items():
        column_name = f"{feature}_{value}"
        if column_name in df.columns:
            df.at[0, column_name] = 1
    
    return df

def make_prediction(model, prepared_features):
    prediction = model.predict(prepared_features)
    return prediction[0]

if __name__ == "__main__":
    model = load_model()

    input_features = {
        'cap_shape': 'x',
        'cap_surface': 's',
        'cap_color': 'n',
        'bruises': 't',
        'odor': 'p',
        'gill-attachment': 'a',
        'gill-spacing': 'c',
        'gill-size': 'n',
        'gill-color': 'e',
        'stalk-shape': 't',
        'stalk-root': 'c',
        'stalk-surface-above-ring': 'f',
        'stalk-surface-below-ring': 's',
        'stalk-color-above-ring': 'c',
        'stalk-color-below-ring': 'g',
        'veil-type': 'u',
        'veil-color': 'o',
        'ring-number': 't',
        'ring-type': 'z',
        'spore-print-color': 'y',
        'population': 'y',
        'habitat': 'd'
    }

    prepared_features = prepare_features(input_features)
    prediction = make_prediction(model, prepared_features)
    print(f"Prediction: {'Poisonous' if prediction else 'Edible'}")

# Required Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Attention
from hmmlearn import hmm
from skfuzzy import control as ctrl
import osmnx as ox
import networkx as nx
import pandas as pd

def load_geolife_data(filepath):
    """
    Load and preprocess GeoLife dataset.
    :param filepath: Path to the GeoLife dataset.
    :return: Processed GNSS data.
    """
    # Load GeoLife dataset (assuming it's a CSV with 'latitude' and 'longitude' columns)
    data = pd.read_csv(filepath)
    gnss_data = data[['latitude', 'longitude']].values
    return preprocess_gnss_data(gnss_data)

# Step 1: GNSS Data Preprocessing Function
def preprocess_gnss_data(gnss_data):
    """
    Preprocess GNSS data (e.g., remove noise, handle missing values)
    :param gnss_data: Raw GNSS coordinates as a time series.
    :return: Processed GNSS data.
    """
    # Handle missing or noisy GNSS data by filling NaNs with the mean of the array
    processed_data = np.nan_to_num(gnss_data, nan=np.mean(gnss_data, axis=0))
    return processed_data

# Step 2: Transfer Learning for Feature Extraction
def transfer_learning_feature_extraction(input_data):
    """
    Uses a pre-trained model (e.g., ResNet) to extract features from GNSS data (e.g., time series).
    :param input_data: Input GNSS data (e.g., coordinates, time series).
    :return: Extracted features.
    """
    # Placeholder for using ResNet (or similar model) for feature extraction
    base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(128)(x)

    model = tf.keras.Model(inputs, outputs)
    features = model.predict(input_data)

    return features

# Step 4: Sequence Learning with Transformer Model for GNSS Data
def build_transformer_model_with_classification(input_shape):
    """
    Build Transformer model to handle GNSS sequences and classify road type.
    :param input_shape: Shape of input GNSS data.
    :return: Transformer model with classification output.
    """
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    attention = Attention()([x, x])
    x = TimeDistributed(Dense(128, activation='relu'))(attention)

    # Original transformer output (for GNSS prediction)
    gnss_output = Dense(2)(x)  # Output: 2D coordinates (latitude, longitude)

    # Road type classification (new task)
    x_classification = TimeDistributed(Dense(64, activation='relu'))(x)
    road_type_output = Dense(2, activation='softmax')(x_classification)  # 2 classes: highway, service road

    # Build final model
    model = tf.keras.Model(inputs, [gnss_output, road_type_output])
    model.compile(optimizer='adam',
                  loss=['mse', 'categorical_crossentropy'],
                  metrics=['accuracy'])

    return model

# Step 4 (Modified): OSM Data Preprocessing with Road Type Classification
def preprocess_osm_data_with_road_type(location):
    """
    Preprocess OSM data to extract road networks and classify road types (e.g., highway, service road).
    :param location: Name of the place to download OSM data.
    :return: Road network graph with road type labels.
    """
    # Get the OSM graph for the given location
    graph = ox.graph_from_place(location, network_type='drive')
    simple_graph = ox.utils_graph.get_undirected(graph)
    nodes, edges = ox.graph_to_gdfs(simple_graph)

    # Classify roads as highway or service road based on OSM tags
    edges['road_type'] = edges['highway'].apply(lambda x: 'highway' if 'motorway' in x else 'service_road' if 'service' in x else 'other')

    return nodes, edges, simple_graph

# Step 5: Map Matching Using OSM Data
def map_match_with_osm_and_road_type(gnss_data, osm_graph, edges):
    """
    Snap GNSS points to the nearest road in the OSM road network and return road type.
    :param gnss_data: GNSS coordinates as a list of (lat, lon).
    :param osm_graph: OSM road network graph.
    :param edges: Edge GeoDataFrame containing road types.
    :return: List of matched road segments and road types.
    """
    matched_points = []
    road_types = []

    for point in gnss_data:
        nearest_node = ox.distance.nearest_nodes(osm_graph, point[1], point[0])  # lon, lat format
        nearest_edge = edges.loc[edges['u'] == nearest_node]
        road_type = nearest_edge['road_type'].values[0] if not nearest_edge.empty else 'unknown'

        matched_points.append(nearest_node)
        road_types.append(road_type)

    return matched_points, road_types

# Step 6: Apply HMM with OSM-enhanced Map Matching
def apply_osm_hmm(gnss_data, osm_graph):
    """
    Use HMM for map-matching, enhanced with OSM data.
    :param gnss_data: Preprocessed GNSS data.
    :param osm_graph: OSM road network graph.
    :return: Sequence of road segments.
    """
    matched_segments = map_match_with_osm(gnss_data, osm_graph)
    n_states = len(set(matched_segments))  # Number of unique road segments
    model = hmm.MultinomialHMM(n_components=n_states, n_iter=100)
    obs = np.array(matched_segments).reshape(-1, 1)
    model.fit(obs)
    predicted_segments = model.predict(obs)

    return predicted_segments

# Step 7: Fuzzy Logic for Handling Ambiguities
def apply_fuzzy_logic(gnss_accuracy, signal_strength):
    """
    Apply fuzzy logic to handle GNSS uncertainties and make decisions on accuracy.
    :param gnss_accuracy: GNSS signal accuracy as input.
    :param signal_strength: GNSS signal strength.
    :return: Fuzzy decision output.
    """
    accuracy = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'accuracy')
    strength = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'strength')
    decision = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'decision')

    accuracy['poor'] = ctrl.trapmf(accuracy.universe, [0, 0, 0.2, 0.4])
    accuracy['good'] = ctrl.trimf(accuracy.universe, [0.3, 0.5, 0.7])
    accuracy['excellent'] = ctrl.trapmf(accuracy.universe, [0.6, 0.8, 1.0, 1.0])
    strength['low'] = ctrl.trapmf(strength.universe, [0, 0, 0.3, 0.5])
    strength['medium'] = ctrl.trimf(strength.universe, [0.4, 0.6, 0.8])
    strength['high'] = ctrl.trapmf(strength.universe, [0.7, 0.9, 1.0, 1.0])
    decision['reject'] = ctrl.trapmf(decision.universe, [0, 0, 0.2, 0.4])
    decision['accept'] = ctrl.trapmf(decision.universe, [0.6, 0.8, 1.0, 1.0])

    rule1 = ctrl.Rule(accuracy['poor'] & strength['low'], decision['reject'])
    rule2 = ctrl.Rule(accuracy['good'] & strength['medium'], decision['accept'])
    rule3 = ctrl.Rule(accuracy['excellent'] & strength['high'], decision['accept'])

    decision_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    decision_sim = ctrl.ControlSystemSimulation(decision_ctrl)

    decision_sim.input['accuracy'] = gnss_accuracy
    decision_sim.input['strength'] = signal_strength
    decision_sim.compute()

    return decision_sim.output['decision']

def integrate_cam_data(cam_data, gnss_data):
    """
    Integrate CAM data from nearby vehicles for better map matching.
    :param cam_data: CAM message data (location, speed, status).
    :param gnss_data: Original GNSS data.
    :return: Adjusted GNSS data after considering CAM inputs.
    """
    # Here we assume CAM data is in the format: [latitude, longitude, speed, status]
    # For now, let's integrate CAM data by adjusting GNSS data with the average of nearby CAM points
    cam_locations = np.array(cam_data)[:, :2]  # Extract only location data
    if len(cam_locations) > 0:
        # Compute average location from CAM data
        avg_cam_location = np.mean(cam_locations, axis=0)
        # Adjust GNSS data based on CAM average location (simple example)
        adjusted_data = gnss_data + (avg_cam_location - np.mean(gnss_data, axis=0))
    else:
        adjusted_data = gnss_data

    return adjusted_data

# Step 10: Complete Model Pipeline with Road Type Prediction
def main_model_pipeline_with_road_type(gnss_data_path, cam_data_path, location):
    """
    Complete pipeline for GNSS-based map-matching with HMM, Transformer, Fuzzy Logic, and CAM integration.
    Also includes classification of road type (highway or service road).
    :param gnss_data_path: Path to the GeoLife GNSS data file.
    :param cam_data_path: Path to the CAM data file.
    :param location: OSM location name to download data.
    """
    # Load and preprocess GeoLife data
    gnss_data = load_geolife_data(gnss_data_path)
    processed_data = preprocess_gnss_data(gnss_data)

    # Download and preprocess OSM data with road types
    nodes, edges, osm_graph = preprocess_osm_data_with_road_type(location)

    # Load CAM data
    cam_data = pd.read_csv(cam_data_path).values  # Assuming CAM data is in CSV format

    # Integrate CAM data with GNSS data
    adjusted_gnss_data = integrate_cam_data(cam_data, processed_data)

    # Extract features using Transfer Learning
    features = transfer_learning_feature_extraction(adjusted_gnss_data)

    # Apply Transformer for sequence learning and road type classification
    transformer_model = build_transformer_model_with_classification(input_shape=(None, 128))
    transformer_model.load_weights('transformer_model_weights.h5')  # Load pre-trained weights
    predicted_gnss, predicted_road_type = transformer_model.predict(features)

    # Apply HMM with OSM-enhanced map matching
    road_states, road_types = map_match_with_osm_and_road_type(predicted_gnss, osm_graph, edges)

    # Use fuzzy logic for decision making
    decision = apply_fuzzy_logic(gnss_accuracy=0.85, signal_strength=0.75)

    print(f"Final GNSS Prediction: {predicted_gnss}")
    print(f"Predicted Road Type: {predicted_road_type}")
    print(f"Fuzzy Decision: {decision}")

# Example usage
gnss_data_path = 'path_to_geolife_data.csv'
cam_data_path = 'path_to_cam_data.csv'
osm_location = 'New York, USA'
main_model_pipeline_with_road_type(gnss_data_path, cam_data_path, osm_location)
# Map_Matching_Road_Classification
Advanced GNSS-Based Map-Matching with Road Classification
This project focuses on building an innovative GNSS-based map-matching solution using advanced machine learning techniques, OpenStreetMap (OSM) data, and Cooperative Awareness Messages (CAM). The model enhances positioning accuracy by integrating multiple layers of data and models to predict vehicle location and classify road types (highway or service road) in real time.

Features
1. GNSS Preprocessing: Cleans and processes raw GNSS data to reduce noise and handle missing values.
2. Transfer Learning: Utilizes pre-trained models (like ResNet) for feature extraction from GNSS data.
3. Transformer Model: Sequence learning for map-matching and road classification (highway/service road).
4. Hidden Markov Model (HMM): Enhanced map-matching using OSM road network data.
5. Fuzzy Logic: Handles GNSS uncertainties and improves decision-making.
6. CAM Integration: Incorporates data from nearby vehicles for refined positioning.
   
Technologies Used
1. Python
2. TensorFlow/Keras
3. HMMlearn
4. Scikit-Fuzzy
5. OSMnx & NetworkX
6. OpenStreetMap (OSM) Data
7. Cooperative Awareness Messages (CAM)
   
How It Works
1. GNSS Preprocessing: Raw GNSS data is cleaned and processed to handle noise.
2. Feature Extraction: Transfer learning is applied to extract key features.
3. Transformer Sequence Learning: The Transformer model predicts GNSS points and road types.
4. OSM Integration & HMM: GNSS points are map-matched using OSM road networks and HMM.
5. Fuzzy Logic Decision Making: GNSS accuracy and signal strength are fed into fuzzy logic to improve decision accuracy.
6. CAM Data Integration: CAM data from nearby vehicles refines the final GNSS positioning.
   
Usage
Clone this repository, install the necessary dependencies, and run the script on GNSS data alongside OSM data for your region to predict vehicle positioning and classify road types in real time.

Contact
Feel free to reach out with any feedback or suggestions!

<<<<<<< HEAD
# ROADNET

RoadNet is a computer vision project that leverages a 3D simulated environment in Unity to generate synthetic datasets. These datasets are used to train deep learning models for semantic segmentation and depth estimation of urban road scenes.

## PROJECT SUMMARY

We created a Unity simulation that includes a car model, a road, and urban elements such as buildings, sidewalks, and traffic lights.

From this environment, we captured datasets containing RGB images, semantic segmentation masks, and depth maps. These were later used to train two separate neural networks:

- One for semantic segmentation, which identifies key elements in the scene such as roads, buildings, vehicles, traffic lights, and roadblocks.
- Another for monocular depth estimation, which predicts the distance from the camera to each point in the scene.

Although our initial goal was to implement autonomous driving within the Unity simulation, this feature was not fully achieved. However, the environment proved to be a valuable tool for generating data and training vision models.

The combination of segmentation and depth perception allows the system to not only recognize the presence of critical elements (e.g., a traffic light), but also estimate how far they are. This information is essential for deciding whether an immediate driving action is required, such as stopping or slowing down.

➡️ **You can find the datasets we used and a working version of the Unity environment (with the car ready to run) in this Google Drive folder**:  
📁 [RoadNet Datasets and Simulation](https://drive.google.com/drive/folders/1quDstc_EpigkF57H0DyeBs-2vHb7s1h2?usp=drive_link)

## Project Structure

RoadNet/
├── RoadNet+Perception/ # Unity project: 3D environment with road, car, buildings
├── decisionNeuralNetwork/ # Placeholder for decision-making model (not functional)
├── depthEstimation/ # Training and evaluation code for depth estimation
├── segmentation/ #Segmentation Model
├── requirements/ # Project requirements and setup guidelines
├── .gitignore # Files and folders excluded from version control
└── ...


## Components Overview

This project is composed of multiple modules that support the training and evaluation of vision-based models for road scene understanding.

### `RoadNet+Perception/`

This folder contains the Unity project used to generate synthetic data. It includes:

- A 3D environment with a road, vehicles, buildings, sidewalks, and traffic lights.
- A virtual camera mounted on a vehicle to simulate driving through the environment.
- Manual driving was used to collect diverse scene images from different angles and lighting conditions.

### `segmentation/`

A standalone inference module for semantic segmentation. It enables prediction and visualization of masks from new input images using a pre-trained model.

- Uses a U-Net with a ResNet18 encoder.
- Supports 5 semantic classes: `road`, `building`, `car`, `traffic_light`, and `road_block`, each with a distinct color.
- Includes preprocessing and postprocessing methods to convert model outputs into visual masks.
- CLI usage: e.g., `python inference.py path/to/image.jpg` to process and save the result.

This module is useful for quick testing and visualization of segmentation results.


### `depthEstimation/`

This module focuses on monocular depth estimation:

- Trains a neural network to predict a depth map from a single RGB image.
- The predicted map provides information about the **distance** to each object in the scene.
- Useful for understanding **how far away** a traffic light, car, or obstacle is, in order to determine **if immediate action is required**.

The depth estimation complements the segmentation model by adding spatial awareness to scene understanding.

### `decisionNeuralNetwork/`

A placeholder directory intended for a future decision-making module (e.g., to control autonomous vehicle behavior based on visual input). Currently not implemented.

---

## Notes

- **Unity version**: _(6000.1.5f1)_
- **Dataset format**: RGB images, categorical masks (for segmentation), and EXR depth maps (for depth estimation).
- **Model types**: CNN-based models for segmentation and depth estimation (e.g., U-Net variants).

## Authors

This project was developed by:

- **Alejandra Castaño Tobón**
- **Daniel Correa Carreño**
- **Nixon Lizcano Santana**
=======
# ROADNET

RoadNet is a computer vision project that leverages a 3D simulated environment in Unity to generate synthetic datasets. These datasets are used to train deep learning models for semantic segmentation and depth estimation of urban road scenes.

## PROJECT SUMMARY

We created a Unity simulation that includes a car model, a road, and urban elements such as buildings, sidewalks, and traffic lights.

From this environment, we captured datasets containing RGB images, semantic segmentation masks, and depth maps. These were later used to train two separate neural networks:

- One for semantic segmentation, which identifies key elements in the scene such as roads, buildings, vehicles, traffic lights, and roadblocks.
- Another for monocular depth estimation, which predicts the distance from the camera to each point in the scene.

Although our initial goal was to implement autonomous driving within the Unity simulation, this feature was not fully achieved. However, the environment proved to be a valuable tool for generating data and training vision models.

The combination of segmentation and depth perception allows the system to not only recognize the presence of critical elements (e.g., a traffic light), but also estimate how far they are. This information is essential for deciding whether an immediate driving action is required, such as stopping or slowing down.

➡️ **You can find the datasets we used and a working version of the Unity environment (with the car ready to run) in this Google Drive folder**:  
📁 [RoadNet Datasets and Simulation](https://drive.google.com/drive/folders/1quDstc_EpigkF57H0DyeBs-2vHb7s1h2?usp=drive_link)

## Project Structure

RoadNet/
├── RoadNet+Perception/ # Unity project: 3D environment with road, car, buildings
├── decisionNeuralNetwork/ # Placeholder for decision-making model (not functional)
├── depthEstimation/ # Training and evaluation code for depth estimation
├── segmentation/ #Segmentation Model
├── requirements/ # Project requirements and setup guidelines
├── .gitignore # Files and folders excluded from version control
└── ...


## Components Overview

This project is composed of multiple modules that support the training and evaluation of vision-based models for road scene understanding.

### `RoadNet+Perception/`

This folder contains the Unity project used to generate synthetic data. It includes:

- A 3D environment with a road, vehicles, buildings, sidewalks, and traffic lights.
- A virtual camera mounted on a vehicle to simulate driving through the environment.
- Manual driving was used to collect diverse scene images from different angles and lighting conditions.

### `segmentation/`

A standalone inference module for semantic segmentation. It enables prediction and visualization of masks from new input images using a pre-trained model.

- Uses a U-Net with a ResNet18 encoder.
- Supports 5 semantic classes: `road`, `building`, `car`, `traffic_light`, and `road_block`, each with a distinct color.
- Includes preprocessing and postprocessing methods to convert model outputs into visual masks.
- CLI usage: e.g., `python inference.py path/to/image.jpg` to process and save the result.

This module is useful for quick testing and visualization of segmentation results.


### `depthEstimation/`

This module focuses on monocular depth estimation:

- Trains a neural network to predict a depth map from a single RGB image.
- The predicted map provides information about the **distance** to each object in the scene.
- Useful for understanding **how far away** a traffic light, car, or obstacle is, in order to determine **if immediate action is required**.

The depth estimation complements the segmentation model by adding spatial awareness to scene understanding.

### `decisionNeuralNetwork/`

A placeholder directory intended for a future decision-making module (e.g., to control autonomous vehicle behavior based on visual input). Currently not implemented.

---

## Notes

- **Unity version**: _(6000.1.5f1)_
- **Dataset format**: RGB images, categorical masks (for segmentation), and EXR depth maps (for depth estimation).
- **Model types**: CNN-based models for segmentation and depth estimation (e.g., U-Net variants).

## Authors

This project was developed by:

- **Alejandra Castaño Tobón**
- **Daniel Correa Carreño**
- **Nixon Lizcano Santana**
>>>>>>> 30d8f8852ba238b7508de7b4b93942a965e9e6de

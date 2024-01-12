## **Product Recommendation Program**

This program makes personalized shopping recommendations based on user input criteria. It uses a machine learning model trained on shopping trends data to predict what product a user is most likely to purchase along with a recommended color.

**Usage**
The main components are:
- `data.py`: Contains data loading and preprocessing functions
- `image.py`: Handles generating product images with the predicted color
- `gui.py`: Contains GUI code to get user input criteria
- `main.py`: Builds and trains a HistGradientBoostingClassifier model to make product predictions and ties everything together, makes predictions based on user input and displays outputs

To run, simply execute `main.py`. The program will:
- Load and preprocess data
- Train machine learning models
- Launch a GUI to get user criteria
- Make product and color predictions
- Display recommended product and color
- Ask if user wants to try other criteria
- Key output is a recommendation image showing the predicted product in the predicted color.

**Requirements**
Requires Python and packages listed in requirements.txt, mainly:
    scikit-learn
    matplotlib
    seaborn
    easygui
    tqdm

First install requirements with:

    pip install -r requirements.txt


**Customization**
The main areas to customize the model are:
Try different machine learning models in `model.py`
Adjust model hyperparameters for tuning
Modify categorical mappings to handle new product/color values
Swap out image generation approach in `image.py`
Overall, the modular design allows easily changing model, data, or outputs.

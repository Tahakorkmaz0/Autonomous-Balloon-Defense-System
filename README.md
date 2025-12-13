# 🏠 House Price Prediction System

A Machine Learning application built with PyTorch that predicts house prices based on key property features using a Linear Regression Neural Network.

## 📋 Features

- **Interactive Prediction System**: Input house details and get instant price predictions
- **PyTorch Neural Network**: Implements linear regression using `nn.Module`
- **Data Normalization**: Uses Z-score standardization for accurate predictions
- **User-Friendly Interface**: Simple command-line interface with emoji indicators
- **Error Handling**: Validates user inputs and provides helpful error messages

## 🔧 Requirements

- Python 3.14.0 (or compatible version)
- PyTorch 2.9.1
- NumPy

## 📦 Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     .venv\Scripts\Activate.ps1
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required packages**:
   ```bash
   py -m pip install torch numpy
   ```

## 🚀 Usage

Run the program:
```bash
python deneme.py
```

### Input Parameters

The system will ask you to enter the following information:

1. **📐 Square meters (m²)**: Total area of the property
2. **🚪 Number of rooms**: Count of rooms in the house
3. **📅 Building age (years)**: How old the building is
4. **🏢 Floor number**: Which floor the property is on

### Example Session

```
============================================================
🏠 HOUSE PRICE PREDICTION SYSTEM 🏠
============================================================

Please enter house information:
------------------------------------------------------------
📐 Square meters (m²): 120
🚪 Number of rooms: 3
📅 Building age (years): 5
🏢 Floor number: 4

============================================================
📊 PREDICTION RESULT
============================================================
🏠 House Details:
   • Square meters: 120.0 m²
   • Number of rooms: 3
   • Building age: 5 years
   • Floor: 4

💰 Predicted Price: $1,187,234.56
============================================================

------------------------------------------------------------
Would you like to predict another house price? (y/n): 
```

## 🧠 How It Works

### 1. **Data Preparation**
   - Uses a dataset of 20 sample houses with 4 features each
   - Features: Square meters, number of rooms, building age, floor number
   - Normalizes data using Z-score standardization

### 2. **Model Architecture**
   - Single linear layer neural network
   - Input: 4 features
   - Output: 1 predicted price
   - Mathematical formula: `y = Wx + b`

### 3. **Training Process**
   - **Loss Function**: Mean Squared Error (MSE)
   - **Optimizer**: Stochastic Gradient Descent (SGD)
   - **Learning Rate**: 0.01
   - **Epochs**: 2000
   - Final Mean Absolute Error (MAE): ~$7,280

### 4. **Prediction**
   - Normalizes new input using training statistics
   - Passes through trained model
   - Returns predicted price in USD

## 📊 Training Dataset

The model is trained on 20 real estate examples with the following ranges:
- **Square meters**: 50-160 m²
- **Rooms**: 1-4
- **Building age**: 0-25 years
- **Floor**: 1-10
- **Prices**: $500,000 - $1,600,000

## 🎯 Model Performance

After 2000 training epochs:
- **MSE Loss**: ~71,253,872
- **MAE**: ~$7,280
- **RMSE**: ~$8,441

*Note: The model's accuracy is limited by the small training dataset (20 samples). For production use, a larger dataset would significantly improve predictions.*

## 🔍 Code Structure

```
deneme.py
├── Data Preparation (Lines 1-51)
│   └── Dataset creation and normalization
├── Model Architecture (Lines 54-80)
│   └── HousePricePredictor class (nn.Module)
├── Training Loop (Lines 98-128)
│   └── 2000 epochs with progress logging
├── Helper Function (Lines 136-152)
│   └── predict_price() for new predictions
└── User Interface (Lines 154-200)
    └── Interactive input/output system
```

## 🛠️ Customization

### Modify Training Parameters

```python
LEARNING_RATE = 0.01  # Adjust learning speed
EPOCHS = 2000         # Change number of training iterations
```

### Add More Features

To include additional features (e.g., number of bathrooms, parking spaces):
1. Add data to the dataset lists
2. Update `INPUT_DIM` accordingly
3. Modify the input normalization and prediction function

### Upgrade to Deep Neural Network

Uncomment lines 71-74 in the code to add hidden layers:
```python
self.layer1 = nn.Linear(input_dim, 16)
self.relu = nn.ReLU()
self.layer2 = nn.Linear(16, output_dim)
```

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

Created as a demonstration of PyTorch fundamentals and Machine Learning workflow.

## 🤝 Contributing

Feel free to fork, modify, and improve this project. Suggestions and pull requests are welcome!

## 📧 Support

If you encounter any issues or have questions, please open an issue in the repository.

---

**Happy Predicting! 🎉**

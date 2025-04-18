# Password Strength Classifier

A machine learning-based password strength classification system that analyzes passwords and provides detailed security insights.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a sophisticated password strength classifier using PyTorch and Streamlit. It analyzes various password features and provides:
- Password strength classification (Weak/Good/Strong)
- Estimated time to crack under different attack scenarios
- Detailed feature analysis
- Security recommendations

## Features

- **Machine Learning Model**: Uses a neural network to classify password strength
- **Feature Analysis**: Extracts 13 different password features including:
  - Character composition (uppercase, lowercase, numbers, special characters)
  - Entropy calculation
  - Pattern detection (keyboard patterns, English words)
  - Linguistic analysis (syllables, NER detection)
- **Security Metrics**:
  - Estimated crack time for different attack methods
  - Entropy calculation
  - Confidence scores for each strength category
- **User Interface**:
  - Modern, responsive web interface
  - Real-time password analysis
  - Visual strength indicators
  - Security tips and recommendations

## Technical Stack

- **Backend**:
  - Python 3.x
  - PyTorch (for neural network)
  - Pandas (for data handling)
  - NLTK (for linguistic analysis)
  - spaCy (for NER detection)

- **Frontend**:
  - Streamlit
  - Custom CSS for enhanced UI

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── model.py              # Neural network model definition
├── feature_extractor.py  # Password feature extraction
├── loader.py            # Data loading and preprocessing
├── create_balanced.py   # Dataset creation and balancing
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SreesanthJPN/Password-Strength-Analyzer.git
cd Password-Strength-Analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('words')
nltk.download('cmudict')
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open the web interface in your browser
2. Enter a password in the input field
3. View the analysis results including:
   - Password strength classification
   - Estimated crack time
   - Feature analysis
   - Security recommendations

## Model Architecture

The neural network consists of:
- Input layer (13 features)
- 4 hidden layers (32, 128, 64, 64 neurons)
- Output layer (3 classes: Weak, Good, Strong)
- ReLU activation functions
- Softmax output layer

## Security Features

The system analyzes passwords based on:
- Length and character composition
- Entropy and randomness
- Common patterns and sequences
- Dictionary word presence
- Keyboard patterns
- Named entity recognition

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NLTK for linguistic analysis
- spaCy for NER detection
- Streamlit for the web interface

## Contact

For any queries or suggestions, please feel free to reach out or create an issue in the [GitHub repository](https://github.com/SreesanthJPN/Password-Strength-Analyzer). 

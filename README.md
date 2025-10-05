# Handwritten Digit Recognition using a Neural Network

This project implements a neural network to recognize handwritten digits (0-9). I used Streamlit to display the model's results and allow users to test the model on random images from an unseen dataset.

![Streamlit Interface](screenshot.png)

## Datasets

Datasets obtained from [GeeksforGeeks](https://www.geeksforgeeks.org/).

## Installation

1. Clone Repository:
```bash
git clone https://github.com/mdesan/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
```

2. Install Dependencies:
```bash
pip install -r requirements.txt
```

3. Run
```bash
streamlit run Gui.py
```
to open GUI in browser.

**Note**: The model is pre-trained and ready to use. To retrain model (optional), run Main.py. This overwrites existing model files:


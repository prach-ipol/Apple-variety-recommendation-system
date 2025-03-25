# Apple Farming Recommendation System

A web-based recommendation system for apple farming that provides personalized recommendations based on environmental conditions and farming parameters.

## Features

- User-friendly web interface for inputting farming conditions
- Real-time recommendations based on historical data
- Similarity-based recommendation algorithm
- MongoDB integration for data storage
- Display of recent user inputs and their recommendations
- Statistical analysis of the dataset

## Prerequisites

- Python 3.x
- MongoDB
- Required Python packages:
  - Flask
  - pandas
  - numpy
  - scikit-learn
  - pymongo

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Recommendation-System.git
cd Recommendation-System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure MongoDB is running on your system

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application file
- `templates/index.html`: HTML template for the web interface
- `dataset1.csv`: Dataset containing apple farming information
- `requirements.txt`: List of Python package dependencies

## Usage

1. Enter your farming conditions in the input form:
   - Plant Age
   - Temperature
   - Humidity
   - Rain Level
   - Soil pH
   - Expected Yield

2. Click "Get Recommendations" to receive personalized farming recommendations

3. View similar cases and their success rates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
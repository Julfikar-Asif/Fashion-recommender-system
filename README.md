# Fashion Recommender System 👗👔

A deep learning-powered fashion recommendation system that suggests similar fashion items based on image similarity. Built with ResNet50 for feature extraction and K-Nearest Neighbors (KNN) for finding similar products.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Model Architecture](#model-architecture)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This Fashion Recommender System uses deep learning to analyze uploaded fashion images and recommend visually similar products. The system extracts deep features from images using a pre-trained ResNet50 model and uses KNN algorithm to find the most similar items from a fashion product database.

**Key Highlights:**
- 🎯 Accurate similarity matching using deep learning features
- 🚀 Fast recommendations using KNN algorithm
- 📱 User-friendly web interface built with Streamlit
- 🏷️ Category-aware recommendations
- 📊 Trained on 44,000+ fashion product images

## ✨ Features

- **Image-based Search**: Upload any fashion product image to find similar items
- **Deep Learning Feature Extraction**: Uses ResNet50 (pre-trained on ImageNet) for robust feature extraction
- **Category Filtering**: Prioritizes recommendations from the same product category
- **Interactive Web UI**: Clean and intuitive Streamlit interface
- **Fast Recommendations**: Optimized KNN search for quick results
- **Scalable Architecture**: Easily extensible to larger datasets

## 🎬 Demo

Upload a fashion image → Get 5 similar product recommendations instantly!

The system:
1. Extracts features from your uploaded image
2. Finds the most similar products using cosine similarity
3. Filters results by product category
4. Displays the top 5 recommendations

## 🛠️ Technology Stack

### Core Technologies
- **Python 3.10**: Programming language
- **TensorFlow 2.12.0**: Deep learning framework
- **ResNet50**: Pre-trained CNN for feature extraction
- **scikit-learn**: Machine learning utilities (KNN, metrics)
- **Streamlit**: Web application framework

### Libraries
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Pillow**: Image processing
- **OpenCV**: Computer vision operations
- **Matplotlib**: Data visualization

## 📊 Dataset

The system uses the **Fashion Product Images Dataset** which contains:
- **44,000+** fashion product images
- **10 attributes** per product (id, gender, category, subcategory, color, season, etc.)
- **Multiple categories**: Apparel, Accessories, Footwear, etc.
- **Product metadata** stored in `styles.csv`

### Data Structure
```
styles.csv columns:
- id: Product ID
- gender: Men/Women/Boys/Girls/Unisex
- masterCategory: Main category (Apparel, Accessories, etc.)
- subCategory: Subcategory (Topwear, Bottomwear, Watches, etc.)
- articleType: Specific type (Shirts, Jeans, Watches, etc.)
- baseColour: Primary color
- season: Season (Summer, Winter, Fall, Spring)
- year: Year of release
- usage: Usage type (Casual, Formal, Ethnic, etc.)
- productDisplayName: Full product name
```

## 📥 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU (optional, but recommended for faster processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Julfikar-Asif/Fashion-recommender-system.git
cd Fashion-recommender-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
1. Download the Fashion Product Images dataset
2. Extract images to an `images/` folder in the project directory
3. Place `styles.csv` in the root directory

### Step 4: Generate Model Files
Run the Jupyter notebook to process data and generate required files:
```bash
jupyter notebook project.ipynb
```

This will create:
- `features.npy`: Pre-computed ResNet50 features
- `images.npy`: Processed image arrays
- `labels.npy`: Product category labels
- `knn_model.pkl`: Trained KNN model

## 🚀 Usage

### Running the Streamlit App

Once you have generated all required files (`.npy` and `.pkl`), start the web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

1. **Upload an Image**: Click "Upload an Image" and select a fashion product image (JPG, JPEG, or PNG)
2. **View Results**: The system will display your uploaded image and 5 similar product recommendations
3. **Explore**: Try different fashion items to see how the system performs across categories

### Running the Notebook

To train the model from scratch or experiment with parameters:

```bash
jupyter notebook project.ipynb
```

The notebook contains:
1. **Data Loading**: Load and preprocess the fashion dataset
2. **Feature Extraction**: Extract features using ResNet50
3. **Model Training**: Train KNN model on extracted features
4. **Evaluation**: Test recommendations and evaluate performance
5. **Visualization**: Analyze results with confusion matrices and plots
6. **Export**: Save model and features for the Streamlit app

## 📁 Project Structure

```
Fashion-recommender-system/
│
├── app.py                  # Streamlit web application
├── project.ipynb           # Jupyter notebook for model training
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version specification
├── styles.csv             # Product metadata
├── README.md              # Project documentation
├── .gitignore            # Git ignore rules
│
├── images/               # Fashion product images (not in repo)
│   ├── 15970.jpg
│   ├── 39386.jpg
│   └── ...
│
└── Generated files (after running notebook):
    ├── features.npy      # Pre-computed image features
    ├── images.npy        # Processed image arrays
    ├── labels.npy        # Product category labels
    └── knn_model.pkl     # Trained KNN model
```

## 🧠 How It Works

### 1. Feature Extraction
- Uses **ResNet50** (pre-trained on ImageNet)
- Removes the classification layer (include_top=False)
- Uses global average pooling to get 2048-dimensional feature vectors
- Each image is represented as a point in 2048-dimensional space

### 2. Similarity Search
- **K-Nearest Neighbors (KNN)** algorithm with cosine similarity
- Finds the k most similar images in feature space
- Category-aware filtering: prioritizes items from the same category

### 3. Recommendation Flow
```
User Image → Resize (224×224) → Preprocess → ResNet50 → Feature Vector (2048-d)
                                                                ↓
                                                         KNN Search
                                                                ↓
                                                    Find 15 Nearest Neighbors
                                                                ↓
                                                    Filter by Same Category
                                                                ↓
                                                    Return Top 5 Results
```

### 4. Category-Aware Filtering
The system:
- Identifies the category of the uploaded image
- Filters recommendations to prefer items from the same category
- Falls back to other categories if not enough matches exist

## 🏗️ Model Architecture

### ResNet50 Feature Extractor
```
Input Image (224×224×3)
    ↓
ResNet50 Convolutional Layers
    ↓
Global Average Pooling
    ↓
Feature Vector (2048-d)
```

### KNN Recommender
- **Algorithm**: K-Nearest Neighbors
- **Metric**: Cosine Similarity
- **K value**: 15 (pool) → filtered to top 5
- **Training**: Fit on 2000+ product feature vectors

## 📈 Performance

- **Feature Extraction**: ~2048 features per image
- **Search Time**: < 1 second per query
- **Accuracy**: High similarity matching within same categories
- **Dataset Size**: Tested on 2000 samples, scalable to full 44K+ dataset

## 🚀 Future Improvements

- [ ] Add text-based search using product descriptions
- [ ] Implement advanced filtering (color, price range, brand)
- [ ] Add user feedback mechanism to improve recommendations
- [ ] Deploy to cloud platform (Heroku, AWS, or Streamlit Cloud)
- [ ] Add product metadata display with recommendations
- [ ] Implement hybrid recommendation (collaborative + content-based)
- [ ] Add multi-modal search (image + text)
- [ ] Optimize for mobile devices
- [ ] Add caching for faster repeated searches
- [ ] Implement A/B testing framework

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- UI/UX improvements
- Additional filtering options
- Performance optimization
- Documentation enhancements
- Bug fixes
- Test coverage

## 📄 License

This project is open source and available for educational and personal use.

## 🙏 Acknowledgments

- **Dataset**: Fashion Product Images Dataset
- **Pre-trained Model**: ResNet50 from TensorFlow/Keras
- **Framework**: Streamlit for the web interface
- **Community**: Thanks to all contributors and users

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the repository owner.

---

**Made with ❤️ for fashion enthusiasts and data science learners**

*Star ⭐ this repository if you find it helpful!*

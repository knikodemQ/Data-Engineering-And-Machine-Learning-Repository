# Project 02: Data Processing - Advanced Data Manipulation and Transformation

## 📊 Overview

This project demonstrates advanced data processing techniques for real-world data science workflows. It focuses on handling complex data formats, implementing custom business logic, and creating robust data transformation pipelines.

## 🎯 Learning Objectives

- Master automatic CSV delimiter detection and parsing
- Implement custom scaling and ordinal encoding systems
- Handle advanced categorical data types with proper ordering
- Extract numerical data from mixed text fields using regex
- Create intelligent one-hot encoding strategies
- Build comprehensive data processing pipelines
- Implement data quality assessment and validation

## 📁 Project Structure

```
project02_data_processing/
├── data/                                    # Input datasets
│   ├── proj2_data.csv                      # Main dataset with custom formatting
│   └── proj2_scale.txt                     # Custom scaling reference file
├── output/                                 # Generated output files
├── data_processing_pipeline.ipynb         # Main processing notebook
└── README.md                              # This file
```

## 📈 Dataset Description

### Main Dataset (proj2_data.csv)
- **Purpose**: Demonstrates complex data format handling
- **Format**: CSV with custom delimiters and European decimal notation
- **Challenges**: Mixed data types, custom scaling requirements, text-embedded numbers
- **Use Case**: Real-world data processing scenarios

### Scaling Reference (proj2_scale.txt)
- **Purpose**: Defines custom ordinal scaling for categorical variables
- **Format**: Text file with ordered categories
- **Application**: Business rule implementation in data processing

## 🔧 Technical Skills Demonstrated

### 1. Smart CSV Loading
```python
def detect_csv_delimiter(filepath, possible_delimiters=['|', ';', ',', '\t']):
    sniffer = csv.Sniffer()
    sniffer.delimiters = possible_delimiters
    detected_delimiter = sniffer.sniff(sample).delimiter
    return detected_delimiter
```

### 2. Custom Scaling Implementation
- Load business rules from external files
- Apply domain-specific ordinal mappings
- Preserve data integrity during transformation

### 3. Advanced Categorical Handling
```python
# Ordered categorical with business logic
df[col] = pd.Categorical(df[col], categories=sorted_categories, ordered=True)

# Nominal categorical for general text data
df[col] = pd.Categorical(df[col])
```

### 4. Regex-Based Data Extraction
```python
def extract_numbers_from_text(text_value):
    number_pattern = r'[-+]?\d*[,.]?\d+(?:[,.]?\d+)?'
    numbers = re.findall(number_pattern, str(text_value))
    return float(numbers[0].replace(',', '.')) if numbers else None
```

### 5. Intelligent One-Hot Encoding
- Automatic candidate identification
- Exclusion of already-scaled variables
- Batch processing with individual file outputs

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy regex pathlib
```

### Running the Pipeline
1. Place input files in the `data/` directory
2. Open `data_processing_pipeline.ipynb`
3. Execute cells sequentially
4. Review generated outputs in `output/` directory

### Expected Outputs
- `original_data.pkl`: Preserved original dataset
- `custom_scaled_data.pkl`: Business rule scaled data
- `categorical_data.pkl`: Properly typed categorical columns
- `extracted_numeric_data.pkl`: Extracted numerical values
- `one_hot_*.pkl`: Individual one-hot encoded features
- `quality_assessment.json`: Data quality metrics

## 📋 Key Functions and Classes

### `detect_csv_delimiter(filepath, possible_delimiters)`
Automatically detects CSV delimiter using Python's csv.Sniffer.

**Parameters:**
- `filepath`: Path to CSV file
- `possible_delimiters`: List of delimiter candidates

**Returns:** Detected delimiter character

### `apply_custom_scaling(dataframe, scale_mapping)`
Applies business-specific ordinal scaling to appropriate columns.

**Parameters:**
- `dataframe`: Input DataFrame
- `scale_mapping`: Dictionary mapping categories to numerical values

**Returns:** Tuple of (scaled_dataframe, scaled_columns_list)

### `create_categorical_columns(dataframe, scale_mapping)`
Converts appropriate columns to pandas Categorical type with proper ordering.

**Features:**
- Automatic ordered categorical creation for scaled variables
- Nominal categorical for general text data
- Preserves data relationships and enables efficient operations

### `extract_numbers_from_text(text_value)`
Extracts numerical values from mixed text using regex patterns.

**Capabilities:**
- Handles European decimal notation (comma separators)
- Extracts first numerical value found
- Robust error handling for edge cases

### `identify_encoding_candidates(dataframe, scale_mapping, max_categories)`
Intelligently identifies columns suitable for one-hot encoding.

**Criteria:**
- Reasonable number of unique values
- Excludes already-scaled variables
- Text-only content validation
- Sufficient data variation

## 🎓 Learning Outcomes

After completing this project, you will understand:

1. **Advanced Data Loading**
   - Automatic format detection and handling
   - Multi-delimiter CSV processing
   - European vs. American number formats

2. **Business Logic Integration**
   - External configuration file handling
   - Custom scaling system implementation
   - Domain knowledge incorporation

3. **Categorical Data Mastery**
   - Ordered vs. nominal categoricals
   - Memory-efficient categorical storage
   - Statistical operation optimization

4. **Pattern Matching and Extraction**
   - Regular expression design and testing
   - Robust data extraction techniques
   - Error handling for malformed data

5. **Encoding Strategy Design**
   - Smart feature selection for encoding
   - Avoiding redundant transformations
   - Modular encoding pipeline design

6. **Data Quality Assurance**
   - Systematic quality metric calculation
   - Data retention tracking
   - Memory usage optimization

## 🔍 Advanced Features

### Pipeline Architecture
The project implements a modular pipeline design:
- **Input Validation**: Automatic format detection
- **Transformation Modules**: Independent processing steps
- **Quality Gates**: Validation at each stage
- **Output Management**: Structured result storage

### Error Handling
Comprehensive error handling for:
- Missing input files
- Malformed data
- Type conversion failures
- Memory constraints

### Performance Optimization
- Lazy evaluation where possible
- Memory-efficient categorical types
- Vectorized operations
- Minimal data copying

## 🔄 Extensions and Improvements

Potential enhancements for advanced learning:

### Data Engineering Extensions
- Add support for streaming data processing
- Implement parallel processing for large datasets
- Create configuration-driven pipeline definitions
- Add data lineage tracking

### Machine Learning Integration
- Feature importance-based encoding selection
- Automated feature engineering suggestions
- Statistical significance testing for transformations
- Cross-validation for transformation parameters

### Production Readiness
- Add comprehensive logging and monitoring
- Implement data drift detection
- Create automated testing frameworks
- Add performance benchmarking

## 📊 Performance Metrics

The pipeline includes built-in quality assessment:

- **Data Completeness**: Percentage of non-null values
- **Data Retention**: Percentage of original rows preserved
- **Memory Efficiency**: Memory usage optimization tracking
- **Processing Speed**: Transformation time measurements

## 🤝 Contributing

Extend this project by:
- Adding new data format support (XML, Parquet, etc.)
- Implementing additional scaling methods
- Creating visualization modules for quality metrics
- Adding automated parameter tuning
- Enhancing error recovery mechanisms

## 📚 Related Concepts

- **ETL/ELT Pipelines**: Data engineering best practices
- **Feature Engineering**: Advanced transformation techniques
- **Data Quality Management**: Systematic quality assurance
- **Schema Evolution**: Handling changing data structures
- **Performance Optimization**: Efficient data processing

---

**Previous**: [Project 01 - Pandas Basics](../project01_pandas_basics/README.md)  
**Next**: [Project 03 - Data Cleaning](../project03_data_cleaning/README.md)

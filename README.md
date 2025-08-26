# Anomaly Detection Dashboard

A comprehensive Streamlit-based dashboard for real-time anomaly detection across multiple application gateways.

## Features

### 🔍 **SubDomain Management**
- **RelAPG**: Reliability Application Gateway monitoring
- **IdentityAPG**: Identity management and authentication tracking  
- **PreferenceAPG**: User preference and settings analysis

### 📊 **Interactive Topological Graphs**
- Real-time 3-node network topology visualization
- Dynamic node status updates
- Interactive node configuration

### 🛠️ **Anomaly Detection Engine**
- Multiple detection algorithms (Threshold, Statistical, ML-based)
- Configurable feature selection:
  - **ResponseTime**: API response metrics
  - **Client**: Client identification and categorization
  - **ErrorCode**: HTTP status and error code analysis
  - **ErrorMessage**: Error message pattern detection

### 📈 **Advanced Analytics**
- Real-time anomaly counting
- Confidence scoring
- Feature-level anomaly breakdown
- Threshold-based alerting

### 📄 **Comprehensive Reporting**
- PDF report generation with detailed analysis
- Export functionality for all results
- Historical data tracking
- Customizable report templates

### 🔄 **Flow Control**
- Sequential node processing
- Continue/Abort workflow management
- State persistence across sessions

## Installation

1. **Clone or download the application files**

2. **Install required dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application**:
```bash
streamlit run anomaly_dashboard.py
```

## Usage

### Getting Started

1. **Select SubDomain**: Choose from RelAPG, IdentityAPG, or PreferenceAPG
2. **View Topology**: Interactive graph displays with 3 configurable nodes
3. **Configure Nodes**: Click nodes to set up anomaly detection parameters
4. **Set Features**: Select from ResponseTime, Client, ErrorCode, ErrorMessage
5. **Define Thresholds**: Choose sensitivity levels (0.01, 0.05, 0.10, 0.20)
6. **Trigger Analysis**: Run anomaly detection on configured parameters
7. **Review Results**: View real-time anomaly counts and detailed analysis
8. **Generate Reports**: Download PDF reports for documentation
9. **Control Flow**: Continue to next node or abort the analysis workflow

### Advanced Configuration

- **Algorithm Selection**: Choose detection method based on use case
- **Threshold Tuning**: Adjust sensitivity for different environments  
- **Custom Features**: Extend feature set for specific monitoring needs
- **Report Customization**: Modify PDF templates and export formats

## Architecture

```
anomaly_dashboard.py     # Main Streamlit application
backend_utils.py        # Enhanced anomaly detection engine
config.py              # Configuration and settings
requirements.txt       # Python dependencies
```

### Key Components

- **AnomalyDetectionEngine**: Core detection algorithms and feature generation
- **GraphTopologyManager**: Network topology management and visualization
- **ReportGenerator**: PDF generation and export functionality
- **DashboardConfig**: Centralized configuration management

## Configuration Options

### SubDomains
Each subdomain has unique characteristics:
- **Topology**: Different node connection patterns
- **Default Features**: Recommended monitoring parameters  
- **Color Schemes**: Visual differentiation in graphs

### Detection Algorithms
- **Threshold**: Simple threshold-based anomaly detection
- **Statistical**: Z-score based statistical analysis
- **ML-based**: Machine learning approach simulation

### Customization
- Modify `config.py` for custom subdomains and features
- Extend `backend_utils.py` for new detection algorithms
- Update UI themes in main dashboard configuration

## Technical Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Plotly 5.15.0+
- NetworkX 3.1+
- ReportLab 4.0.4+
- Pandas 2.0.0+
- NumPy 1.24.0+

## License

This project is designed for enterprise anomaly detection and monitoring use cases.

## Support

For technical issues or feature requests, please refer to the documentation or contact the development team.

# OMR Answer Evaluation System

A complete Python-based system for scanning, validating, and automatically evaluating OMR (Optical Mark Recognition) sheets.

## Features

✅ Automated OMR sheet scanning and validation  
✅ Answer extraction using OpenCV image processing  
✅ Answer key management  
✅ Detailed evaluation and scoring  
✅ Result visualization with charts  
✅ Batch result export to CSV  
✅ Modern web-based interface  
✅ SQLite database for persistent storage  

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd omr_system

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application

```bash
python app.py
```

Access the application at: **http://localhost:5000**

### 3. Using the System

#### Create Answer Key
1. Go to "Answer Key Management"
2. Enter key name (e.g., "Test_Jan_2025")
3. Enter answers as comma-separated numbers:
   - 0 = Option A
   - 1 = Option B
   - 2 = Option C
   - 3 = Option D
4. Click "Create Answer Key"

**Example:** `0,1,2,3,0,1,2,3,1,2`

#### Process OMR Sheet
1. Select the answer key
2. Enter student roll number and name
3. Upload scanned OMR sheet (JPG/PNG)
4. Click "Process OMR Sheet"
5. View results with score and breakdown

#### Export Results
Click "Export to CSV" to download all evaluation results.

## OMR Sheet Requirements

### Physical Requirements
- **Size:** Standard A4 (210mm × 297mm)
- **Format:** Grayscale or black & white
- **Resolution:** 300+ DPI
- **Quality:** Clear, no folds or damage

### Sheet Layout
- Clearly marked bubble fields
- 4 options (A, B, C, D) per question
- Sequential question numbering
- Roll number/name section at top

### Marking Guidelines
- Use dark pencil or pen
- Fill bubbles completely (60%+ coverage)
- One mark per question
- Use HB pencil recommended

## Project Structure

```
omr_system/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── omr_system.db              # Database (auto-created)
├── templates/
│   ├── index.html             # Main UI
│   └── base.html              # Base template
├── static/
│   ├── style.css              # Styling
│   └── script.js              # JavaScript
├── uploads/                   # Temporary uploads
├── results/                   # Exported results
└── sample_data/               # Sample files
```

## API Endpoints

### POST `/api/upload`
Upload and process OMR sheet
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@omr.jpg" \
  -F "answer_key=Test1" \
  -F "roll_number=A001" \
  -F "student_name=John Doe"
```

### POST `/api/create-answer-key`
Create answer key
```bash
curl -X POST http://localhost:5000/api/create-answer-key \
  -H "Content-Type: application/json" \
  -d '{"key_name":"Test1","answers":[0,1,2,3,0]}'
```

### GET `/api/answer-keys`
Get all available keys

### GET `/api/results`
Get all evaluation results

### GET `/api/export-results`
Export results to CSV

### GET `/api/statistics`
Get overall statistics

## Configuration

Edit `config.py` to adjust:

```python
# Image processing
THRESHOLD_VALUE = 127
DARKNESS_THRESHOLD = 100

# Bubble detection
BUBBLE_AREA_MIN = 200
BUBBLE_AREA_MAX = 5000
MIN_BUBBLES = 20

# File size
MAX_FILE_SIZE = 10 * 1024 * 1024
```

## Troubleshooting

### "No bubbles detected"
- Ensure OMR sheet is scanned at 300+ DPI
- Check bubble darkness and fill percentage
- Verify bubbles are clearly visible

### "Image clarity poor"
- Re-scan OMR sheet
- Use proper lighting during scanning
- Ensure scanner is clean

### "Incorrect answers extracted"
- Fill bubbles completely
- Mark only one option per question
- Use dark pencil or pen

### Database errors
```bash
# Reset database
rm omr_system.db
python app.py
```

## System Architecture

### Core Components

**OMRProcessor** - Image processing & answer extraction
- load_image() - Validate image
- preprocess_image() - Convert to binary
- detect_bubbles() - Find bubble contours
- extract_answers() - Read marked answers
- validate_sheet() - Quality checks

**AnswerKeyManager** - Answer key operations
- create_key() - Store key
- get_key() - Retrieve key
- list_keys() - List all keys
- delete_key() - Remove key

**EvaluationEngine** - Answer evaluation
- evaluate() - Compare & score
- save_result() - Store result

## Performance Tips

1. **Optimize images before upload**
   - Compress to reasonable size
   - Use 300 DPI (not 600+)

2. **Database indexing**
   - Results are indexed by timestamp
   - Add more indexes as needed

3. **Batch processing**
   - Process multiple sheets sequentially
   - Export results afterward

## Security Notes

- File uploads validated
- SQL injection prevention with parameterized queries
- Filename sanitization with secure_filename()
- File size limits enforced

## Advanced Customization

### Add Authentication
Edit `app.py` to add login decorator

### Custom Bubble Detection
Adjust parameters in `config.py`:
```python
BUBBLE_AREA_MIN = 150  # More sensitive
ASPECT_RATIO_MIN = 0.6
```

### Email Notifications
Add email integration for results

### Analytics Dashboard
Create dashboard for overall performance

## Version History

**v1.0.0** - Initial release
- Complete OMR processing
- Web interface
- Result management
- CSV export

## Support & Documentation

- GitHub: [Your Repository]
- Documentation: See docs/ folder
- Issues: Report on GitHub

## License

MIT License - Free for personal and educational use

## Contributing

Contributions welcome! Please submit pull requests.

## Authors

OMR System Development Team

---

For detailed documentation, see the sample files in `sample_data/` folder.

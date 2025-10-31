# Quick Start Guide

## Windows Users

1. **Extract the folder**
   - Extract `omr_system.zip` to your desired location

2. **Run Installation**
   - Double-click `install.bat`
   - Wait for completion

3. **Start Application**
   - Open Command Prompt in the omr_system folder
   - Run: `venv\Scripts\activate.bat`
   - Run: `python app.py`
   - Open browser: http://localhost:5000

## Linux/Mac Users

1. **Extract the folder**
   ```bash
   unzip omr_system.zip
   cd omr_system
   ```

2. **Run Installation**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

3. **Start Application**
   ```bash
   source venv/bin/activate
   python app.py
   ```
   - Open browser: http://localhost:5000

## First Time Usage

### Step 1: Create Answer Key
1. Navigate to "Answer Key Management"
2. Enter a name (e.g., "Test_Jan_2025")
3. Enter answers as comma-separated numbers
   - Example: `0,1,2,3,0,1,2,3,1,2`
   - 0=A, 1=B, 2=C, 3=D
4. Click "Create Answer Key"

### Step 2: Scan OMR Sheet
1. Use scanner/camera to scan OMR sheet
2. Save as JPG or PNG (300+ DPI recommended)
3. Ensure image is clear and bubbles are visible

### Step 3: Process OMR
1. Select answer key
2. Enter student roll number
3. Enter student name
4. Upload OMR image
5. Click "Process OMR Sheet"

### Step 4: View Results
- Score and breakdown displayed immediately
- Results saved to database
- Export to CSV anytime

## Troubleshooting

### "Python not found"
- Install Python 3.8+ from python.org
- Add Python to PATH during installation

### "No module named flask"
- Re-run installation script
- Or manually: `pip install -r requirements.txt`

### "No bubbles detected"
- Scan OMR at higher resolution (300+ DPI)
- Ensure bubbles are filled clearly
- Check image clarity

### Application won't start
- Check if port 5000 is available
- Use different port: `python app.py --port 8000`

## File Structure

```
omr_system/
├── app.py                 # Main application
├── config.py              # Settings
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── install.sh             # Linux/Mac installer
├── install.bat            # Windows installer
├── templates/
│   ├── index.html
│   └── base.html
├── static/
│   ├── style.css
│   └── script.js
├── uploads/               # Temporary uploads
├── results/               # Exported results
├── sample_data/           # Sample files
└── omr_system.db          # Database (auto-created)
```

## Next Steps

1. Explore the web interface
2. Create multiple answer keys
3. Test with sample OMR sheets
4. Export and analyze results
5. Customize settings in config.py

## Support

For issues or questions:
1. Check README.md
2. Review sample_data/ folder
3. Check browser console for errors
4. Ensure all dependencies installed

---

**Enjoy using OMR Evaluation System!**

# Soccer-Player-Reidentification

### How to Set Up and Run the Code

1. **Clone the Repository**
   ```bash
   git clone https://github.com/vyomeshgala/Soccer-Player-Reidentification.git
   cd Soccer-Player-Reidentification
   ```

2. **Install Dependencies**  
   Make sure you have Python 3.8 or above. 
   ```bash
   pip install opencv-python numpy torch ultralytics deep-sort-realtime
   ```

3. **Download YOLOv8 Model Weights**  
   The `best.pt` model is not included due to GitHub size limits. Download it from the link present in the best.pt text file and place it in the root directory.
   
   Your folder structure should look like:
   ```
   Soccer-Player-Reidentification/
   ├── best.pt 
   ├── main.py
   ├── 15sec_input_720p.zip (unzip this)
   ```

5. **Run the Tracker**
   ```bash
   python main.py
   ```
---

### Dependencies / Environment Requirements

- Python >= 3.8  
- torch  
- ultralytics  
- opencv-python  
- numpy  
- deep-sort-realtime  

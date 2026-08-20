import os

from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for

from road_extractor.pipeline import extract_roads
from road_extractor.config import ExtractionConfig, get_default_weights_path

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit to prevent heavy server load

# Ensure required folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload interface."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Handles image upload and runs the full road extraction pipeline."""
    # 1. Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
        
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
        
    # 2. Save uploaded image locally
    filename = "input_image.png"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    # 3. Setup paths for pipeline
    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'run')
    weights_path = get_default_weights_path()
    weights = str(weights_path) if weights_path.exists() else None
    
    # 4. Run the Pipeline
    # This dynamically generates raw_mask.png, repaired_mask.png, skeleton.png, graph_plot.png
    result = extract_roads(
        image_path=upload_path,
        output_dir=output_dir,
        weights_path=weights, 
        config=ExtractionConfig()
    )
    
    # 5. Pass results back to frontend
    # The frontend will read the images directly from the static/outputs/run/ directory
    return render_template('index.html', 
                           processed=True,
                           nodes=result['nodes'],
                           edges=result['edges'],
                           upload_path=upload_path.replace('\\', '/'),
                           output_dir=output_dir.replace('\\', '/'))

if __name__ == '__main__':
    # Runs locally on http://127.0.0.1:5000
    # threaded=False ensures Keras doesn't crash trying to load weights in a worker thread
    app.run(debug=True, use_reloader=False, threaded=False, port=5000)

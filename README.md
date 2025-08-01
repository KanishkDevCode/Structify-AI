
# Structify AI - Floorplan to 3D Scene Converter

Structify AI is a FastAPI-based backend application that transforms 2D floorplan images into annotated segmentation maps, labeled GeoJSON files, and a 3D `.glb` model. It combines YOLO-based instance segmentation, OpenCV preprocessing, Shapely polygon processing, and trimesh 3D scene generation into a single API endpoint.

## Features

- Detects and segments walls, rooms, and furniture from a floorplan image
- Generates:
  - Annotated detection images
  - Binary class-wise masks
  - Composite overlay image
  - GeoJSON file with per-instance polygon labeling
  - GLB 3D model with extruded rooms and placed furniture
- Adjustable confidence thresholds per class
- Packaged into a downloadable ZIP file

## Models and Assets

Trained YOLOv8 models and 3D GLB files must be placed in the following paths:

## Supported Classes

### Room Classes
Bedroom, Dining Room, Foyer, Kitchen, Living Room, Terrace, attachBedroom, Balcony, garage, lobby, study, toilet, utility, walkin

### Object Classes
Bed, Dining table, Sofa, Wardrobe, commode, door, duct, fridge, sink, stove, tv, washing-machine, etc.

## API Endpoint

### POST /process

Accepts a floorplan image and returns a ZIP archive containing detection images, masks, overlay, GeoJSON, and 3D model.

### Request Parameters (multipart/form-data)

- `image` (required): Floorplan image file (JPEG or PNG)
- Optional form fields for confidence threshold override:
  - `bedroom`, `kitchen`, `living_room`, `toilet`, `bed`, `sofa`, `wardrobe`, `commode`, `door`, `wall`

### Example (using curl)

```bash
curl -X POST http://localhost:8000/process \
  -F image=@floorplan.jpg \
  -F bedroom=0.6 -F bed=0.5 -F wall=0.7 \
  --output structify_output.zip
````

### ZIP Output Includes:

* wall\_detection\_annotated.jpg
* room\_detection\_annotated.jpg
* object\_detection\_annotated.jpg
* composite\_overlay.jpg
* polygons\_output.geojson
* floorplan\_scene\_final.glb

## Installation

Install all dependencies:

```bash
pip install fastapi uvicorn opencv-python shapely geojson trimesh[all] ultralytics
```

## Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Optional: Expose via ngrok

```bash
ngrok http 8000
```

## License

This project is licensed under the MIT License.

## Author

Kanishk Singh

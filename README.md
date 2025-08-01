Sure. Below is a clean, **emoji-free**, single `README.md` script that you can directly copy-paste into your repository:

---

```markdown
# Structify AI - Floorplan to 3D Scene Converter

Structify AI is a FastAPI-based backend application that transforms 2D floorplan images into segmentation maps, labeled GeoJSON polygons, and a 3D `.glb` scene. It integrates deep learning (YOLOv8), image processing (OpenCV), geometry handling (Shapely), and 3D mesh generation (Trimesh).

## Features

- Detects and segments walls, rooms, and furniture using pretrained YOLOv8 models
- Generates:
  - Annotated detection images
  - Class-wise binary masks
  - GeoJSON with labeled polygons
  - Composite segmentation overlay
  - Exported 3D GLB model
- Supports per-class confidence thresholds as input
- Automatically places and scales 3D furniture models based on polygon geometry
- Returns all results in a downloadable ZIP file

## Models Used

- `wall_segmentor.pt`: for wall segmentation
- `image_segmentor.pt`: for rooms and furniture segmentation

These models must be located at:
```

/content/drive/MyDrive/Trained\_Model/

```

## Directory Structure

```

/content/drive/MyDrive/
├── Trained\_Model/
│   ├── wall\_segmentor.pt
│   └── image\_segmentor.pt
├── 3D Models/
│   ├── bed.glb
│   ├── wardrobe.glb
│   ├── commode.glb
│   └── door.glb

````

## Supported Classes

### Room Classes
- Bedroom, Dining Room, Foyer, Kitchen, Living Room, Terrace, attachBedroom, Balcony, garage, lobby, study, toilet, utility, walkin

### Object Classes
- Bed, Dining table, Sofa, Wardrobe, commode, door, duct, fridge, sink, stove, tv, washing-machine, etc.

## API Endpoint

### `POST /process`

Processes a floorplan image and returns a ZIP with:
- wall_detection_annotated.jpg
- room_detection_annotated.jpg
- object_detection_annotated.jpg
- composite_overlay.jpg
- polygons_output.geojson
- floorplan_scene_final.glb

#### Parameters
- `image` (file): required JPEG or PNG floorplan
- Optional float form fields for thresholding:
  - `bedroom`, `kitchen`, `bed`, `sofa`, `wardrobe`, `door`, `wall`, etc.

#### Example curl command
```bash
curl -X POST http://localhost:8000/process \
  -F image=@floorplan.jpg \
  -F bedroom=0.6 -F wall=0.5 -F bed=0.5 --output structify_output.zip
````

## Dependencies

Install required packages:

```bash
pip install fastapi uvicorn opencv-python shapely geojson trimesh[all] ultralytics
```

## Running the App

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

If you want public access:

```bash
ngrok http 8000
```

## License

This project is licensed under the MIT License.

## Author

Kanishk Singh

```

---

Let me know if you want a separate `requirements.txt` or example frontend code to connect with this backend.
```



# Libraries need
!pip install --upgrade pip
!pip install shapely
!pip install trimesh
!pip install ultralytics
!pip install opencv-python
!pip install fastapi
!pip install uvicorn
!pip install geojson
!pip install matplotlib
!pip install ifcopenshell  # Optional: only if you also want IFC export
!pip install mapbox-earcut
!pip install triangle
!pip install manifold3d
!pip install pyngrok

# !pip install -q git+https://github.com/IfcOpenShell/IfcOpenShell.git

from google.colab import drive
drive.mount('/content/drive')

# === Imports ===
import os
import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import Polygon, MultiPolygon, Point, mapping, box
from shapely.ops import unary_union
from shapely import affinity
import geojson, json, shutil
from io import BytesIO
import zipfile
from typing import Optional
import trimesh
from trimesh.creation import extrude_polygon
from trimesh.scene import Scene
from trimesh.visual import ColorVisuals
from trimesh.transformations import scale_matrix, translation_matrix, rotation_matrix
from collections import defaultdict

# === Output folder for results ===
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Classes for detection ===
ROOM_CLASSES = [
    'Bedroom', 'Dining Room', 'Foyer', 'Kitchen', 'Living Room',
    'Terrace', 'Terrace Lounge', 'attachBedroom', 'Balcony',
    'garage', 'lobby', 'study', 'toilet', 'utility', 'walkin'
]

OBJECT_CLASSES = [
    'Bed', 'Dining table', 'Sofa', 'Wardrobe', 'commode', 'door',
    'dress', 'duct', 'fridge', 'kitchen-slab', 'lift', 'sink',
    'stove', 'tv', 'wash', 'washing-machine', 'balcoa'
]

# === Default confidence thresholds ===
CLASS_THRESHOLDS = {
    **{cls: 0.5 for cls in ROOM_CLASSES},
    **{cls: 0.5 for cls in OBJECT_CLASSES},
    'Wall': 0.5
}

# === Furniture supported for 3D placement ===
FURNITURE_FOR_3D = ["Bed", "Wardrobe", "commode", "door"]

# === Paths to 3D models ===
MODEL_PATHS = {
    "bed": "/content/drive/MyDrive/3D Models/bed.glb",
    "wardrobe": "/content/drive/MyDrive/3D Models/wardrobe.glb",
    "commode": "/content/drive/MyDrive/3D Models/commode.glb",
    "door": "/content/drive/MyDrive/3D Models/door.glb"
}

# === Model rotations to align properly ===
ROTATION_MAP = {
    "bed": {'x': 270, 'y': 180, 'z': -90},
    "wardrobe": {'x': 270, 'y': 180, 'z': -90},
    "commode": {'x': 270, 'y': 180, 'z': -90},
    "door": {'x': 270, 'y': 180, 'z': -90}
}

# === Heights for extrusion (in meters) ===
HEIGHT_MAP = {
    "wall": 100.0,
    "bedroom": 3.0,
    "living room": 3.0,
    "terrace": 3.0,
    "surface": 0.15,
    "default": 3.0
}

# Filter boxes & masks by confidence and allowed classes
def filter_boxes(boxes, masks, class_names, valid_classes, class_thresholds):
    keep = [
        i for i, (cls, conf) in enumerate(zip(boxes.cls, boxes.conf))
        if class_names[int(cls)] in valid_classes and conf > class_thresholds.get(class_names[int(cls)], 0.5)
    ]
    boxes.data = boxes.data[keep]
    if masks:
        masks.data = masks.data[keep]
    return boxes, masks

# Save mask to file
def save_filtered_mask(mask_result, path):
    if not mask_result[0].masks or not len(mask_result[0].masks.data):
        print(f"No valid masks for {path}")
        return
    height, width = mask_result[0].masks.data[0].shape
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in mask_result[0].masks.data:
        binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
        combined_mask = np.maximum(combined_mask, binary_mask)

    # Clean mask (morphological close)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(OUTPUT_DIR, path), cleaned_mask)

# Overlay a mask on an image
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask_resized > 0] = color
    return cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)

# Preprocess binary mask (remove noise)
def preprocess_mask(mask):
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return clean

# Read mask safely
def safe_read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return mask

# Convert binary mask → list of Shapely polygons
def mask_to_polygons(mask, min_area=100):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            coords = np.squeeze(cnt)
            if coords.ndim != 2 or coords.shape[0] < 4:
                continue
            poly = Polygon(coords)
            if poly.is_valid:
                polygons.append(poly)
    return polygons

#  Clean & fix invalid polygons
def clean_and_fix_polygon(poly):
    poly = poly.buffer(0)  # fix minor invalidities
    if not poly.is_valid or poly.area < 1.0:
        return None
    if isinstance(poly, Polygon):
        return Polygon(poly.exterior)
    elif isinstance(poly, MultiPolygon):
        exteriors = [Polygon(p.exterior) for p in poly.geoms if p.is_valid and p.area > 1.0]
        if not exteriors:
            return None
        elif len(exteriors) == 1:
            return exteriors[0]
        else:
            return unary_union(exteriors)
    return None

# FastAPI App Setup
app = FastAPI()

# Allow CORS for all origins (useful for testing/frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_image(
    # Input parameters: an uploaded image and optional detection thresholds for each class
    image: UploadFile = File(...),
    bedroom: Optional[float] = Form(None),
    dining_room: Optional[float] = Form(None),
    kitchen: Optional[float] = Form(None),
    living_room: Optional[float] = Form(None),
    toilet: Optional[float] = Form(None),
    bed: Optional[float] = Form(None),
    sofa: Optional[float] = Form(None),
    wardrobe: Optional[float] = Form(None),
    commode: Optional[float] = Form(None),
    door: Optional[float] = Form(None),
    wall: Optional[float] = Form(None),
):
    # Validate image type
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Update class-specific confidence thresholds if provided
    if bedroom is not None: CLASS_THRESHOLDS['Bedroom'] = bedroom
    if dining_room is not None: CLASS_THRESHOLDS['Dining Room'] = dining_room
    if kitchen is not None: CLASS_THRESHOLDS['Kitchen'] = kitchen
    if living_room is not None: CLASS_THRESHOLDS['Living Room'] = living_room
    if toilet is not None: CLASS_THRESHOLDS['toilet'] = toilet
    if bed is not None: CLASS_THRESHOLDS['Bed'] = bed
    if sofa is not None: CLASS_THRESHOLDS['Sofa'] = sofa
    if wardrobe is not None: CLASS_THRESHOLDS['Wardrobe'] = wardrobe
    if commode is not None: CLASS_THRESHOLDS['commode'] = commode
    if door is not None: CLASS_THRESHOLDS['door'] = door
    if wall is not None: CLASS_THRESHOLDS['Wall'] = wall

    # Save uploaded image to disk
    image_path = os.path.join(OUTPUT_DIR, "uploaded_image.jpg")
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Load YOLO models for walls and other elements
    wall_model = YOLO("/content/drive/MyDrive/Trained_Model/wall_segmentor.pt")
    image_model = YOLO("/content/drive/MyDrive/Trained_Model/image_segmentor.pt")

    # Run wall detection on the image
    wall_result = wall_model(image_path, conf=wall if wall else 0.5, iou=0.5)
    wall_img = wall_result[0].plot()  # draw boxes/masks
    wall_annot_path = os.path.join(OUTPUT_DIR, "wall_detection_annotated.jpg")
    cv2.imwrite(wall_annot_path, cv2.cvtColor(wall_img, cv2.COLOR_RGB2BGR))

    # Run room detection (rooms & furniture) and filter results
    room_result_raw = image_model(image_path, conf=0.3, iou=0.5)
    room_boxes, room_masks = room_result_raw[0].boxes, room_result_raw[0].masks
    room_names = image_model.names
    room_boxes, room_masks = filter_boxes(room_boxes, room_masks, room_names, ROOM_CLASSES, CLASS_THRESHOLDS)
    room_result_raw[0].boxes = room_boxes
    if room_masks: room_result_raw[0].masks.data = room_masks.data

    # Save annotated room detection image
    room_annot_path = os.path.join(OUTPUT_DIR, "room_detection_annotated.jpg")
    room_img = room_result_raw[0].plot()
    cv2.imwrite(room_annot_path, cv2.cvtColor(room_img, cv2.COLOR_RGB2BGR))

    # Run object detection (furniture & fixtures) and filter results
    object_result_raw = image_model(image_path, conf=0.3, iou=0.5)
    object_boxes, object_masks = object_result_raw[0].boxes, object_result_raw[0].masks
    object_names = image_model.names
    object_boxes, object_masks = filter_boxes(object_boxes, object_masks, object_names, OBJECT_CLASSES, CLASS_THRESHOLDS)
    object_result_raw[0].boxes = object_boxes
    if object_masks: object_result_raw[0].masks.data = object_masks.data

    # Save annotated object detection image
    object_annot_path = os.path.join(OUTPUT_DIR, "object_detection_annotated.jpg")
    object_img = object_result_raw[0].plot()
    cv2.imwrite(object_annot_path, cv2.cvtColor(object_img, cv2.COLOR_RGB2BGR))

    # Save binary masks of walls, rooms, and objects
    save_filtered_mask(wall_result, "walls.jpg")
    save_filtered_mask(room_result_raw, "rooms.jpg")
    save_filtered_mask(object_result_raw, "objects.jpg")

    # Load original and mask images for overlay
    original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    wall_mask = cv2.imread(os.path.join(OUTPUT_DIR, "walls.jpg"), 0)
    room_mask = cv2.imread(os.path.join(OUTPUT_DIR, "rooms.jpg"), 0)
    object_mask = cv2.imread(os.path.join(OUTPUT_DIR, "objects.jpg"), 0)

    # Overlay masks on original image in different colors
    wall_overlay = overlay_mask(original_img, wall_mask, (255, 0, 0))     # red for walls
    room_overlay = overlay_mask(wall_overlay, room_mask, (0, 255, 0))    # green for rooms
    final_overlay = overlay_mask(room_overlay, object_mask, (0, 0, 255)) # blue for objects

    # Save the composite overlay image
    overlay_path = os.path.join(OUTPUT_DIR, "composite_overlay.jpg")
    cv2.imwrite(overlay_path, cv2.cvtColor(final_overlay, cv2.COLOR_RGB2BGR))

    # --- GeoJSON - Per-instance furniture labeling ---
    geojson_features = []
    detected_labels = set()

    # Walls
    if wall_result[0].masks:
        for cls, mask in zip(wall_result[0].boxes.cls, wall_result[0].masks.data):
            cname = wall_model.names[int(cls)]
            detected_labels.add(cname)
            mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
            polys = mask_to_polygons(preprocess_mask(mask_np))
            for poly in polys:
                poly = clean_and_fix_polygon(poly)
                if poly:
                    geojson_features.append(
                        geojson.Feature(geometry=mapping(poly), properties={"label": cname})
                    )

    # Rooms
    if room_masks:
        for cls, mask in zip(room_boxes.cls, room_masks.data):
            cname = image_model.names[int(cls)]
            detected_labels.add(cname)
            mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
            polys = mask_to_polygons(preprocess_mask(mask_np))
            for poly in polys:
                poly = clean_and_fix_polygon(poly)
                if poly:
                    geojson_features.append(
                        geojson.Feature(geometry=mapping(poly), properties={"label": cname})
                    )

    # Furniture objects (per-instance, per-label)
    if object_masks:
        for cls, mask in zip(object_boxes.cls, object_masks.data):
            cname = image_model.names[int(cls)]
            detected_labels.add(cname)
            mask_np = (mask.cpu().numpy() > 0.5).astype(np.uint8) * 255
            polys = mask_to_polygons(preprocess_mask(mask_np))
            for poly in polys:
                poly = clean_and_fix_polygon(poly)
                if poly:
                    geojson_features.append(
                        geojson.Feature(geometry=mapping(poly), properties={"label": cname})
                    )

    # Fallback: Add dummy points for labels detected but not polygonized
    for label in detected_labels:
        if not any(f.properties["label"] == label for f in geojson_features):
            geojson_features.append(
                geojson.Feature(
                    geometry=mapping(Point(0, 0)),
                    properties={"label": label, "note": "no valid polygon found"}
                )
            )

    feature_collection = geojson.FeatureCollection(geojson_features)
    geojson_path = os.path.join(OUTPUT_DIR, "polygons_output.geojson")
    with open(geojson_path, "w") as f:
        geojson.dump(feature_collection, f, indent=2)

    # --- 3D Scene Generation ---
    with open(geojson_path) as f:
        data = json.load(f)
    all_coords = []
    for feat in data["features"]:
        coords = feat["geometry"]["coordinates"][0]
        all_coords.extend(coords)
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    global_offset = (cx, cy)
    print(f"Global offset: {global_offset}")

    scene = Scene()
    object_counts = defaultdict(int)

    # Custom slab (optional)

    # Slab Size
    slab_width, slab_length, slab_height = 550.0, 450.0, 3.0

    # Slab Center
    slab_center_x = 0.0
    slab_center_y = 0.0

    # Slab Creation
    slab_poly = box(
        slab_center_x - slab_width / 2,
        slab_center_y - slab_length / 2,
        slab_center_x + slab_width / 2,
        slab_center_y + slab_length / 2
    )

    # Color for the Slab and naming
    try:
        slab_mesh = extrude_polygon(slab_poly, height=slab_height)
        slab_mesh.visual = ColorVisuals(slab_mesh, face_colors=[200, 200, 200, 255])
        slab_mesh.metadata = {
            "name": "Custom Slab",
            "width": slab_width, "length": slab_length, "height": slab_height
        }
        scene.add_geometry(slab_mesh)
    except Exception as e:
        print(f"Failed to create custom slab: {e}")

    # Iterate over each feature (polygon) in the GeoJSON data
    for feat in data["features"]:
        # Get the label (e.g., "bed", "table") and normalize it
        label = feat["properties"].get("label", "Unnamed")
        norm_label = label.strip().lower()

        # Get the polygon coordinates and create a Shapely Polygon
        coords = feat["geometry"]["coordinates"][0]
        poly = Polygon(coords)

        # Skip invalid or tiny polygons
        if not poly.is_valid or poly.area < 1.0:
            continue

        # Translate polygon to align with the global offset
        poly = affinity.translate(poly, xoff=-global_offset[0], yoff=-global_offset[1])

        # Get the bounding box and centroid of the polygon
        bbox = poly.bounds
        bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        bbox_center = poly.centroid.coords[0]

        # If the normalized label matches a known 3D model path
        if norm_label in MODEL_PATHS:
            model_path = MODEL_PATHS[norm_label]

            # Check if the model file exists
            if not os.path.exists(model_path):
                print(f"Model missing for {label}: {model_path}")
                continue

            try:
                # Load the 3D model using trimesh
                mesh = trimesh.load(model_path, force="scene")

                # If it's a scene, extract all geometries and combine into a single mesh
                if isinstance(mesh, trimesh.Scene):
                    geometries = [g for g in mesh.dump()]
                    if not geometries:
                        continue
                    mesh = trimesh.util.concatenate(geometries)
                elif isinstance(mesh, list):  # if it's a list of meshes
                    mesh = trimesh.util.concatenate(mesh)

                # Apply any predefined rotations for this object (from ROTATION_MAP)
                rotations = ROTATION_MAP.get(norm_label, {'x': 0, 'y': 0, 'z': 0})
                for axis, deg in rotations.items():
                    if deg != 0:
                        R = rotation_matrix(
                            angle=np.radians(deg),
                            direction={'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis],
                            point=mesh.centroid
                        )
                        mesh.apply_transform(R)

                # Scale the model to fit inside the polygon’s bounding box
                model_bbox = mesh.bounding_box.extents[:2]
                target_bbox = [bbox_size[0], bbox_size[1]]
                if min(model_bbox) <= 0:  # avoid division by zero
                    continue
                scale_x = target_bbox[0] / model_bbox[0]
                scale_y = target_bbox[1] / model_bbox[1]
                scale_factor = min(scale_x, scale_y)
                mesh.apply_transform(scale_matrix(scale_factor))

                # Move model so its lowest Z is at 0
                min_z = mesh.bounds[0][2]
                mesh.apply_translation([0, 0, -min_z])

                # Move the model to the center of the polygon
                mesh.apply_transform(translation_matrix([bbox_center[0], bbox_center[1], 0]))

                # Add the mesh to the scene
                scene.add_geometry(mesh)
                object_counts[label] += 1
            except Exception as e:
                print(f"Failed to add model {label}: {e}")
            continue  # move to next feature

        # If no model exists, clean and fix the polygon geometry
        poly = clean_and_fix_polygon(poly)
        if poly is None or poly.area < 1.0:
            continue

        try:
            # Get height for extrusion (from HEIGHT_MAP or default)
            h = HEIGHT_MAP.get(norm_label, HEIGHT_MAP["default"])

            # Extrude the polygon into a 3D mesh with given height
            mesh = extrude_polygon(poly, height=h, engine='earcut')

            # Optionally, set the color of the mesh
            mesh.visual.face_colors = [200, 200, 200, 255]

            # Add the extruded mesh to the scene
            scene.add_geometry(mesh)
            object_counts[label] += 1
        except Exception as e:
            print(f"Failed to extrude {label}: {e}")

    # Define path for final 3D scene file (.glb)
    OUTPUT_GLB_PATH = os.path.join(OUTPUT_DIR, "floorplan_scene_final.glb")

    # Check if the 3D scene has any geometry; if not, raise an error
    if len(scene.geometry) == 0:
        print("Scene is empty! Nothing to export.")
        raise RuntimeError("Scene is empty! Nothing to export.")

    # Export the 3D scene to a .glb file
    scene.export(OUTPUT_GLB_PATH)

    # === Create ZIP file in memory ===
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(wall_annot_path, arcname="wall_detection_annotated.jpg")
        zipf.write(room_annot_path, arcname="room_detection_annotated.jpg")
        zipf.write(object_annot_path, arcname="object_detection_annotated.jpg")
        zipf.write(overlay_path, arcname="composite_overlay.jpg")
        zipf.write(geojson_path, arcname="polygons_output.geojson")
        zipf.write(OUTPUT_GLB_PATH, arcname="floorplan_scene_final.glb")

    # Reset stream position to start of the buffer
    zip_bytes.seek(0)

    # Return the zip as a downloadable HTTP response
    return StreamingResponse(
        zip_bytes,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=structify_output.zip"}
    )

import uvicorn
from pyngrok import ngrok
import nest_asyncio
!ngrok authtoken 2zP4micfwCjuavLFrrHTpDd84yi_3M4eWDYBZUHB5hXxWQ5se
# Allow async in Colab
nest_asyncio.apply()
# Start ngrok tunnel
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# Run FastAPI server
uvicorn.run(app, host="0.0.0.0", port=8000)

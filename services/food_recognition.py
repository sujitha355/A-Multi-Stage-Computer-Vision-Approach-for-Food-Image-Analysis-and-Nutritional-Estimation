# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import logging
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_DIR = os.path.join(_BACKEND_DIR, "models")


class FoodRecognitionService:
    """
    Per-item AI pipeline for Indian food:

    Image
      -> YOLOv8              : bounding boxes per food item
      -> crop each bbox
      -> EfficientNet-B3     : precise Indian food class per crop
      -> SAM / GrabCut       : tight segmentation mask using bbox
      -> MiDaS on crop       : depth map of the cropped region
      -> Portion engine      : weight from mask + depth
      -> Nutrition calculator: macros from weight
    """

    IMAGE_SIZE = 224

    def __init__(self, classifier_path=None, yolo_path=None):
        if classifier_path is None:
            classifier_path = os.path.join(_MODELS_DIR, "food_classifier_best_40_class_v2.pth")
        if yolo_path is None:
            yolo_path = os.path.join(_MODELS_DIR, "food_yolov8.pt")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.classifier = None
        self.class_names = []
        self.yolo_model = None
        self.sam_model = None
        self.midas_model = None
        self.midas_transform = None

        self.classifier_transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._load_yolo(yolo_path)
        self._load_classifier(classifier_path)
        self._load_sam()
        self._load_midas()

    # --------------------------------------------------------------------------
    # Model Loading
    # --------------------------------------------------------------------------

    def _load_yolo(self, yolo_path):
        try:
            from ultralytics import YOLO
            if yolo_path and os.path.exists(yolo_path):
                path = yolo_path
            else:
                path = os.path.join(_MODELS_DIR, "yolov8n.pt")
                logger.warning("Custom YOLOv8 not found -- using base yolov8n (COCO fallback)")
            self.yolo_model = YOLO(path)
            logger.info(f"YOLOv8 loaded: {path}")
        except Exception as e:
            logger.error(f"YOLOv8 loading failed: {e}")

    def _load_classifier(self, model_path):
        if not os.path.exists(model_path):
            logger.warning(f"EfficientNet classifier not found at {model_path} -- YOLO-only mode")
            return
        try:
            logger.info(f"Loading EfficientNet classifier from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.class_names = checkpoint["class_names"]
            num_classes = len(self.class_names)

            # Support both B0 (old) and B3 (new improved model)
            model_name = checkpoint.get("model_name", "efficientnet_b0")
            if model_name == "efficientnet_b3":
                base = models.efficientnet_b3(weights=None)
            else:
                base = models.efficientnet_b0(weights=None)

            # Update classifier transform image size from checkpoint if available
            img_size = checkpoint.get("image_size", self.IMAGE_SIZE)
            if img_size != self.IMAGE_SIZE:
                self.IMAGE_SIZE = img_size
                self.classifier_transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                logger.info(f"Classifier input size updated to {img_size}x{img_size}")

            base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
            base.load_state_dict(checkpoint["model_state_dict"])
            base.to(self.device)
            base.eval()
            self.classifier = base
            logger.info(f"{model_name} loaded -- {num_classes} classes, val_acc: {checkpoint.get('val_acc', 0):.2f}%")
        except Exception as e:
            logger.error(f"EfficientNet loading failed: {e}")

    def _load_sam(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam_path = os.path.join(_MODELS_DIR, "sam_vit_b_01ec64.pth")
            if os.path.exists(sam_path):
                sam = sam_model_registry["vit_b"](checkpoint=sam_path)
                sam.to(self.device)
                self.sam_model = SamPredictor(sam)
                logger.info("SAM loaded")
            else:
                logger.info("SAM not found -- GrabCut fallback active")
        except ImportError:
            logger.info("segment-anything not installed -- GrabCut fallback active")

    def _load_midas(self):
        try:
            self.midas_model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self.midas_model.to(self.device)
            self.midas_model.eval()
            self.midas_transform = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            ).small_transform
            logger.info("MiDaS loaded")
        except Exception as e:
            logger.warning(f"MiDaS not available: {e}")

    # --------------------------------------------------------------------------
    # Stage 1: YOLO -- detect bounding boxes
    # --------------------------------------------------------------------------

    def _detect_with_yolo(self, image_path: str) -> List[Dict]:
        """
        Returns list of YOLO detections with bbox + class.
        Custom model knows all 40 food classes directly.
        """
        is_coco = not os.path.exists(os.path.join(_MODELS_DIR, "food_yolov8.pt"))
        # COCO fallback filter (only used if custom model missing)
        COCO_FOOD_CLASSES = {
            "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "bowl",
            "hot_dog", "donuts", "hamburger", "french fries", "ice cream",
            "sushi", "steak", "waffle", "pancake"
        }
        detections = []
        try:
            results = self.yolo_model(image_path, conf=0.25, iou=0.45)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id].lower()
                    if is_coco and class_name not in COCO_FOOD_CLASSES:
                        continue
                    detections.append({
                        "food_key": class_name,
                        "food": class_name.replace("_", " ").title(),
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    })
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            detections = detections[:3]
            logger.info(f"YOLO found {len(detections)} food region(s)")
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
        return detections

    # --------------------------------------------------------------------------
    # Stage 2: EfficientNet -- classify cropped bbox region
    # --------------------------------------------------------------------------

    def _classify_crop(self, crop: Image.Image) -> Tuple[str, str, float]:
        """
        Runs EfficientNet-B0 on a PIL crop.
        Returns (food_key, food_name, confidence).
        """
        try:
            tensor = self.classifier_transform(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.classifier(tensor), dim=1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)
            food_key = self.class_names[top_idx.item()]
            return food_key, food_key.replace("_", " ").title(), float(top_prob)
        except Exception as e:
            logger.error(f"EfficientNet inference error: {e}")
            return "", "", 0.0

    # --------------------------------------------------------------------------
    # Stage 3: SAM / GrabCut -- segmentation mask from bbox
    # --------------------------------------------------------------------------

    def _segment_bbox(self, image_bgr: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Returns a binary mask (same size as image_bgr) for the food region inside bbox.
        Uses SAM if available, otherwise GrabCut.
        """
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        try:
            if self.sam_model is not None:
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                self.sam_model.set_image(image_rgb)
                masks, _, _ = self.sam_model.predict(
                    box=np.array([x1, y1, x2, y2]), multimask_output=False
                )
                return masks[0].astype(np.uint8)
            else:
                # GrabCut with bbox as rect
                mask = np.zeros(image_bgr.shape[:2], np.uint8)
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                rect = (x1, y1, x2 - x1, y2 - y1)
                cv2.grabCut(image_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
                return np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            # Fallback: filled bbox rectangle
            mask = np.zeros(image_bgr.shape[:2], np.uint8)
            mask[y1:y2, x1:x2] = 1
            return mask

    def _segment_full_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Otsu threshold fallback when no bbox is available."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return (thresh > 0).astype(np.uint8)

    # --------------------------------------------------------------------------
    # Stage 4: MiDaS -- depth map on crop
    # --------------------------------------------------------------------------

    def _estimate_depth_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Runs MiDaS on a cropped BGR image.
        Returns normalized depth map (0-1) at crop resolution.
        Input is capped at 384px to keep inference fast on CPU.
        """
        try:
            if self.midas_model is not None:
                h, w = crop_bgr.shape[:2]
                # Cap size for speed — MiDaS small works well at 256-384px
                max_side = 384
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)))

                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                batch = self.midas_transform(crop_rgb).to(self.device)

                with torch.inference_mode():
                    pred = self.midas_model(batch)
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1),
                        size=crop_rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()

                d = pred.cpu().numpy().astype(np.float32)
                # Normalize to [0, 1]
                d_min, d_max = d.min(), d.max()
                if d_max - d_min > 1e-6:
                    d = (d - d_min) / (d_max - d_min)
                else:
                    d = np.ones_like(d) * 0.5
                return d
            else:
                gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
                d = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32)
                d_max = d.max()
                return d / (d_max + 1e-8)
        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return np.ones(crop_bgr.shape[:2], dtype=np.float32) * 0.5
            return (d - d.min()) / (d.max() - d.min() + 1e-8)
        except Exception as e:
            logger.error(f"Depth estimation error: {e}")
            return np.ones(crop_bgr.shape[:2], dtype=np.float32) * 0.5

    def _estimate_weight(
        self,
        food_key: str,
        mask: np.ndarray,          # full-image binary mask
        depth_map: np.ndarray,     # depth at crop resolution
        crop_shape: Tuple[int, int],  # (h, w) of crop
        full_shape: Tuple[int, int],  # (h, w) of full image
    ) -> Dict:
        """
        Estimates food weight using:
            area_cm2  = food_pixels * pixel_to_cm^2
            depth_cm  = MiDaS depth scaled to real cm
            volume    = area_cm2 * depth_cm
            weight    = volume * density
        """
        try:
            full_h, full_w = full_shape
            total_px = full_h * full_w

            food_px = int(mask.sum()) if mask.sum() > 0 else int(total_px * 0.35)
            area_ratio = food_px / total_px

            # Assume a standard 25cm plate fills ~60% of frame width
            pixel_to_cm = 25.0 / (0.6 * full_w)
            area_cm2 = food_px * (pixel_to_cm ** 2)

            # Depth from MiDaS — use relative contrast between food and background
            # MiDaS gives relative depth: higher value = closer to camera
            # Food should be closer (higher depth) than the plate/table behind it
            if depth_map is not None and mask.sum() > 0:
                crop_h, crop_w = crop_shape
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (depth_map.shape[1], depth_map.shape[0])
                ) > 0.5
                bg_mask = ~mask_resized

                if mask_resized.sum() > 0:
                    food_depth = float(np.mean(depth_map[mask_resized]))
                    bg_depth = float(np.mean(depth_map[bg_mask])) if bg_mask.sum() > 0 else 0.0
                    # Relative elevation: how much higher the food is vs background
                    # Clamp to [0, 1] — negative means food is behind background (unlikely)
                    relative_elev = max(0.0, food_depth - bg_depth)
                    # Scale relative elevation [0,1] → real height [0.3, 5.0] cm
                    depth_cm = 0.3 + relative_elev * 4.7
                else:
                    depth_cm = self._default_depth(food_key)
            else:
                depth_cm = self._default_depth(food_key)

            volume_cm3 = area_cm2 * depth_cm
            density = self._get_density(food_key)
            raw_weight = volume_cm3 * density
            lo, hi = self._get_weight_bounds(food_key)

            return {
                "weight_grams": round(float(np.clip(raw_weight, lo, hi)), 1),
                "volume_ml": round(volume_cm3, 1),
                "area_ratio": round(area_ratio, 3),
                "depth_cm": round(depth_cm, 2),
                "area_cm2": round(area_cm2, 2),
            }
        except Exception as e:
            logger.error(f"Portion estimation error: {e}")
            return {"weight_grams": 150.0, "volume_ml": 150.0, "area_ratio": 0.3, "depth_cm": 2.5, "area_cm2": 0}

    # --------------------------------------------------------------------------
    # Main pipeline -- per-item processing
    # --------------------------------------------------------------------------

    def analyze_image(self, image_path: str) -> Optional[Dict]:
        """
        Full per-item pipeline:
          1. YOLO          -> bounding boxes
          2. crop bbox     -> PIL crop
          3. EfficientNet  -> precise food class
          4. SAM/GrabCut   -> segmentation mask (on full image using bbox)
          5. MiDaS         -> depth map on crop
          6. Portion engine-> weight per item
        Returns the primary (highest confidence) item with all detections.
        """
        try:
            logger.info(f"Analyzing image: {image_path}")
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logger.error(f"Could not read image: {image_path}")
                return None

            full_h, full_w = image_bgr.shape[:2]
            pil_image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            # Stage 1: YOLO detection
            yolo_dets = self._detect_with_yolo(image_path) if self.yolo_model else []

            processed_items = []

            # Pre-classify full image once — used as fallback/comparison for all crops
            full_img_key, full_img_name, full_img_conf = ("", "", 0.0)
            if self.classifier is not None:
                full_img_key, full_img_name, full_img_conf = self._classify_crop(pil_image)
                logger.info(f"Full-image EfficientNet: {full_img_name} ({full_img_conf:.2f})")

            if yolo_dets:
                for det in yolo_dets:
                    bbox = det["bbox"]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(full_w, x2), min(full_h, y2)

                    # Stage 2: Crop the bbox region
                    crop_bgr = image_bgr[y1:y2, x1:x2]
                    crop_pil = pil_image.crop((x1, y1, x2, y2))

                    # Stage 3: EfficientNet on crop + full image, take best
                    if self.classifier is not None:
                        food_key, food_name, clf_conf = self._classify_crop(crop_pil)
                        # Full image classification wins if more confident
                        if full_img_conf > clf_conf:
                            food_key, food_name, clf_conf = full_img_key, full_img_name, full_img_conf
                        if not food_key:
                            food_key, food_name = det["food_key"], det["food"]
                            clf_conf = 0.0
                        # YOLO contributes localization confidence only, EfficientNet owns the class
                        blended_conf = round(0.25 * det["confidence"] + 0.75 * clf_conf, 3)
                        source = "yolo+efficientnet"
                    else:
                        food_key, food_name = det["food_key"], det["food"]
                        blended_conf = round(det["confidence"], 3)
                        clf_conf = 0.0
                        source = "yolo"

                    logger.debug(f"Item: {food_name} | YOLO={det['confidence']:.2f} | EfficientNet={clf_conf:.2f} | blended={blended_conf:.2f}")

                    # Stage 4: Segmentation mask using bbox on full image
                    mask = self._segment_bbox(image_bgr, [x1, y1, x2, y2])
                    seg_method = "SAM" if self.sam_model else "GrabCut"

                    # Stage 5: MiDaS depth on crop
                    depth_map = self._estimate_depth_crop(crop_bgr)
                    depth_method = "MiDaS" if self.midas_model else "Laplacian"

                    # Stage 6: Portion estimation
                    portion = self._estimate_weight(
                        food_key, mask, depth_map,
                        crop_shape=(y2 - y1, x2 - x1),
                        full_shape=(full_h, full_w)
                    )

                    logger.debug(f"  -> weight: {portion['weight_grams']}g | depth: {portion['depth_cm']}cm | area: {portion['area_cm2']}cm2")

                    processed_items.append({
                        "food": food_name,
                        "food_key": food_key,
                        "confidence": blended_conf,
                        "yolo_conf": round(det["confidence"], 3),
                        "classifier_conf": round(clf_conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "weight_grams": portion["weight_grams"],
                        "volume_ml": portion["volume_ml"],
                        "area_ratio": portion["area_ratio"],
                        "depth_cm": portion["depth_cm"],
                        "source": source,
                        "segmentation": seg_method,
                        "depth_method": depth_method,
                    })

            elif self.classifier is not None:
                # No YOLO detections -- run EfficientNet on full image
                logger.info("YOLO found nothing -- running EfficientNet on full image")
                food_key, food_name, conf = self._classify_crop(pil_image)
                if food_key:
                    mask = self._segment_full_image(image_bgr)
                    depth_map = self._estimate_depth_crop(image_bgr)
                    portion = self._estimate_weight(
                        food_key, mask, depth_map,
                        crop_shape=(full_h, full_w),
                        full_shape=(full_h, full_w)
                    )
                    processed_items.append({
                        "food": food_name,
                        "food_key": food_key,
                        "confidence": round(conf, 3),
                        "yolo_conf": 0.0,
                        "classifier_conf": round(conf, 3),
                        "bbox": None,
                        "weight_grams": portion["weight_grams"],
                        "volume_ml": portion["volume_ml"],
                        "area_ratio": portion["area_ratio"],
                        "depth_cm": portion["depth_cm"],
                        "source": "efficientnet_fullimage",
                        "segmentation": "Otsu",
                        "depth_method": "MiDaS" if self.midas_model else "Laplacian",
                    })

            if not processed_items:
                logger.warning("No food detected")
                return None

            processed_items.sort(key=lambda x: x["confidence"], reverse=True)
            primary = processed_items[0]
            logger.info(f"Result: {primary['food']} | confidence={primary['confidence']:.2f} | weight={primary['weight_grams']}g | items={len(processed_items)}")

            return {
                # Primary item (highest confidence)
                "food_name": primary["food"],
                "food_key": primary["food_key"],
                "weight": primary["weight_grams"],
                "confidence": primary["confidence"],
                "volume": primary["volume_ml"],
                "area_ratio": primary["area_ratio"],
                "depth_cm": primary["depth_cm"],
                # All detected items (for multi-food images)
                "all_detections": processed_items,
                "processing_details": {
                    "detection_method": primary["source"],
                    "segmentation_method": primary["segmentation"],
                    "depth_method": primary["depth_method"],
                    "yolo_active": self.yolo_model is not None,
                    "classifier_active": self.classifier is not None,
                    "items_detected": len(processed_items),
                }
            }

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return None

    # --------------------------------------------------------------------------
    # Legacy entry points (used by food_analysis.py)
    # --------------------------------------------------------------------------

    def detect_food(self, image_path: str) -> Dict:
        """Kept for compatibility -- wraps analyze_image."""
        result = self.analyze_image(image_path)
        if not result:
            return {"food_items": [], "confidence": 0.0, "all_detections": []}
        return {
            "food_items": [d["food"] for d in result["all_detections"]],
            "confidence": result["confidence"],
            "all_detections": result["all_detections"],
            "model_type": result["processing_details"]["detection_method"],
        }

    def segment_food(self, image_path: str, bbox=None) -> Optional[np.ndarray]:
        """Kept for compatibility."""
        image_bgr = cv2.imread(image_path)
        if bbox:
            return self._segment_bbox(image_bgr, bbox)
        return self._segment_full_image(image_bgr)

    def estimate_depth(self, image_path: str) -> Optional[np.ndarray]:
        """Kept for compatibility -- runs MiDaS on full image."""
        return self._estimate_depth_crop(cv2.imread(image_path))

    # --------------------------------------------------------------------------
    # Food property tables
    # --------------------------------------------------------------------------

    def _get_density(self, food_key):
        return {
            # Indian
            "dosa": 0.35, "idli": 0.55, "vada": 0.70,
            "sambar": 1.00, "dal": 1.00, "dal_tadka": 1.00,
            "biryani": 0.55, "rice": 0.60, "fried_rice": 0.58,
            "roti": 0.50, "chapati": 0.50, "naan": 0.60, "paratha": 0.70, "poori": 0.65,
            "paneer": 1.10, "palak_paneer": 0.90, "paneer_butter_masala": 0.88,
            "butter_chicken": 0.85, "chicken_curry": 0.85, "tandoori_chicken": 0.90,
            "chole": 0.95, "samosa": 0.80, "vada_pav": 0.75, "pav_bhaji": 0.90,
            "pani_puri": 0.70, "ariselu": 0.85,
            "gulab_jamun": 1.10, "rasgulla": 1.05, "jalebi": 0.90, "rasmalai": 1.00,
            "lassi": 1.02, "kheer": 1.00,
            # Global
            "pizza": 0.65, "hamburger": 0.75, "hot_dog": 0.80,
            "french_fries": 0.45, "onion_rings": 0.50,
            "donuts": 0.55, "waffle": 0.55, "waffles": 0.55, "pancakes": 0.60,
            "cheesecake": 0.95, "chocolate_cake": 0.80, "ice_cream": 0.90,
            "sushi": 0.85, "ramen": 0.95, "tacos": 0.70,
            "steak": 1.05, "grilled_salmon": 1.00,
            "caesar_salad": 0.40, "spaghetti_bolognese": 0.85,
            "macaroni_and_cheese": 0.80, "popcorn": 0.10,
        }.get(food_key, 0.70)

    def _default_depth(self, food_key):
        return {
            # Indian
            "dosa": 0.3, "idli": 3.0, "vada": 2.5, "sambar": 4.0, "dal": 4.0,
            "biryani": 4.0, "rice": 3.0, "roti": 0.4, "chapati": 0.4,
            "naan": 1.0, "paratha": 0.8, "poori": 0.5,
            "samosa": 4.0, "vada_pav": 5.0, "pav_bhaji": 3.5, "pani_puri": 3.0,
            "gulab_jamun": 3.5, "rasgulla": 4.0, "lassi": 8.0,
            # Global
            "pizza": 1.5, "hamburger": 7.0, "hot_dog": 5.0,
            "french_fries": 3.0, "onion_rings": 3.0,
            "donuts": 3.5, "waffles": 2.0, "pancakes": 1.5,
            "cheesecake": 5.0, "chocolate_cake": 6.0, "ice_cream": 5.0,
            "sushi": 2.5, "ramen": 6.0, "tacos": 6.0,
            "steak": 2.5, "grilled_salmon": 2.0,
            "caesar_salad": 4.0, "spaghetti_bolognese": 4.0,
            "macaroni_and_cheese": 4.0, "popcorn": 5.0,
        }.get(food_key, 2.5)

    def _get_weight_bounds(self, food_key):
        return {
            # Indian
            "dosa": (60, 180), "idli": (25, 80), "vada": (30, 100),
            "sambar": (80, 250), "dal": (80, 250), "dal_tadka": (80, 250),
            "biryani": (150, 450), "rice": (100, 350),
            "roti": (25, 70), "chapati": (25, 70), "naan": (60, 150),
            "paratha": (50, 120), "poori": (30, 80),
            "palak_paneer": (100, 300), "paneer_butter_masala": (100, 300),
            "butter_chicken": (100, 350), "chicken_curry": (100, 350),
            "tandoori_chicken": (80, 300),
            "samosa": (40, 120), "vada_pav": (80, 180), "pav_bhaji": (150, 350),
            "pani_puri": (20, 60), "ariselu": (20, 60),
            "gulab_jamun": (30, 80), "rasgulla": (40, 100),
            "jalebi": (30, 100), "rasmalai": (50, 150), "lassi": (150, 400),
            # Global
            "pizza": (80, 300), "hamburger": (150, 350), "hot_dog": (80, 200),
            "french_fries": (80, 250), "onion_rings": (60, 200),
            "donuts": (40, 120), "waffles": (80, 200), "pancakes": (60, 180),
            "cheesecake": (80, 200), "chocolate_cake": (80, 200), "ice_cream": (80, 250),
            "sushi": (30, 200), "ramen": (200, 500), "tacos": (80, 200),
            "steak": (150, 400), "grilled_salmon": (100, 300),
            "caesar_salad": (100, 300), "spaghetti_bolognese": (150, 400),
            "macaroni_and_cheese": (150, 350), "popcorn": (20, 100),
        }.get(food_key, (50, 400))

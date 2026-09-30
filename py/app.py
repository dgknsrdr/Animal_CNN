from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
import logging
import warnings

import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from Transforms import Get_Transform
from animal_model import AnimalModel


# app.py'nin bir üstünde models ve template klasörleri bulunuyor.
BASE_DIR = Path(__file__).resolve().parent.parent

EFFICIENT_MODEL_PATH = BASE_DIR / "models" / "efficientnet_animals.pth"

# Kendi modelini eklediğinde None yerine dosya yolunu yaz:
BASIC_MODEL_PATH = BASE_DIR / "models" / "basic_animals.pth"

MAX_BYTES = 10 * 1024 * 1024
lock = Lock()
logger = logging.getLogger(__name__)

# Mevcut transform dosyanı kullanıyoruz.
train_transform, test_transform, model1_train_transform = Get_Transform()

basic_test_transform = test_transform
efficient_test_transform = EfficientNet_B0_Weights.DEFAULT.transforms()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Modeller ilk kullanıldığında yüklenip burada saklanır.
    app.state.loaded_models = {}

    yield

    app.state.loaded_models.clear()


app = FastAPI(
    title="Hayvan Tanıma",
    lifespan=lifespan
)


def get_model(model_type):
    # HTML'den gelen seçime göre model ve transform belirlenir.
    if model_type == "transfer_learning":
        model_path = EFFICIENT_MODEL_PATH
        model_name = "EfficientNet-B0"
        transform = efficient_test_transform

    elif model_type == "basic":
        model_path = BASIC_MODEL_PATH
        model_name = "Kendi CNN modelim"
        transform = basic_test_transform

    else:
        raise HTTPException(400, "Geçersiz model seçimi.")

    if model_path is None:
        raise HTTPException(503, "Bu model henüz eklenmedi.")

    if not Path(model_path).is_file():
        raise HTTPException(503, "Model dosyası bulunamadı.")

    if model_type not in app.state.loaded_models:
        try:
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True
            )

            if not isinstance(checkpoint, dict):
                raise ValueError("Model kaydı sözlük biçiminde olmalı.")

            if not {"model_state_dict", "class_names"} <= checkpoint.keys():
                raise ValueError(
                    "Model kaydı model_state_dict ve class_names içermeli."
                )

            class_names = checkpoint["class_names"]

            if (
                not isinstance(class_names, (list, tuple))
                or not class_names
                or not all(isinstance(name, str) for name in class_names)
            ):
                raise ValueError("class_names geçersiz.")

            num_classes = len(class_names)

            if model_type == "transfer_learning":
                model = efficientnet_b0(weights=None)
                model.classifier[1] = nn.Linear(
                    model.classifier[1].in_features,
                    num_classes
                )

            elif model_type == "basic":
                model = AnimalModel()
                model.cnn[-1] = nn.Linear(
                    model.cnn[-1].in_features,
                    num_classes
                )

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True
            )

            model.eval()

            app.state.loaded_models[model_type] = (
                model,
                list(class_names)
            )

        except Exception as exc:
            logger.exception("Model yüklenemedi: %s", model_type)
            raise HTTPException(
                503,
                "Model yüklenemedi. Kayıt ve mimariyi kontrol et; "
                "ayrıntı terminalde."
            ) from exc

    model, class_names = app.state.loaded_models[model_type]

    return model, transform, class_names, model_name


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE_DIR / "template" / "index.html")


@app.get("/models")
def models():
    efficient_available = EFFICIENT_MODEL_PATH.is_file()

    basic_available = (
        BASIC_MODEL_PATH is not None
        and Path(BASIC_MODEL_PATH).is_file()
    )

    return {
        "models": [
            {
                "id": "transfer_learning",
                "name": "EfficientNet-B0",
                "available": efficient_available,
                "reason": (
                    "" if efficient_available
                    else "Model dosyası bulunamadı"
                )
            },
            {
                "id": "basic",
                "name": "Kendi CNN modelim",
                "available": basic_available,
                "reason": (
                    "" if basic_available
                    else "Henüz eklenmedi"
                    if BASIC_MODEL_PATH is None
                    else "Model dosyası bulunamadı"
                )
            }
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": str(app.state.device),
        "loaded_models": list(app.state.loaded_models)
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    model_type: str = Form(...)
):
    try:
        raw = file.file.read(MAX_BYTES + 1)

        if len(raw) > MAX_BYTES:
            raise HTTPException(413, "Fotoğraf en fazla 10 MB olabilir.")

        if not raw:
            raise HTTPException(400, "Boş dosya gönderildi.")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    Image.DecompressionBombWarning
                )

                with Image.open(BytesIO(raw)) as source:
                    if source.width * source.height > 20_000_000:
                        raise HTTPException(
                            413,
                            "Fotoğraf en fazla 20 megapiksel olabilir."
                        )

                    image = ImageOps.exif_transpose(source).convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning
        ):
            raise HTTPException(400, "Geçerli bir fotoğraf yükle.")

        with lock:
            model, transform, class_names, model_name = get_model(
                model_type
            )

            tensor = transform(image).unsqueeze(0).to(app.state.device)

            try:
                model.to(app.state.device)

                with torch.inference_mode():
                    logits = model(tensor)
                    probabilities = logits.softmax(dim=1)[0]

                    values, indexes = probabilities.topk(
                        min(3, len(class_names))
                    )

                    predictions = [
                        {
                            "class_name": class_names[index],
                            "probability": probability
                        }
                        for probability, index in zip(
                            values.cpu().tolist(),
                            indexes.cpu().tolist()
                        )
                    ]

            finally:
                # İki modelin ağırlıklarını aynı anda GPU'da tutma.
                model.to("cpu")

        return {
            "model_type": model_type,
            "model_name": model_name,
            "prediction": predictions[0]["class_name"],
            "top_predictions": predictions
        }

    finally:
        file.file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
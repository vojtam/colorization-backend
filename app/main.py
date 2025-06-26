import io
from pathlib import Path

import torch
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import inference_single
from network import Generator
from PIL import Image

app = FastAPI()

MODEL_PATH = Path("models/G.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Computing with {}!".format(device))

# loading the model
model = Generator().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "colorization.dyn.cloud.e-infra.cz",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/colorize")
async def colorize_image(image_file: UploadFile):
    print("Received file:", image_file.filename)
    input_image_bytes = await image_file.read()
    input_image = Image.open(io.BytesIO(input_image_bytes)).convert("RGB")
    output_image = inference_single(model, input_image, device)

    img_byte_arr = io.BytesIO()

    # 2. Save the PIL image to the buffer in PNG format
    output_image.save(img_byte_arr, format="PNG")

    # 3. Get the contents of the buffer
    img_byte_arr = img_byte_arr.getvalue()

    return Response(content=img_byte_arr, media_type="image/png")

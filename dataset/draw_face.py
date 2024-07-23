from models.face_landmarker.face_landmarker import face_landmarker
import tqdm, os

def draw_face(in_path, out_path, ctx=1024):
    for i in tqdm.tqdm(os.listdir(in_path)):
        face_landmarker(f"{in_path}/{i}", f"{out_path}/{i}", ctx)
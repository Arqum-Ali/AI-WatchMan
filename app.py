from flask import Flask, request, jsonify
from flask_cors import CORS
from model import db, FaceEmbedding
import os, cv2, numpy as np
from insightface.app import FaceAnalysis

# ================================
# STEP 1: Init Flask + DB
# ================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# PostgreSQL connection
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql+psycopg2://postgres:Arqum%40123@localhost:5432/face_recognition"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()

# ================================
# STEP 2: Face Model
# ================================
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_embeddings_from_image(img_path):
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    return [f.normed_embedding.astype('float32') for f in faces]

# ================================
# STEP 3: Routes
# ================================
import re
@app.route('/', methods=['GET'])
def hello():
    return jsonify({
        "message": "Hello, brother! Your API is working."
    })

@app.route("/upload-references", methods=["POST"])
def upload_references():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    added = 0
    for file in files:
        filename = file.filename
        save_path = os.path.join(upload_dir, filename)
        file.save(save_path)

        # Extract only letters before numbers
        base_name = os.path.splitext(filename)[0]  # remove .jpg/.png
        person = re.match(r"[A-Za-z]+", base_name).group(0).lower()

        embeddings = get_embeddings_from_image(save_path)

        for emb in embeddings:
            db.session.add(FaceEmbedding(person=person, embedding=emb.tolist()))
            added += 1

    db.session.commit()
    return jsonify({"message": f"Added {added} embeddings to DB"})


@app.route("/identify-image", methods=["POST"])
def identify_image():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    save_path = os.path.join("queries", filename)
    os.makedirs("queries", exist_ok=True)
    file.save(save_path)

    query_embeddings = get_embeddings_from_image(save_path)
    results = []

    # Load all embeddings from DB
    all_embeddings = FaceEmbedding.query.all()
    print("all_embeddings",all_embeddings)
    for emb in query_embeddings:
        best_name, best_score = "unknown", -1
        for entry in all_embeddings:
            db_emb = np.array(entry.embedding)
            score = float(np.dot(emb, db_emb))  # cosine similarity
            if score > best_score:
                best_score, best_name = score, entry.person

        # ✅ Apply threshold (e.g. 0.6 = 60%)
        if best_score < 0.6:
            best_name = "unknown"

        results.append({"person": best_name, "similarity": best_score})

    print("results",results)
    return jsonify({"filename": filename, "results": results})

@app.route("/load-folder", methods=["POST"])
def load_folder():
    folder_path = r"C:\Users\Arham Ali\arqum model\images_matching"

    if not os.path.exists(folder_path):
        return jsonify({"error": f"Folder {folder_path} not found"}), 404

    count_added = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)
            person = filename.split("_")[0].lower()

            embeddings = get_embeddings_from_image(path)  # should return numpy array(s)
            if embeddings is not None:
                for emb in embeddings:
                    # Directly store NumPy array (PickleType handles it)
                    db.session.add(FaceEmbedding(person=person, embedding=emb))
                    count_added += 1

    db.session.commit()

    return jsonify({"message": f"Loaded {count_added} embeddings from {folder_path}"})


# ================================
# STEP 4: Run
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

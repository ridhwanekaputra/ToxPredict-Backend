from flask import Flask, request, jsonify
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs 
from rdkit import RDLogger 
import numpy as np
import pandas as pd
import base64
import io
import joblib
import os
import random
import time
import warnings

# ==========================================
# MEMBUNGKAM PERINGATAN RDKIT & SKLEARN
# ==========================================
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning) 

try:
    from mordred import Calculator, descriptors as mordred_descriptors
    calc_mordred = Calculator(mordred_descriptors, ignore_3D=True)
except ImportError:
    calc_mordred = None

app = Flask(__name__)
CORS(app)

# ==========================================
# CACHE DATABASE & MODEL MACHINE LEARNING
# ==========================================
DYNAMIC_REFERENCE_DB = {}
LOADED_ML_MODELS = {}

ENDPOINTS_LIST = [
    "skin_sensitization", "skin_irritation", "respiratory_irritation",
    "oral_toxicity", "inhalation_toxicity", "eye_irritation", 
    "dermal_toxicity", "cardiac_toxicity"
]

def load_dynamic_datasets():
    print("--------------------------------------------------")
    print("1. Memuat dataset referensi (.sdf) ke memori...")
    
    for ep in ENDPOINTS_LIST:
        folder_path = os.path.join("models", ep)
        DYNAMIC_REFERENCE_DB[ep] = []
        
        if not os.path.exists(folder_path): 
            continue
            
        sdf_files = [f for f in os.listdir(folder_path) if f.endswith('.sdf')]
        if not sdf_files: 
            continue
            
        for sdf_file in sdf_files:
            sdf_path = os.path.join(folder_path, sdf_file)
            try:
                supplier = Chem.SDMolSupplier(sdf_path)
                valid_mols = [mol for mol in supplier if mol is not None]
                if not valid_mols: 
                    continue
                    
                random.seed(42)
                sample_mols = random.sample(valid_mols, min(50, len(valid_mols)))
                
                for mol in sample_mols:
                    try:
                        smi = Chem.MolToSmiles(mol)
                        props = mol.GetPropsAsDict()
                        tox_val = 0.50 
                        
                        for key, value in props.items():
                            if key.lower() in ['class', 'toxic', 'target', 'label', 'activity', 'hasil', 'toxicity', 'y', 'endpoint', 'value', 'outcome']:
                                try:
                                    val = float(value)
                                    tox_val = max(0.0, min(1.0, val)) 
                                    break
                                except: 
                                    continue
                        
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        DYNAMIC_REFERENCE_DB[ep].append({
                            "smiles": smi, 
                            "tox": tox_val, 
                            "fp": fp, 
                            "source": sdf_file
                        })
                    except: 
                        continue
            except: 
                pass
                
        if len(DYNAMIC_REFERENCE_DB[ep]) > 0:
            print(f"[{ep}] BERHASIL memuat {len(DYNAMIC_REFERENCE_DB[ep])} referensi SDF.")

def preload_ml_models():
    print("--------------------------------------------------")
    print("2. Memuat model Machine Learning (.pkl) ke memori...")
    
    for ep in ENDPOINTS_LIST:
        LOADED_ML_MODELS[ep] = []
        base_dir = os.path.join("models", ep)
        
        if not os.path.exists(base_dir): 
            continue
            
        for file in os.listdir(base_dir):
            if not file.endswith(".pkl"): 
                continue
                
            file_lower = file.lower()
            req_feat = None
            
            if "modred" in file_lower or "mordred" in file_lower:
                print(f"   -> [SKIP] Mengabaikan {file} karena sangat berat.")
                continue
                
            if "maccs" in file_lower: 
                req_feat = "maccs"
            elif "morgan" in file_lower: 
                req_feat = "morgan"
            elif "rdkit" in file_lower: 
                req_feat = "rdkit"

            if req_feat:
                model_path = os.path.join(base_dir, file)
                try:
                    model = joblib.load(model_path)
                    LOADED_ML_MODELS[ep].append({
                        "model": model,
                        "feature": req_feat,
                        "filename": file
                    })
                except: 
                    pass
                
        if len(LOADED_ML_MODELS[ep]) > 0:
            print(f"[{ep}] BERHASIL memuat {len(LOADED_ML_MODELS[ep])} model ML.")
    print("--------------------------------------------------")

load_dynamic_datasets()
preload_ml_models()

# ==========================================
# GERBANG PELINDUNG & EKSTRAKSI FITUR
# ==========================================
def check_applicability_domain(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        return False, "SMILES tidak valid."
        
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    if carbon_count == 0: 
        return False, "Out of Domain: Senyawa anorganik tidak didukung."
        
    mw = Descriptors.MolWt(mol)
    if mw < 30: 
        return False, f"Out of Domain: Molekul terlalu kecil ({round(mw,1)} g/mol)."
        
    return True, "OK"

def smiles_to_base64(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 150))
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            return base64.b64encode(img_io.getvalue()).decode('utf-8')
    except: 
        return None

def extract_features(smiles, desc_type):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: 
        return None

    try:
        if desc_type == "maccs":
            maccs = MACCSkeys.GenMACCSKeys(mol)
            return np.array([list(maccs.ToBitString())], dtype=float)
            
        elif desc_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=float)
            from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            return np.array([arr])
            
        elif desc_type == "rdkit":
            res = [func(mol) if True else 0.0 for name, func in Descriptors.descList]
            arr = np.nan_to_num(np.array(res, dtype=float), nan=0.0)
            return np.array([arr])
    except: 
        return None
        
    return None

def calculate_read_across(target_smiles, endpoint_name):
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not target_mol: 
        return 0.0
        
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
    samples = DYNAMIC_REFERENCE_DB.get(endpoint_name, [])
    if not samples: 
        return 0.50 

    similarities = [(DataStructs.TanimotoSimilarity(target_fp, s["fp"]), s["tox"]) for s in samples]
    if not similarities: 
        return 0.50
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_k = similarities[:3]
    
    total_sim = sum(s for s, _ in top_k)
    if total_sim == 0: 
        return 0.50
        
    weighted_tox = sum(s * t for s, t in top_k) / total_sim
    return max(0.01, min(0.99, weighted_tox))

def predict_with_all_models(smiles):
    results = {}
    features_dict = {
        "maccs": None, 
        "morgan": None, 
        "rdkit": None
    }

    for ep_key in ENDPOINTS_LIST:
        ep_models = LOADED_ML_MODELS.get(ep_key, [])
        
        if not ep_models:
            results[ep_key] = {
                "ml_prob": "-", "read_across": "-", "status": "No Model",
                "consensus": "Model Not Loaded", "algorithms": "Hanya sisa model berat",
                "best_algo": "-", "best_feat": "-"
            }
            continue

        ep_probs = []
        loaded_models_count = len(ep_models)
        
        best_confidence = -1
        best_algo = "-"
        best_feat = "-"

        for item in ep_models:
            model = item["model"]
            req_feat = item["feature"]
            filename = item["filename"].lower()
            
            if features_dict[req_feat] is None:
                features_dict[req_feat] = extract_features(smiles, req_feat)
                
            features_to_use = features_dict[req_feat]
            
            if features_to_use is None: 
                continue

            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(features_to_use)[0][1]
                else:
                    pred_class = model.predict(features_to_use)[0]
                    prob = 1.0 if pred_class == 1 else 0.0
                
                ep_probs.append(prob)
                
                # MENGHITUNG KEYAKINAN MODEL & TIE-BREAKER
                confidence = abs(prob - 0.5)
                is_better = False
                
                if confidence > best_confidence:
                    is_better = True
                elif confidence == best_confidence and best_confidence > 0:
                    if req_feat == "morgan" or "rf" in filename or "randomforest" in filename:
                        is_better = True

                if is_better:
                    best_confidence = confidence
                    
                    if "rf" in filename or "randomforest" in filename: 
                        best_algo = "Random Forest (RF)"
                    elif "svm" in filename or "svc" in filename: 
                        best_algo = "SVM"
                    elif "xgb" in filename or "xgboost" in filename: 
                        best_algo = "XGBoost"
                    elif "mlp" in filename or "nn" in filename: 
                        best_algo = "Neural Network"
                    elif "knn" in filename: 
                        best_algo = "KNN"
                    elif "lgbm" in filename or "lightgbm" in filename:
                        best_algo = "LightGBM"
                    elif "lr" in filename or "logistic" in filename: 
                        best_algo = "Logistic Regression"
                    else: 
                        best_algo = "Machine Learning"

                    if req_feat == "morgan": 
                        best_feat = "Morgan ECFP4 2048"
                    elif req_feat == "maccs": 
                        best_feat = "MACCS Keys 166"
                    elif req_feat == "rdkit": 
                        best_feat = "RDKit 2D"
                    
            except: 
                pass

        if ep_probs:
            # Xa = ML Prob
            avg_prob = np.mean(ep_probs)
            
            # Xb = Read-Across
            calculated_ra = calculate_read_across(smiles, ep_key)
            
            # Fc = Final Consensus = (Xa + Xb) / 2
            final_consensus = (avg_prob + calculated_ra) / 2
            
            # Label Toxic / Non-Toxic ditentukan oleh nilai Final Consensus
            status = "Toxic" if final_consensus >= 0.5 else "Non-Toxic"
            
            # REVISI: Tampilan hasil persentase konsensus tanpa embel-embel jumlah model
            results[ep_key] = {
                "ml_prob": f"{round(avg_prob * 100, 1)}%",
                "read_across": f"{round(calculated_ra * 100, 1)}%", 
                "status": status,
                "consensus": f"{round(final_consensus * 100, 1)}%",
                "algorithms": "Morgan/MACCS/RDKit Combined",
                "best_algo": best_algo,
                "best_feat": best_feat
            }
        else:
            results[ep_key] = {
                "ml_prob": "-", "read_across": "-", "status": "No Model",
                "consensus": "Error", "algorithms": "-",
                "best_algo": "-", "best_feat": "-"
            }
            
    return results

def generate_similar(smiles):
    target_mol = Chem.MolFromSmiles(smiles)
    if not target_mol: 
        return []
    
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
    sim_data = []
    
    for ep, samples in DYNAMIC_REFERENCE_DB.items():
        for sample in samples:
            sim = DataStructs.TanimotoSimilarity(target_fp, sample["fp"])
            sim_data.append((sample["smiles"], round(sim * 100), sample.get("source", "Unknown.sdf")))
            
    if not sim_data: 
        return []
            
    sim_data.sort(key=lambda x: x[1], reverse=True)
    
    unique_sim_data = []
    seen = set()
    for item in sim_data:
        if item[0] not in seen:
            unique_sim_data.append(item)
            seen.add(item[0])
            if len(unique_sim_data) == 3: 
                break
            
    return [{"smiles": i[0], "similarity": i[1], "dataset": i[2]} for i in unique_sim_data]

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    data = request.json
    smiles = data.get('smiles', '').strip()
    if not smiles: 
        return jsonify({"error": "Empty SMILES"}), 400

    is_valid, error_msg = check_applicability_domain(smiles)
    if not is_valid: 
        return jsonify({"error": error_msg}), 400

    preds = predict_with_all_models(smiles)
    mol = Chem.MolFromSmiles(smiles)
    comp_mw = f"{round(Descriptors.MolWt(mol), 2)} g/mol" if mol else "0.00 g/mol"

    waktu = round(time.time() - start_time, 2)
    print(f"\n✅ Prediksi sukses super kilat dalam {waktu} detik!\n")

    return jsonify({
        "smiles": smiles, 
        "name": "-", 
        "mw": comp_mw,
        "image": smiles_to_base64(smiles),
        "predictions": preds,
        "similar": generate_similar(smiles),
        "datasets": [{"label": "Trained from SDF", "type": "green"}],
        "descriptor": "All Features Combined (Grand Consensus)"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
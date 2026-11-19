# ============================================================
# SAPPHIRE BANK - ALL-IN-ONE APPLICATION
# Backend + Frontend in a Single Script
# Run: python sapphire_app.py
# ============================================================

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Request, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import sqlite3
from contextlib import contextmanager
import bcrypt as _bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, Field, EmailStr, validator
from uuid import uuid4
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import numpy as np
import pandas as pd
import csv
from io import BytesIO, StringIO
import random
import time
import sys
import os
import base64
import re
import json
import hashlib
import secrets
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ========== ML IMPORTS ==========
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shap

# ========== CONFIG ==========
SECRET_KEY = "sapphire-super-secret-key-change-in-production-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7
DATABASE = "sapphire_bank.db"

# ========== NEURAL NETWORK MODEL ==========

class FraudNeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super(FraudNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# ========== SCANNER AI (Rule-based) ==========

class ScannerAI:
    def __init__(self):
        self.rules = [
            {"name": "velocity", "direction": "outgoing", "threshold": 3, "score": 0.85},
            {"name": "tier_violation", "direction": "outgoing", "threshold": 1.0, "score": 1.0},
            {"name": "near_tier_limit", "direction": "outgoing", "threshold": 0.7, "score": 0.70},
            {"name": "amount_spike", "direction": "outgoing", "threshold": 5, "score": 0.80},
            {"name": "unusual_hour", "direction": "outgoing", "threshold": (5, 22), "score": 0.60},
            {"name": "new_recipient_large", "direction": "outgoing", "threshold": 3, "score": 0.70},
            {"name": "new_device_large", "direction": "outgoing", "threshold": 3, "score": 0.75},
            {"name": "structuring", "direction": "outgoing", "threshold": (45000, 50000), "score": 0.75},
            {"name": "rapid_same_amount", "direction": "outgoing", "threshold": 2, "score": 0.70},
            {"name": "incoming_velocity", "direction": "incoming", "threshold": 5, "score": 0.85},
            {"name": "incoming_structuring", "direction": "incoming", "threshold": (450000, 500000), "score": 0.85},
            {"name": "mule_pattern", "direction": "incoming", "threshold": 3, "score": 0.90},
            {"name": "many_senders", "direction": "incoming", "threshold": 5, "score": 0.75},
            {"name": "sender_blacklisted", "direction": "incoming", "threshold": 1, "score": 0.95},
            {"name": "new_sender_large", "direction": "incoming", "threshold": (7, 100000), "score": 0.80},
            {"name": "suspicious_purpose", "direction": "incoming", "threshold": ["unknown", "prize", "investment", "gift", "donation"], "score": 0.70},
        ]
    
    def scan(self, transaction: Dict) -> Tuple[float, List[str]]:
        max_score = 0.0
        triggered_rules = []
        direction = transaction.get("direction", "outgoing")
        
        for rule in self.rules:
            if rule["direction"] != direction:
                continue
            try:
                triggered = False
                if rule["name"] == "velocity":
                    triggered = transaction.get("transactions_last_hour", 0) >= rule["threshold"]
                elif rule["name"] == "tier_violation":
                    triggered = transaction.get("tier_limit_ratio", 0) > rule["threshold"]
                elif rule["name"] == "near_tier_limit":
                    triggered = transaction.get("tier_limit_ratio", 0) >= rule["threshold"]
                elif rule["name"] == "amount_spike":
                    triggered = transaction.get("amount_to_avg_ratio", 0) > rule["threshold"]
                elif rule["name"] == "unusual_hour":
                    hour = transaction.get("hour", 12)
                    triggered = hour < rule["threshold"][0] or hour > rule["threshold"][1]
                elif rule["name"] == "new_recipient_large":
                    triggered = transaction.get("is_new_recipient", 0) == 1 and transaction.get("amount_to_avg_ratio", 0) > rule["threshold"]
                elif rule["name"] == "new_device_large":
                    triggered = transaction.get("device_matched", 1) == 0 and transaction.get("amount_to_avg_ratio", 0) > rule["threshold"]
                elif rule["name"] == "structuring":
                    amount = transaction.get("amount", 0)
                    triggered = rule["threshold"][0] < amount < rule["threshold"][1] and transaction.get("transactions_last_hour", 0) > 2
                elif rule["name"] == "rapid_same_amount":
                    triggered = transaction.get("transactions_last_hour", 0) > rule["threshold"] and transaction.get("amount_to_avg_ratio", 0) > 2
                elif rule["name"] == "incoming_velocity":
                    triggered = transaction.get("multiple_deposits_today", 0) > rule["threshold"]
                elif rule["name"] == "incoming_structuring":
                    amount = transaction.get("amount", 0)
                    triggered = rule["threshold"][0] < amount < rule["threshold"][1]
                elif rule["name"] == "mule_pattern":
                    triggered = transaction.get("immediate_forwarding", 0) == 1 and transaction.get("multiple_deposits_today", 0) > rule["threshold"]
                elif rule["name"] == "many_senders":
                    triggered = transaction.get("senders_today", 0) > rule["threshold"]
                elif rule["name"] == "sender_blacklisted":
                    triggered = transaction.get("sender_blacklisted", 0) == 1
                elif rule["name"] == "new_sender_large":
                    triggered = transaction.get("sender_age_days", 30) < rule["threshold"][0] and transaction.get("amount", 0) > rule["threshold"][1]
                elif rule["name"] == "suspicious_purpose":
                    triggered = transaction.get("purpose", "") in rule["threshold"]
                
                if triggered:
                    max_score = max(max_score, rule["score"])
                    triggered_rules.append(rule["name"])
            except Exception as e:
                pass
        
        return max_score, triggered_rules

# ========== FUZZY LOGIC AGGREGATOR ==========

class FuzzyLogicAggregator:
    def __init__(self):
        self.low_threshold = 0.3
        self.medium_threshold = 0.7
        self.weights = {
            "scanner": 0.50,
            "random_forest": 0.20,
            "xgboost": 0.20,
            "neural_network": 0.10
        }
    
    def aggregate(self, scanner_score: float, rf_score: float, xgb_score: float, nn_score: float, direction: str, shap_values: Optional[Dict] = None) -> Dict:
        scores = [scanner_score, rf_score, xgb_score, nn_score]
        weights = [self.weights["scanner"], self.weights["random_forest"], self.weights["xgboost"], self.weights["neural_network"]]
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        rules_triggered = []
        
        if scanner_score > 0.7:
            weighted_score = max(weighted_score, 0.65)
            rules_triggered.append("scanner_ai_high_confidence")
        
        if max(scores) > 0.9:
            weighted_score = max(weighted_score, 0.85)
            rules_triggered.append("high_confidence_fraud")
        
        if scanner_score > 0.7 and (rf_score > 0.5 or xgb_score > 0.5 or nn_score > 0.5):
            weighted_score = max(weighted_score, 0.80)
            rules_triggered.append("scanner_ml_agreement")
        
        if all(s > 0.4 and s < 0.8 for s in scores):
            weighted_score = (weighted_score + 0.5) / 2
            rules_triggered.append("unanimous_suspicious")
        
        if max(scores) - min(scores) > 0.5:
            if not (scanner_score > 0.7 and max(rf_score, xgb_score, nn_score) < 0.3):
                weighted_score = weighted_score * 0.9
                rules_triggered.append("model_disagreement")
        
        if weighted_score >= 0.7:
            classification = "FRAUDULENT"
            action = "BLOCK"
            confidence = "HIGH"
        elif weighted_score >= 0.3:
            classification = "SUSPICIOUS"
            action = "OTP_REQUIRED" if direction == "outgoing" else "BLOCK"
            confidence = "MEDIUM"
        else:
            classification = "SAFE"
            action = "APPROVE"
            confidence = "HIGH"
        
        return {
            "final_score": round(weighted_score, 4),
            "classification": classification,
            "action": action,
            "confidence": confidence,
            "rules_triggered": rules_triggered,
            "model_scores": {
                "scanner": round(scanner_score, 4),
                "random_forest": round(rf_score, 4),
                "xgboost": round(xgb_score, 4),
                "neural_network": round(nn_score, 4)
            }
        }

# ========== AI FRAUD DETECTOR WITH SHAP ==========

class AIFraudDetector:
    def __init__(self):
        self.scanner = ScannerAI()
        self.fuzzy = FuzzyLogicAggregator()
        self.rf_model_out = None
        self.xgb_model_out = None
        self.nn_model_out = None
        self.rf_model_in = None
        self.xgb_model_in = None
        self.nn_model_in = None
        self.scaler_out = None
        self.scaler_in = None
        self.outgoing_features = None
        self.incoming_features = None
        self.is_trained = False
        self.shap_explainer_out = None
        self.shap_explainer_in = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate_training_data(self) -> pd.DataFrame:
        transactions = []
        
        for _ in range(3000):
            is_fraud = random.random() < 0.15
            
            if random.random() < 0.6:
                direction = "outgoing"
                features = {
                    "amount": random.uniform(1000, 500000),
                    "hour": random.randint(0, 23),
                    "day_of_week": random.randint(0, 6),
                    "recipient_age_days": random.randint(0, 730),
                    "user_avg_transaction": random.uniform(1000, 50000),
                    "account_age_days": random.randint(1, 730),
                    "transactions_last_hour": random.randint(0, 15),
                    "distance_from_last_txn": random.uniform(0, 5000),
                    "time_since_last_txn_minutes": random.uniform(0, 720),
                    "device_matched": 1 if random.random() > 0.3 else 0,
                    "location_matched": 1 if random.random() > 0.3 else 0,
                    "amount_to_avg_ratio": random.uniform(0.5, 15),
                    "tier_limit_ratio": random.uniform(0.1, 1.5),
                    "is_new_recipient": 1 if random.random() > 0.7 else 0,
                }
                
                if is_fraud:
                    features["amount"] *= random.uniform(2, 5)
                    features["hour"] = random.choice([2, 3, 4])
                    features["transactions_last_hour"] = random.randint(8, 20)
                    features["device_matched"] = 0
                    features["location_matched"] = 0
                    features["amount_to_avg_ratio"] = random.uniform(5, 20)
                    if features.get("tier_limit_ratio", 0) < 1:
                        features["tier_limit_ratio"] = random.uniform(1, 1.5)
            else:
                direction = "incoming"
                features = {
                    "amount": random.uniform(1000, 1000000),
                    "sender_age_days": random.randint(0, 730),
                    "sender_has_history": 1 if random.random() > 0.3 else 0,
                    "multiple_deposits_today": random.randint(1, 10),
                    "immediate_forwarding": 1 if random.random() > 0.8 else 0,
                    "senders_today": random.randint(1, 15),
                    "sender_blacklisted": 1 if random.random() > 0.9 else 0,
                    "amount_vs_avg_incoming": random.uniform(0.1, 5),
                    "account_age_days": random.randint(1, 730),
                    "hour": random.randint(0, 23),
                    "day_of_week": random.randint(0, 6),
                }
                
                if is_fraud:
                    features["sender_age_days"] = random.randint(0, 7)
                    features["sender_has_history"] = 0
                    features["multiple_deposits_today"] = random.randint(5, 20)
                    features["immediate_forwarding"] = 1
                    features["senders_today"] = random.randint(5, 20)
                    features["sender_blacklisted"] = 1
                    features["amount"] = random.uniform(500000, 2000000)
            
            features["is_fraud"] = 1 if is_fraud else 0
            features["direction"] = direction
            transactions.append(features)
        
        return pd.DataFrame(transactions)
    
    def train(self):
        print("\n🧠 Training AI Fraud Detection Models...")
        
        df = self.generate_training_data()
        
        df_out = df[df["direction"] == "outgoing"]
        self.outgoing_features = [
            "amount", "hour", "day_of_week", "recipient_age_days", 
            "user_avg_transaction", "account_age_days", "transactions_last_hour",
            "distance_from_last_txn", "time_since_last_txn_minutes",
            "device_matched", "location_matched", "amount_to_avg_ratio",
            "tier_limit_ratio", "is_new_recipient"
        ]
        
        if len(df_out) > 0:
            X_out = df_out[self.outgoing_features].values
            y_out = df_out["is_fraud"].astype(int).values
            
            self.scaler_out = StandardScaler()
            X_out_scaled = self.scaler_out.fit_transform(X_out)
            
            X_train, X_test, y_train, y_test = train_test_split(X_out_scaled, y_out, test_size=0.2, random_state=42)
            
            self.rf_model_out = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self.rf_model_out.fit(X_train, y_train)
            
            self.xgb_model_out = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
            self.xgb_model_out.fit(X_train, y_train)
            
            X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            model = FraudNeuralNetwork(len(self.outgoing_features)).to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            X_train_tensor = torch.FloatTensor(X_train_nn).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train_nn).to(self.device)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            for epoch in range(30):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze(-1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            self.nn_model_out = model
            
            try:
                self.shap_explainer_out = shap.TreeExplainer(self.rf_model_out)
                print("✅ SHAP explainer initialized for outgoing transactions")
            except Exception as e:
                print(f"⚠️ SHAP explainer initialization warning: {e}")
        
        df_in = df[df["direction"] == "incoming"]
        self.incoming_features = [
            "amount", "sender_age_days", "sender_has_history", "multiple_deposits_today",
            "immediate_forwarding", "senders_today", "sender_blacklisted",
            "amount_vs_avg_incoming", "account_age_days", "hour", "day_of_week"
        ]
        
        if len(df_in) > 0:
            X_in = df_in[self.incoming_features].values
            y_in = df_in["is_fraud"].astype(int).values
            
            self.scaler_in = StandardScaler()
            X_in_scaled = self.scaler_in.fit_transform(X_in)
            
            X_train, X_test, y_train, y_test = train_test_split(X_in_scaled, y_in, test_size=0.2, random_state=42)
            
            self.rf_model_in = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self.rf_model_in.fit(X_train, y_train)
            
            self.xgb_model_in = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
            self.xgb_model_in.fit(X_train, y_train)
            
            X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            model = FraudNeuralNetwork(len(self.incoming_features)).to(self.device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            X_train_tensor = torch.FloatTensor(X_train_nn).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train_nn).to(self.device)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            for epoch in range(30):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze(-1)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            self.nn_model_in = model
            
            try:
                self.shap_explainer_in = shap.TreeExplainer(self.rf_model_in)
                print("✅ SHAP explainer initialized for incoming transactions")
            except Exception as e:
                print(f"⚠️ SHAP explainer initialization warning: {e}")
        
        self.is_trained = True
        print("✅ AI Fraud Detection Models Trained Successfully!")
    
    def predict_nn(self, model, features_scaled):
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            prediction = model(features_tensor).cpu().numpy()[0][0]
        return float(prediction)
    
    def get_shap_explanation(self, features_scaled, feature_names, explainer) -> Dict:
        if explainer is None:
            return {"error": "SHAP explainer not available"}
        try:
            shap_values = explainer.shap_values(features_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            else:
                shap_values = shap_values
            shap_values_flat = shap_values.flatten()
            
            feature_importance = {}
            for i, name in enumerate(feature_names):
                feature_importance[name] = float(shap_values_flat[i])
            
            sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            top_contributors = sorted_importance[:5]
            
            return {
                "feature_importance": feature_importance,
                "top_contributors": top_contributors,
                "explanation": [
                    {"feature": name, "impact": round(value, 4), "direction": "positive" if value > 0 else "negative"}
                    for name, value in top_contributors
                ]
            }
        except Exception as e:
            return {"error": f"SHAP calculation failed: {str(e)}"}
    
    def predict_outgoing(self, transaction: Dict) -> Dict:
        if not self.is_trained:
            return {
                "final_score": 0.0,
                "classification": "SAFE",
                "action": "APPROVE",
                "confidence": "HIGH",
                "triggered_rules": [],
                "model_scores": {},
                "shap_explanation": None
            }
        
        try:
            features = np.array([[transaction[col] for col in self.outgoing_features]])
            features_scaled = self.scaler_out.transform(features)
            
            scanner_score, triggered_rules = self.scanner.scan(transaction)
            rf_score = float(self.rf_model_out.predict_proba(features_scaled)[0][1])
            xgb_score = float(self.xgb_model_out.predict_proba(features_scaled)[0][1])
            nn_score = self.predict_nn(self.nn_model_out, features_scaled)
            
            shap_explanation = self.get_shap_explanation(features_scaled, self.outgoing_features, self.shap_explainer_out)
            
            result = self.fuzzy.aggregate(scanner_score, rf_score, xgb_score, nn_score, "outgoing", shap_explanation)
            result["triggered_rules"] = triggered_rules
            result["shap_explanation"] = shap_explanation
            return result
            
        except Exception as e:
            return {
                "final_score": 0.0,
                "classification": "SAFE",
                "action": "APPROVE",
                "confidence": "LOW",
                "triggered_rules": [],
                "model_scores": {},
                "error": str(e),
                "shap_explanation": None
            }
    
    def predict_incoming(self, transaction: Dict) -> Dict:
        if not self.is_trained:
            return {
                "final_score": 0.0,
                "classification": "SAFE",
                "action": "APPROVE",
                "confidence": "HIGH",
                "triggered_rules": [],
                "model_scores": {},
                "shap_explanation": None
            }
        
        try:
            features = np.array([[transaction[col] for col in self.incoming_features]])
            features_scaled = self.scaler_in.transform(features)
            
            scanner_score, triggered_rules = self.scanner.scan(transaction)
            rf_score = float(self.rf_model_in.predict_proba(features_scaled)[0][1])
            xgb_score = float(self.xgb_model_in.predict_proba(features_scaled)[0][1])
            nn_score = self.predict_nn(self.nn_model_in, features_scaled)
            
            shap_explanation = self.get_shap_explanation(features_scaled, self.incoming_features, self.shap_explainer_in)
            
            result = self.fuzzy.aggregate(scanner_score, rf_score, xgb_score, nn_score, "incoming", shap_explanation)
            result["triggered_rules"] = triggered_rules
            result["shap_explanation"] = shap_explanation
            return result
            
        except Exception as e:
            return {
                "final_score": 0.0,
                "classification": "SAFE",
                "action": "APPROVE",
                "confidence": "LOW",
                "triggered_rules": [],
                "model_scores": {},
                "error": str(e),
                "shap_explanation": None
            }

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Sapphire Bank API",
    description="Complete Banking App with AI Fraud Detection",
    version="3.0.0"
)

# Initialize AI Detector
ai_detector = AIFraudDetector()
ai_detector.train()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def hash_pin(raw: str) -> str:
    """Hash a PIN/password using bcrypt directly (bypasses passlib's broken
    version-detection against modern bcrypt releases)."""
    return _bcrypt.hashpw(raw.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_pin(raw: str, hashed: str) -> bool:
    try:
        return _bcrypt.checkpw(raw.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


class _PwdContextShim:
    """Drop-in replacement for passlib's pwd_context.hash(), used elsewhere
    in this file, backed by direct bcrypt calls."""
    def hash(self, raw: str) -> str:
        return hash_pin(raw)

    def verify(self, raw: str, hashed: str) -> bool:
        return verify_pin(raw, hashed)


pwd_context = _PwdContextShim()
security = HTTPBearer()

os.makedirs("profile_pics", exist_ok=True)

# ============================================================
# DATABASE
# ============================================================

@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def generate_account_number():
    return str(random.randint(1000000000, 9999999999))

def validate_phone_number(phone: str):
    phone = re.sub(r'\D', '', phone)
    if len(phone) != 11:
        return False, "Phone number must be 11 digits"
    if not phone.startswith('0'):
        return False, "Phone number must start with 0"
    return True, phone

def get_tier_limit(tier: int):
    if tier == 1:
        return 50000
    elif tier == 2:
        return 100000
    else:
        return 999999999

def generate_reference():
    return f"TXN{int(time.time())}{uuid4().hex[:6].upper()}"

def generate_otp() -> str:
    return str(random.randint(100000, 999999))

def get_user_by_id(user_id: str):
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(user) if user else None

def get_user_by_phone(phone: str):
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE phone_number = ?", (phone,)).fetchone()
        return dict(user) if user else None

def get_user_by_username(username: str):
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(user) if user else None

def get_user_by_account(account_number: str):
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE account_number = ?", (account_number,)).fetchone()
        return dict(user) if user else None

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def init_db():
    with get_db() as conn:
        # Users table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                account_number TEXT UNIQUE NOT NULL,
                phone_number TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email_verified INTEGER DEFAULT 0,
                full_name TEXT NOT NULL,
                login_pin TEXT NOT NULL,
                transaction_pin TEXT NOT NULL,
                balance REAL DEFAULT 100000.0,
                is_admin INTEGER DEFAULT 0,
                two_fa_enabled INTEGER DEFAULT 0,
                alert_sms INTEGER DEFAULT 1,
                alert_email INTEGER DEFAULT 0,
                alert_in_app INTEGER DEFAULT 1,
                profile_pic TEXT DEFAULT NULL,
                tier_level INTEGER DEFAULT 1,
                account_type TEXT DEFAULT 'savings',
                account_category TEXT DEFAULT 'personal',
                business_name TEXT DEFAULT NULL,
                address TEXT DEFAULT NULL,
                date_of_birth TEXT DEFAULT NULL,
                occupation TEXT DEFAULT NULL,
                nin TEXT DEFAULT NULL,
                bvn TEXT DEFAULT NULL,
                terms_accepted INTEGER DEFAULT 0,
                theme TEXT DEFAULT 'light',
                hide_balance INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Transactions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                reference TEXT UNIQUE NOT NULL,
                from_user_id TEXT,
                to_user_id TEXT,
                type TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'completed',
                recipient_account TEXT,
                recipient_phone TEXT,
                recipient_name TEXT,
                network TEXT,
                risk_score REAL DEFAULT 0,
                fraud_status TEXT DEFAULT 'safe',
                data_plan TEXT,
                direction TEXT DEFAULT 'outgoing',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ai_models_scores TEXT DEFAULT NULL,
                shap_explanation TEXT DEFAULT NULL,
                triggered_rules TEXT DEFAULT NULL,
                classification TEXT DEFAULT 'SAFE'
            )
        ''')
        
        # Airtime purchases
        conn.execute('''
            CREATE TABLE IF NOT EXISTS airtime_purchases (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                amount REAL NOT NULL,
                network TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Data purchases
        conn.execute('''
            CREATE TABLE IF NOT EXISTS data_purchases (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                amount REAL NOT NULL,
                data_plan TEXT NOT NULL,
                network TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # OTP codes
        conn.execute('''
            CREATE TABLE IF NOT EXISTS otp_codes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                transaction_reference TEXT NOT NULL,
                code TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                verified INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Quarantine
        conn.execute('''
            CREATE TABLE IF NOT EXISTS quarantine (
                id TEXT PRIMARY KEY,
                transaction_reference TEXT UNIQUE NOT NULL,
                transaction_data TEXT NOT NULL,
                user_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                amount REAL NOT NULL,
                recipient_account TEXT,
                recipient_phone TEXT,
                risk_score REAL NOT NULL,
                classification TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'pending',
                reviewed_by TEXT,
                reviewed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ai_models_scores TEXT DEFAULT NULL,
                shap_explanation TEXT DEFAULT NULL,
                triggered_rules TEXT DEFAULT NULL
            )
        ''')
        
        # Trusted devices
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trusted_devices (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                last_login TEXT NOT NULL,
                is_current INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Create admin account if not exists
        admin = conn.execute("SELECT * FROM users WHERE is_admin = 1").fetchone()
        if not admin:
            admin_id = str(uuid4())
            admin_account = "9999999999"
            admin_phone = "08000000000"
            admin_email = "admin@sapphire.com"
            admin_username = "admin"
            admin_name = "SYSTEM ADMIN"
            admin_login_pin = pwd_context.hash("0000")
            admin_transaction_pin = pwd_context.hash("0000")
            conn.execute('''
                INSERT INTO users (id, account_number, phone_number, email, username, full_name, login_pin, transaction_pin, balance, is_admin, tier_level, account_type, terms_accepted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (admin_id, admin_account, admin_phone, admin_email, admin_username, admin_name, admin_login_pin, admin_transaction_pin, 10000000, 1, 3, 'current', 1))
            print("✅ Admin account created: admin / PIN 0000")
        
        # Create dummy users with proper account numbers (matching frontend demo)
        dummy_users = [
            ("08011234567", "john@sapphire.com", "john_doe", "John Doe", "1234", "1234", "savings", "personal", 150000, "1234567890"),
            ("08022345678", "jane@sapphire.com", "jane_smith", "Jane Smith", "1234", "1234", "current", "personal", 200000, "0987654321"),
            ("08033456789", "mike@sapphire.com", "mike_johnson", "Mike Johnson", "1234", "1234", "savings", "business", 100000, "1122334455"),
            ("08044455667", "sarah@sapphire.com", "sarah_lee", "Sarah Lee", "1234", "1234", "current", "personal", 250000, "5566778899"),
            ("08055566778", "james@sapphire.com", "james_bond", "James Bond", "1234", "1234", "savings", "personal", 300000, "6677889900"),
        ]
        
        for phone, email, username, name, login_pin, trans_pin, acc_type, acc_cat, balance, acc_num in dummy_users:
            existing = conn.execute("SELECT * FROM users WHERE phone_number = ? OR email = ? OR username = ?", (phone, email, username)).fetchone()
            if not existing:
                user_id = str(uuid4())
                conn.execute('''
                    INSERT INTO users (id, account_number, phone_number, email, username, full_name, login_pin, transaction_pin, balance, is_admin, account_type, account_category, terms_accepted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, acc_num, phone, email, username, name, 
                      pwd_context.hash(login_pin), 
                      pwd_context.hash(trans_pin), 
                      balance, 0, acc_type, acc_cat, 1))
                print(f"✅ Created user: {name} ({username}) - Account: {acc_num}")
        
        # Create some sample transactions for each user
        for user in conn.execute("SELECT id, account_number, full_name FROM users WHERE is_admin = 0").fetchall():
            user_id = user["id"]
            user_acc = user["account_number"]
            
            # Check if user already has transactions
            existing_txns = conn.execute("SELECT COUNT(*) FROM transactions WHERE from_user_id = ? OR to_user_id = ?", (user_id, user_id)).fetchone()
            if existing_txns[0] == 0:
                # Find another user to send money to
                other_users = conn.execute("SELECT id, account_number, full_name FROM users WHERE id != ? AND is_admin = 0 ORDER BY RANDOM() LIMIT 1", (user_id,)).fetchall()
                if other_users:
                    other = other_users[0]
                    # Create some sample transactions
                    sample_txns = [
                        ("transfer", random.randint(1000, 50000), f"Transfer to {other['full_name']}", other["id"], other["account_number"], "outgoing", "completed"),
                        ("transfer", random.randint(5000, 100000), f"Transfer from {other['full_name']}", other["id"], other["account_number"], "incoming", "completed"),
                        ("airtime", random.randint(100, 2000), "Airtime Purchase", None, "08012345678", "outgoing", "completed"),
                        ("data", random.randint(500, 5000), "Data Purchase", None, "08012345678", "outgoing", "completed"),
                    ]
                    
                    for txn_type, amount, desc, to_id, recipient, direction, status in sample_txns[:3]:
                        ref = generate_reference()
                        conn.execute('''
                            INSERT INTO transactions (id, reference, from_user_id, to_user_id, type, amount, description, recipient_account, recipient_name, direction, status)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (str(uuid4()), ref, user_id if direction == "outgoing" else to_id, 
                              to_id if direction == "outgoing" else user_id, 
                              txn_type, amount, desc, recipient if direction == "outgoing" else user_acc,
                              other["full_name"] if direction == "outgoing" else user["full_name"],
                              direction, status))
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_from ON transactions(from_user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_to ON transactions(to_user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transactions_ref ON transactions(reference)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quarantine_status ON quarantine(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quarantine_user ON quarantine(user_id)")
        
        print("✅ Database initialized successfully")

# Initialize database
init_db()

# ============================================================
# PYDANTIC MODELS
# ============================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    full_name: str
    phone_number: str
    login_pin: str = Field(..., min_length=4, max_length=4)
    transaction_pin: str = Field(..., min_length=4, max_length=4)
    account_type: str = "savings"
    account_category: str = "personal"
    terms_accepted: bool = True
    
    @validator('login_pin', 'transaction_pin')
    def validate_pin(cls, v):
        if not v.isdigit():
            raise ValueError('PIN must contain only digits')
        return v
    
    @validator('phone_number')
    def validate_phone(cls, v):
        valid, msg = validate_phone_number(v)
        if not valid:
            raise ValueError(msg)
        return v

class UserLogin(BaseModel):
    username: str
    login_pin: str = Field(..., min_length=4, max_length=4)
    
    @validator('login_pin')
    def validate_pin(cls, v):
        if not v.isdigit():
            raise ValueError('PIN must contain only digits')
        return v

class TransferRequest(BaseModel):
    recipient_account: str = Field(..., min_length=10, max_length=10)
    amount: float = Field(..., gt=0)
    transaction_pin: str = Field(..., min_length=4, max_length=4)
    description: Optional[str] = ""
    
    @validator('transaction_pin')
    def validate_pin(cls, v):
        if not v.isdigit():
            raise ValueError('PIN must contain only digits')
        return v
    
    @validator('recipient_account')
    def validate_account(cls, v):
        if not v.isdigit():
            raise ValueError('Account number must contain only digits')
        return v

class AirtimeRequest(BaseModel):
    phone_number: str
    amount: float = Field(..., gt=0)
    transaction_pin: str = Field(..., min_length=4, max_length=4)
    network: str
    
    @validator('transaction_pin')
    def validate_pin(cls, v):
        if not v.isdigit():
            raise ValueError('PIN must contain only digits')
        return v
    
    @validator('phone_number')
    def validate_phone(cls, v):
        valid, msg = validate_phone_number(v)
        if not valid:
            raise ValueError(msg)
        return v

class DataRequest(BaseModel):
    phone_number: str
    data_plan: str
    amount: float = Field(..., gt=0)
    transaction_pin: str = Field(..., min_length=4, max_length=4)
    network: str
    
    @validator('transaction_pin')
    def validate_pin(cls, v):
        if not v.isdigit():
            raise ValueError('PIN must contain only digits')
        return v

class TierUpgradeRequest(BaseModel):
    address: Optional[str] = None
    date_of_birth: Optional[str] = None
    occupation: Optional[str] = None
    nin: Optional[str] = None
    bvn: Optional[str] = None
    business_name: Optional[str] = None

class SettingsUpdate(BaseModel):
    theme: Optional[str] = None
    hide_balance: Optional[int] = None

class OTPVerifyRequest(BaseModel):
    transaction_reference: str
    otp: str = Field(..., min_length=6, max_length=6)

class ChatRequest(BaseModel):
    message: str

class ChangePinRequest(BaseModel):
    current_pin: str = Field(..., min_length=4, max_length=4)
    new_pin: str = Field(..., min_length=4, max_length=4)
    
    @validator('current_pin', 'new_pin')
    def validate_pin(cls, v):
        if not v.isdigit():
            raise ValueError('PIN must contain only digits')
        return v

# ============================================================
# API ENDPOINTS
# ============================================================

def _serialize_user_public(user: dict):
    return {
        "id": user["id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "email": user["email"],
        "phone_number": user["phone_number"],
        "account_number": user["account_number"],
        "account_type": user["account_type"],
        "account_category": user["account_category"],
        "tier_level": user["tier_level"],
        "balance": user["balance"],
        "is_admin": bool(user["is_admin"]),
        "theme": user["theme"],
        "hide_balance": bool(user["hide_balance"]),
        "created_at": user["created_at"],
    }


def _build_outgoing_features(conn, user: dict, amount: float, recipient_id: str):
    now = datetime.utcnow()

    # Count of outgoing transactions in the last hour
    one_hour_ago = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    txns_last_hour = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE from_user_id = ? AND created_at >= ?",
        (user["id"], one_hour_ago),
    ).fetchone()[0]

    # Average of user's past outgoing transaction amounts
    avg_row = conn.execute(
        "SELECT AVG(amount) FROM transactions WHERE from_user_id = ? AND status = 'completed'",
        (user["id"],),
    ).fetchone()
    user_avg_transaction = avg_row[0] if avg_row and avg_row[0] else amount

    # Has this user sent money to this recipient before?
    prior_to_recipient = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE from_user_id = ? AND to_user_id = ?",
        (user["id"], recipient_id),
    ).fetchone()[0]
    is_new_recipient = 1 if prior_to_recipient == 0 else 0
    recipient_age_days = 0 if is_new_recipient else 365

    # Time since user's last outgoing transaction
    last_txn = conn.execute(
        "SELECT created_at FROM transactions WHERE from_user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user["id"],),
    ).fetchone()
    if last_txn and last_txn[0]:
        try:
            last_time = datetime.fromisoformat(last_txn[0])
            time_since_last_txn_minutes = max(0.0, (now - last_time).total_seconds() / 60.0)
        except Exception:
            time_since_last_txn_minutes = 999999.0
    else:
        time_since_last_txn_minutes = 999999.0

    created_at = user.get("created_at")
    try:
        account_age_days = max(1, (now - datetime.fromisoformat(created_at)).days) if created_at else 30
    except Exception:
        account_age_days = 30

    tier_limit = get_tier_limit(user["tier_level"])
    amount_to_avg_ratio = amount / user_avg_transaction if user_avg_transaction else 1.0
    tier_limit_ratio = amount / tier_limit if tier_limit else 0.0

    return {
        "amount": float(amount),
        "hour": now.hour,
        "day_of_week": now.weekday(),
        "recipient_age_days": recipient_age_days,
        "user_avg_transaction": float(user_avg_transaction),
        "account_age_days": account_age_days,
        "transactions_last_hour": int(txns_last_hour),
        "distance_from_last_txn": 0.0,
        "time_since_last_txn_minutes": float(time_since_last_txn_minutes),
        "device_matched": 1,
        "location_matched": 1,
        "amount_to_avg_ratio": float(amount_to_avg_ratio),
        "tier_limit_ratio": float(tier_limit_ratio),
        "is_new_recipient": is_new_recipient,
        "direction": "outgoing",
    }


# ---------------- AUTH ----------------

@app.post("/auth/register")
def register(payload: UserCreate):
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ? OR email = ? OR phone_number = ?",
            (payload.username, payload.email, payload.phone_number),
        ).fetchone()
        if existing:
            raise HTTPException(status_code=400, detail="Username, email, or phone number already registered")

        if not payload.terms_accepted:
            raise HTTPException(status_code=400, detail="You must accept the terms and conditions")

        # Generate a unique account number
        account_number = generate_account_number()
        while conn.execute("SELECT id FROM users WHERE account_number = ?", (account_number,)).fetchone():
            account_number = generate_account_number()

        user_id = str(uuid4())
        conn.execute(
            """
            INSERT INTO users (id, account_number, phone_number, email, username, full_name,
                                login_pin, transaction_pin, balance, is_admin, account_type,
                                account_category, terms_accepted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                user_id, account_number, payload.phone_number, payload.email, payload.username,
                payload.full_name, hash_pin(payload.login_pin), hash_pin(payload.transaction_pin),
                50000.0, payload.account_type, payload.account_category, 1 if payload.terms_accepted else 0,
            ),
        )

    return {
        "alert": "Account created successfully! You can now log in.",
        "account_number": account_number,
    }


@app.post("/auth/login")
def login(payload: UserLogin):
    user = get_user_by_username(payload.username)
    if not user:
        user = get_user_by_phone(payload.username)
    if not user:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (payload.username,)).fetchone()
            user = dict(row) if row else None

    if not user or not verify_pin(payload.login_pin, user["login_pin"]):
        raise HTTPException(status_code=401, detail="Invalid username or PIN")

    token = create_access_token({"sub": user["id"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "full_name": user["full_name"],
        "is_admin": bool(user["is_admin"]),
    }


# ---------------- ACCOUNT / PROFILE ----------------

@app.get("/balance")
def get_balance(user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"balance": user["balance"], "tier_level": user["tier_level"]}


@app.get("/profile")
def get_profile(user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _serialize_user_public(user)


@app.get("/account-number")
def get_account_number(user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"account_number": user["account_number"], "bank_name": "Sapphire Bank"}


@app.get("/verify-account/{account_number}")
def verify_account(account_number: str, user_id: str = Depends(get_current_user)):
    recipient = get_user_by_account(account_number)
    if not recipient:
        raise HTTPException(status_code=404, detail="Account not found")
    return {
        "account_name": recipient["full_name"],
        "account_number": recipient["account_number"],
        "bank_name": "Sapphire Bank",
    }


@app.get("/transactions")
def get_transactions(limit: int = 50, user_id: str = Depends(get_current_user)):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM transactions
            WHERE from_user_id = ? OR to_user_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (user_id, user_id, limit),
        ).fetchall()

    transactions = []
    for row in rows:
        t = dict(row)
        is_sender = t["from_user_id"] == user_id
        direction = "sent" if is_sender else "received"
        counterparty_id = t["to_user_id"] if is_sender else t["from_user_id"]
        counterparty = None
        if counterparty_id:
            cp = get_user_by_id(counterparty_id)
            counterparty = cp["full_name"] if cp else t.get("recipient_name")
        else:
            counterparty = t.get("recipient_name") or t.get("recipient_phone")

        transactions.append({
            "reference": t["reference"],
            "type": t["type"],
            "amount": abs(t["amount"]),
            "description": t["description"] or counterparty or "Transaction",
            "counterparty": counterparty,
            "status": t["status"],
            "direction": direction,
            "classification": t["classification"],
            "created_at": t["created_at"],
        })

    return {"transactions": transactions}


# ---------------- TRANSFER / FRAUD PIPELINE ----------------

@app.post("/transfer")
def transfer(payload: TransferRequest, user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_pin(payload.transaction_pin, user["transaction_pin"]):
        raise HTTPException(status_code=401, detail="Incorrect transaction PIN")

    if payload.recipient_account == user["account_number"]:
        raise HTTPException(status_code=400, detail="You cannot transfer to your own account")

    recipient = get_user_by_account(payload.recipient_account)
    if not recipient:
        raise HTTPException(status_code=404, detail="Recipient account not found")

    if payload.amount > user["balance"]:
        raise HTTPException(status_code=400, detail="Insufficient balance")

    tier_limit = get_tier_limit(user["tier_level"])
    if payload.amount > tier_limit:
        raise HTTPException(
            status_code=400,
            detail=f"Amount exceeds your tier {user['tier_level']} transaction limit of NGN {tier_limit:,.2f}. Please upgrade your tier.",
        )

    reference = generate_reference()

    with get_db() as conn:
        features = _build_outgoing_features(conn, user, payload.amount, recipient["id"])
        ai_result = ai_detector.predict_outgoing(features)
        action = ai_result.get("action", "APPROVE")
        classification = ai_result.get("classification", "SAFE")
        final_score = ai_result.get("final_score", 0.0)

        common_fields = dict(
            id=str(uuid4()),
            reference=reference,
            from_user_id=user_id,
            to_user_id=recipient["id"],
            type="transfer",
            amount=payload.amount,
            description=payload.description or f"Transfer to {recipient['full_name']}",
            recipient_account=recipient["account_number"],
            recipient_name=recipient["full_name"],
            risk_score=final_score,
            classification=classification,
            ai_models_scores=json.dumps(ai_result.get("model_scores", {})),
            shap_explanation=json.dumps(ai_result.get("shap_explanation", {}) or {}),
            triggered_rules=json.dumps(ai_result.get("triggered_rules", [])),
        )

        if action == "BLOCK":
            conn.execute(
                """
                INSERT INTO transactions (id, reference, from_user_id, to_user_id, type, amount,
                    description, status, recipient_account, recipient_name, risk_score,
                    fraud_status, direction, ai_models_scores, shap_explanation, triggered_rules, classification)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'blocked', ?, ?, ?, 'blocked', 'outgoing', ?, ?, ?, ?)
                """,
                (
                    common_fields["id"], reference, user_id, recipient["id"], "transfer", payload.amount,
                    common_fields["description"], recipient["account_number"], recipient["full_name"],
                    final_score, common_fields["ai_models_scores"], common_fields["shap_explanation"],
                    common_fields["triggered_rules"], classification,
                ),
            )
            conn.execute(
                """
                INSERT INTO quarantine (id, transaction_reference, transaction_data, user_id, direction,
                    amount, recipient_account, risk_score, classification, reason, status)
                VALUES (?, ?, ?, ?, 'outgoing', ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    str(uuid4()), reference, json.dumps(features), user_id, payload.amount,
                    recipient["account_number"], final_score, classification,
                    "Blocked by AI fraud detection: " + ", ".join(ai_result.get("triggered_rules", []) or ["high risk score"]),
                ),
            )
            return {
                "status": "blocked",
                "alert": "This transaction was blocked by our AI fraud detection system for your protection.",
                "reference": reference,
                "classification": classification,
                "ai_fraud_score": final_score,
            }

        elif action == "OTP_REQUIRED":
            conn.execute(
                """
                INSERT INTO transactions (id, reference, from_user_id, to_user_id, type, amount,
                    description, status, recipient_account, recipient_name, risk_score,
                    fraud_status, direction, ai_models_scores, shap_explanation, triggered_rules, classification)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending_otp', ?, ?, ?, 'suspicious', 'outgoing', ?, ?, ?, ?)
                """,
                (
                    common_fields["id"], reference, user_id, recipient["id"], "transfer", payload.amount,
                    common_fields["description"], recipient["account_number"], recipient["full_name"],
                    final_score, common_fields["ai_models_scores"], common_fields["shap_explanation"],
                    common_fields["triggered_rules"], classification,
                ),
            )
            otp_code = generate_otp()
            conn.execute(
                """
                INSERT INTO otp_codes (id, user_id, transaction_reference, code, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()), user_id, reference, otp_code,
                    (datetime.utcnow() + timedelta(minutes=10)).isoformat(),
                ),
            )
            return {
                "status": "otp_required",
                "alert": "This transaction looks unusual. Please verify with the OTP sent to you to continue.",
                "reference": reference,
                "classification": classification,
                "ai_fraud_score": final_score,
                "demo_otp": otp_code,  # exposed for demo purposes only (no real SMS gateway)
            }

        else:  # APPROVE
            conn.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (payload.amount, user_id))
            conn.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (payload.amount, recipient["id"]))
            conn.execute(
                """
                INSERT INTO transactions (id, reference, from_user_id, to_user_id, type, amount,
                    description, status, recipient_account, recipient_name, risk_score,
                    fraud_status, direction, ai_models_scores, shap_explanation, triggered_rules, classification)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, 'safe', 'outgoing', ?, ?, ?, ?)
                """,
                (
                    common_fields["id"], reference, user_id, recipient["id"], "transfer", payload.amount,
                    common_fields["description"], recipient["account_number"], recipient["full_name"],
                    final_score, common_fields["ai_models_scores"], common_fields["shap_explanation"],
                    common_fields["triggered_rules"], classification,
                ),
            )
            return {
                "status": "completed",
                "alert": f"Transfer of NGN {payload.amount:,.2f} to {recipient['full_name']} was successful.",
                "reference": reference,
                "classification": classification,
                "ai_fraud_score": final_score,
            }


@app.post("/transaction/verify-otp")
def verify_otp(payload: OTPVerifyRequest, user_id: str = Depends(get_current_user)):
    with get_db() as conn:
        otp_row = conn.execute(
            """
            SELECT * FROM otp_codes WHERE user_id = ? AND transaction_reference = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id, payload.transaction_reference),
        ).fetchone()

        if not otp_row:
            raise HTTPException(status_code=404, detail="No OTP request found for this transaction")
        if otp_row["verified"]:
            raise HTTPException(status_code=400, detail="This OTP has already been used")
        try:
            expires_at = datetime.fromisoformat(otp_row["expires_at"])
        except Exception:
            expires_at = datetime.utcnow() - timedelta(seconds=1)
        if datetime.utcnow() > expires_at:
            raise HTTPException(status_code=400, detail="OTP has expired. Please retry the transfer")
        if otp_row["code"] != payload.otp:
            raise HTTPException(status_code=400, detail="Incorrect OTP")

        txn = conn.execute(
            "SELECT * FROM transactions WHERE reference = ? AND status = 'pending_otp'",
            (payload.transaction_reference,),
        ).fetchone()
        if not txn:
            raise HTTPException(status_code=404, detail="Pending transaction not found")

        sender = get_user_by_id(txn["from_user_id"])
        if not sender or txn["amount"] > sender["balance"]:
            conn.execute("UPDATE transactions SET status = 'failed' WHERE reference = ?", (payload.transaction_reference,))
            raise HTTPException(status_code=400, detail="Insufficient balance to complete this transfer")

        conn.execute("UPDATE users SET balance = balance - ? WHERE id = ?", (txn["amount"], txn["from_user_id"]))
        conn.execute("UPDATE users SET balance = balance + ? WHERE id = ?", (txn["amount"], txn["to_user_id"]))
        conn.execute("UPDATE transactions SET status = 'completed' WHERE reference = ?", (payload.transaction_reference,))
        conn.execute("UPDATE otp_codes SET verified = 1 WHERE id = ?", (otp_row["id"],))

    return {"alert": "Transaction verified and completed successfully.", "reference": payload.transaction_reference}


# ---------------- SETTINGS / SECURITY ----------------

@app.get("/settings/get")
def settings_get(user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"theme": user["theme"], "hide_balance": bool(user["hide_balance"])}


@app.post("/settings/update")
def settings_update(payload: SettingsUpdate, user_id: str = Depends(get_current_user)):
    updates, params = [], []
    if payload.theme is not None:
        updates.append("theme = ?")
        params.append(payload.theme)
    if payload.hide_balance is not None:
        updates.append("hide_balance = ?")
        params.append(1 if payload.hide_balance else 0)
    if updates:
        params.append(user_id)
        with get_db() as conn:
            conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
    return {"alert": "Settings updated successfully"}


@app.get("/security/get-2fa")
def security_get_2fa(user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"enabled": bool(user["two_fa_enabled"])}


@app.get("/security/get-alerts")
def security_get_alerts(user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "sms": bool(user["alert_sms"]),
        "email": bool(user["alert_email"]),
        "in_app": bool(user["alert_in_app"]),
    }


@app.post("/pin/change")
def change_pin(payload: ChangePinRequest, user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_pin(payload.current_pin, user["transaction_pin"]):
        raise HTTPException(status_code=401, detail="Incorrect current PIN")
    with get_db() as conn:
        conn.execute("UPDATE users SET transaction_pin = ? WHERE id = ?", (hash_pin(payload.new_pin), user_id))
    return {"alert": "Transaction PIN updated successfully"}


# ---------------- ADMIN ----------------

@app.get("/admin/quarantine-list")
def admin_quarantine_list(user_id: str = Depends(get_current_user)):
    user = get_user_by_id(user_id)
    if not user or not user["is_admin"]:
        raise HTTPException(status_code=403, detail="Admin access required")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM quarantine ORDER BY created_at DESC LIMIT 100"
        ).fetchall()

    quarantine = []
    for row in rows:
        q = dict(row)
        owner = get_user_by_id(q["user_id"])
        quarantine.append({
            "id": q["id"],
            "reference": q["transaction_reference"],
            "user": owner["full_name"] if owner else "Unknown",
            "amount": q["amount"],
            "recipient_account": q["recipient_account"],
            "risk_score": q["risk_score"],
            "classification": q["classification"],
            "reason": q["reason"],
            "status": q["status"],
            "created_at": q["created_at"],
        })

    return {"quarantine": quarantine}


# ============================================================
# COMPLETE HTML FRONTEND (Embedded)
# ============================================================

FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sapphire Bank · Full Suite</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" />
    <style>
        /* ===== RESET & BASE ===== */
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: #f1f4f9; color: #0b1a2f; display: flex; min-height: 100vh; overflow-x: hidden; }
        :root { --primary: #0b1a2f; --gold: #f5b041; --light: #f8faff; --muted: #5a6f88; --border: #e9edf2; --shadow: 0 4px 20px rgba(0,0,0,0.02); }

        /* ===== ANIMATED BG LOGO ===== */
        .bg-logo {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 400px;
            color: rgba(11, 26, 47, 0.03);
            pointer-events: none;
            z-index: 0;
            animation: spinLogoBg 30s linear infinite;
            font-weight: 900;
            letter-spacing: -20px;
        }
        @keyframes spinLogoBg {
            0% { transform: translate(-50%, -50%) rotate(0deg) scale(1); }
            50% { transform: translate(-50%, -50%) rotate(180deg) scale(1.05); }
            100% { transform: translate(-50%, -50%) rotate(360deg) scale(1); }
        }
        .page, .login-page, .welcome-page { position: relative; z-index: 1; }

        /* ===== WELCOME PAGE ===== */
        .welcome-page {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: var(--primary);
            padding: 20px;
        }
        .welcome-box {
            background: white;
            border-radius: 40px;
            padding: 60px 50px;
            max-width: 520px;
            width: 100%;
            text-align: center;
            animation: fadeUp 0.8s ease;
        }
        .welcome-box .logo-big {
            font-size: 72px;
            color: var(--gold);
            display: block;
            margin-bottom: 16px;
            animation: spinLogo 20s linear infinite;
            display: inline-block;
        }
        @keyframes spinLogo { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .welcome-box h1 { font-size: 36px; color: var(--primary); margin-bottom: 8px; }
        .welcome-box p { color: var(--muted); font-size: 16px; margin-bottom: 30px; }
        .welcome-box .btn { width: 100%; justify-content: center; padding: 16px; font-size: 18px; }

        /* ===== LOGIN PAGE ===== */
        .login-page { display: none; justify-content: center; align-items: center; min-height: 100vh; background: var(--primary); }
        .login-page.active { display: flex; }
        .login-box { background: white; border-radius: 32px; padding: 48px 40px; max-width: 420px; width: 100%; animation: fadeUp 0.6s ease; }
        .login-box h1 { font-size: 28px; color: var(--primary); margin-bottom: 8px; }
        .login-box p { color: var(--muted); margin-bottom: 24px; }
        .login-box .form-group { margin-bottom: 18px; }
        .login-box .btn { width: 100%; justify-content: center; padding: 14px; }
        .login-box .login-footer { margin-top: 20px; text-align: center; font-size: 14px; color: var(--muted); }
        .login-box .login-footer a { color: var(--gold); text-decoration: none; font-weight: 600; cursor: pointer; }
        .login-box .demo-accounts { background: var(--light); border-radius: 16px; padding: 16px; margin-top: 20px; font-size: 13px; }

        /* ===== REGISTER PAGE ===== */
        .register-page { display: none; justify-content: center; align-items: center; min-height: 100vh; background: var(--primary); }
        .register-page.active { display: flex; }
        .register-container { background: white; border-radius: 32px; padding: 40px 35px; max-width: 480px; width: 100%; animation: fadeUp 0.6s ease; max-height: 90vh; overflow-y: auto; }
        .register-container h1 { font-size: 28px; color: var(--primary); margin-bottom: 8px; }
        .register-container p { color: var(--muted); margin-bottom: 20px; }
        .register-container .form-group { margin-bottom: 16px; }
        .register-container .btn { width: 100%; justify-content: center; padding: 14px; }
        .register-container .login-footer { margin-top: 20px; text-align: center; font-size: 14px; color: var(--muted); }
        .register-container .login-footer a { color: var(--gold); text-decoration: none; font-weight: 600; cursor: pointer; }
        .register-container .checkbox-group { display: flex; align-items: center; gap: 10px; margin: 15px 0; }
        .register-container .checkbox-group input { width: 18px; height: 18px; cursor: pointer; }

        /* ===== SIDEBAR ===== */
        .sidebar { width: 80px; background: var(--primary); color: #b0c4de; display: none; flex-direction: column; align-items: center; padding: 28px 0 20px; position: fixed; top: 0; left: 0; bottom: 0; z-index: 100; transition: width 0.25s ease, transform 0.3s ease; overflow: hidden; box-shadow: 4px 0 20px rgba(0,0,0,0.08); }
        .sidebar:hover { width: 260px; }

        .menu-toggle { display: none; position: fixed; top: 18px; left: 16px; z-index: 150; width: 44px; height: 44px; background: var(--primary); color: var(--gold); border: none; border-radius: 12px; font-size: 18px; align-items: center; justify-content: center; box-shadow: var(--shadow); cursor: pointer; }
        .sidebar-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 99; }
        .sidebar-overlay.show { display: block; }
        .sidebar .brand { font-size: 24px; font-weight: 700; color: var(--gold); letter-spacing: 0.5px; margin-bottom: 32px; white-space: nowrap; display: flex; align-items: center; gap: 14px; padding: 0 16px; }
        .sidebar .brand .logo-wrap { position: relative; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; animation: spinLogo 14s linear infinite; }
        .sidebar .brand .logo-wrap i { font-size: 36px; color: var(--gold); filter: drop-shadow(0 0 6px rgba(245,176,65,0.2)); }
        .sidebar .brand span { display: none; }
        .sidebar:hover .brand span { display: inline; }
        .sidebar .nav { display: flex; flex-direction: column; gap: 2px; width: 100%; padding: 0 10px; }
        .sidebar .nav a { display: flex; align-items: center; gap: 18px; padding: 12px 18px; border-radius: 14px; color: #b0c4de; text-decoration: none; font-size: 15px; font-weight: 500; transition: 0.15s; white-space: nowrap; }
        .sidebar .nav a i { width: 26px; font-size: 18px; text-align: center; }
        .sidebar .nav a span { display: none; }
        .sidebar:hover .nav a span { display: inline; }
        .sidebar .nav a:hover, .sidebar .nav a.active { background: #1e3550; color: white; }
        .sidebar .nav a.active { background: #1e3550; color: var(--gold); }
        .sidebar .bottom { margin-top: auto; padding-top: 20px; border-top: 1px solid #1e3550; width: 100%; display: flex; flex-direction: column; align-items: center; }
        .sidebar .bottom a { color: #b0c4de; text-decoration: none; padding: 12px 18px; display: flex; align-items: center; gap: 18px; width: 100%; font-size: 15px; border-radius: 14px; white-space: nowrap; cursor: pointer; }
        .sidebar .bottom a i { width: 26px; text-align: center; }
        .sidebar .bottom a span { display: none; }
        .sidebar:hover .bottom a span { display: inline; }
        .sidebar .bottom a:hover { background: #1e3550; color: white; }

        /* ===== MAIN ===== */
        .main { margin-left: 80px; flex: 1; padding: 28px 32px 48px; transition: margin-left 0.25s; min-height: 100vh; max-width: calc(100% - 80px); display: none; }
        .page { display: none; animation: fadeUp 0.35s ease; }
        .page.active { display: block; }
        @keyframes fadeUp { from { opacity: 0.3; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }

        /* ===== CARDS ===== */
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; }
        .grid-5 { display: grid; grid-template-columns: repeat(5, 1fr); gap: 20px; }
        .card { background: white; border-radius: 28px; padding: 24px 26px; box-shadow: var(--shadow); border: 1px solid var(--border); transition: 0.2s; position: relative; z-index: 1; }
        .card:hover { box-shadow: 0 8px 32px rgba(0,0,0,0.06); }
        .card h3 { font-size: 15px; font-weight: 600; color: var(--muted); margin-bottom: 10px; letter-spacing: 0.3px; }
        .card .value { font-size: 30px; font-weight: 700; color: var(--primary); }
        .card .change { font-size: 14px; font-weight: 500; color: #10b981; background: #e6f9f0; padding: 2px 14px; border-radius: 40px; display: inline-block; }
        .card .change.down { color: #ef4444; background: #fee9e9; }
        .stat-row { display: flex; justify-content: space-between; align-items: center; margin-top: 12px; flex-wrap: wrap; gap: 6px; }

        /* ===== TRANSACTIONS ===== */
        .txn-list { margin-top: 8px; max-height: 400px; overflow-y: auto; }
        .txn-item { display: flex; justify-content: space-between; align-items: center; padding: 14px 0; border-bottom: 1px solid #f0f2f6; }
        .txn-item:last-child { border: none; }
        .txn-item .left { display: flex; align-items: center; gap: 14px; }
        .txn-item .left .icon { width: 44px; height: 44px; border-radius: 16px; background: #eef2f7; display: flex; align-items: center; justify-content: center; color: var(--primary); font-size: 18px; }
        .txn-item .left .info .name { font-weight: 600; font-size: 15px; }
        .txn-item .left .info .meta { font-size: 13px; color: #7a8aa0; }
        .txn-item .amount { font-weight: 700; font-size: 16px; }
        .txn-item .amount.credit { color: #10b981; }
        .txn-item .amount.debit { color: #ef4444; }

        /* ===== BUTTONS ===== */
        .btn { border: none; padding: 10px 24px; border-radius: 60px; font-weight: 600; font-size: 14px; cursor: pointer; transition: 0.15s; background: #eef2f7; color: var(--primary); display: inline-flex; align-items: center; gap: 8px; }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: #1e3550; }
        .btn-gold { background: var(--gold); color: white; }
        .btn-gold:hover { background: #e09e2f; }
        .btn-outline { background: transparent; border: 1.5px solid #d0d8e2; }
        .btn-outline:hover { background: #f0f4fa; }
        .btn-sm { padding: 6px 16px; font-size: 13px; }
        .btn-danger { background: #ef4444; color: white; }
        .btn-danger:hover { background: #dc2626; }
        .flex-between { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; }

        /* ===== PROGRESS ===== */
        .progress-bar { background: #e9edf2; height: 8px; border-radius: 10px; width: 100%; margin: 8px 0 12px; }
        .progress-bar .fill { height: 8px; border-radius: 10px; background: var(--primary); }

        /* ===== ALERT FEED ===== */
        .alert-feed { max-height: 420px; overflow-y: auto; border-radius: 18px; border: 1px solid var(--border); background: white; }
        .alert-item { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; border-bottom: 1px solid #f0f2f6; font-size: 14px; }
        .alert-item:last-child { border: none; }
        .alert-item .badge { background: #fee9e9; color: #b91c1c; padding: 2px 16px; border-radius: 40px; font-size: 12px; font-weight: 600; }
        .alert-item .badge.green { background: #e6f9f0; color: #0b7a4a; }
        .alert-item .badge.yellow { background: #fef3c7; color: #92400e; }
        .stat-pill { background: #f0f4fa; padding: 6px 20px; border-radius: 40px; font-size: 13px; font-weight: 500; color: #1e3550; display: inline-block; }

        /* ===== TABLE ===== */
        .table-wrap { overflow-x: auto; margin-top: 12px; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th { text-align: left; padding: 12px 10px; background: #f8faff; color: var(--muted); font-weight: 600; border-bottom: 2px solid var(--border); }
        td { padding: 12px 10px; border-bottom: 1px solid #f0f2f6; }
        .status { padding: 4px 14px; border-radius: 40px; font-size: 12px; font-weight: 600; display: inline-block; }
        .status.success { background: #e6f9f0; color: #0b7a4a; }
        .status.pending { background: #fef3c7; color: #92400e; }
        .status.failed { background: #fee9e9; color: #b91c1c; }
        .status.fraud { background: #fee9e9; color: #b91c1c; }
        .status.safe { background: #e6f9f0; color: #0b7a4a; }
        .status.suspicious { background: #fef3c7; color: #92400e; }

        /* ===== FORM ELEMENTS ===== */
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; font-weight: 500; font-size: 14px; color: var(--muted); margin-bottom: 6px; }
        .form-group input, .form-group select, .form-group textarea { width: 100%; padding: 12px 16px; border: 1px solid var(--border); border-radius: 12px; font-size: 14px; transition: 0.15s; background: white; }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus { outline: none; border-color: var(--gold); box-shadow: 0 0 0 3px rgba(245,176,65,0.1); }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .form-error { color: #ef4444; font-size: 12px; margin-top: 4px; display: none; }

        /* ===== CHARTS ===== */
        .chart-container { width: 100%; height: 200px; margin: 12px 0; display: flex; align-items: flex-end; gap: 12px; padding: 10px 0; }
        .chart-bar { flex: 1; min-height: 20px; border-radius: 6px 6px 0 0; transition: height 0.6s ease; position: relative; }
        .chart-bar .label { position: absolute; bottom: -22px; left: 50%; transform: translateX(-50%); font-size: 10px; color: var(--muted); white-space: nowrap; }
        .chart-bar .val { position: absolute; top: -18px; left: 50%; transform: translateX(-50%); font-size: 10px; font-weight: 600; color: var(--primary); }
        .chart-row { display: flex; gap: 20px; flex-wrap: wrap; }
        .chart-box { flex: 1; min-width: 200px; background: var(--light); border-radius: 16px; padding: 16px; }
        .chart-box h4 { font-size: 14px; color: var(--muted); margin-bottom: 8px; }

        .pie-chart {
            width: 140px;
            height: 140px;
            border-radius: 50%;
            margin: 0 auto 12px;
            transition: transform 0.3s;
        }
        .pie-chart:hover { transform: scale(1.05); }
        .pie-legend { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; font-size: 12px; }
        .pie-legend span { display: flex; align-items: center; gap: 4px; }
        .pie-legend .dot { width: 12px; height: 12px; border-radius: 4px; display: inline-block; }

        .line-chart-svg { width: 100%; height: 120px; }
        .line-chart-svg svg { width: 100%; height: 100%; }

        .donut-chart {
            width: 140px;
            height: 140px;
            border-radius: 50%;
            margin: 0 auto 12px;
            position: relative;
            transition: transform 0.3s;
        }
        .donut-chart:hover { transform: scale(1.05); }
        .donut-chart .center-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            font-weight: 700;
            color: var(--primary);
        }

        /* ===== TOAST ===== */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 999;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .toast {
            background: white;
            border-radius: 16px;
            padding: 16px 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
            display: flex;
            align-items: center;
            gap: 12px;
            animation: slideInRight 0.4s ease;
            min-width: 280px;
            border-left: 4px solid var(--gold);
        }
        .toast.success { border-left-color: #10b981; }
        .toast.error { border-left-color: #ef4444; }
        .toast.info { border-left-color: #3b82f6; }
        .toast .toast-icon { font-size: 20px; }
        .toast .toast-msg { font-size: 14px; font-weight: 500; }
        @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* ===== PAGINATION ===== */
        .pagination {
            display: flex;
            gap: 6px;
            justify-content: center;
            margin-top: 16px;
        }
        .pagination button {
            padding: 6px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: white;
            cursor: pointer;
            font-size: 13px;
            transition: 0.15s;
        }
        .pagination button:hover { background: var(--light); }
        .pagination button.active { background: var(--primary); color: white; border-color: var(--primary); }

        /* ===== MODAL ===== */
        .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5); z-index: 200; justify-content: center; align-items: center; }
        .modal-overlay.show { display: flex; }
        .modal { background: white; border-radius: 28px; padding: 32px; max-width: 500px; width: 90%; max-height: 90vh; overflow-y: auto; animation: modalIn 0.3s ease; }
        @keyframes modalIn { from { transform: scale(0.9); opacity: 0; } to { transform: scale(1); opacity: 1; } }
        .modal .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .modal .modal-header h2 { font-size: 22px; }
        .modal .modal-close { background: none; border: none; font-size: 28px; cursor: pointer; color: var(--muted); }

        /* ===== TABS ===== */
        .tabs { display: flex; gap: 4px; background: #f0f4fa; border-radius: 14px; padding: 4px; margin-bottom: 20px; flex-wrap: wrap; }
        .tabs .tab { padding: 10px 24px; border-radius: 12px; border: none; background: transparent; font-weight: 500; cursor: pointer; transition: 0.15s; }
        .tabs .tab.active { background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* ===== MISC ===== */
        .text-muted { color: var(--muted); }
        .mt-12 { margin-top: 12px; }
        .mt-20 { margin-top: 20px; }
        .mb-12 { margin-bottom: 12px; }
        .mb-20 { margin-bottom: 20px; }
        .w-full { width: 100%; }
        .text-center { text-align: center; }
        .text-right { text-align: right; }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 1200px) { .grid-5 { grid-template-columns: repeat(3, 1fr); } }
        @media (max-width: 992px) {
            .sidebar { width: 260px; transform: translateX(-100%); box-shadow: 4px 0 30px rgba(0,0,0,0.3); }
            .sidebar:hover { width: 260px; }
            .sidebar.mobile-open { transform: translateX(0); }
            .sidebar .brand span, .sidebar .nav a span, .sidebar .bottom a span { display: inline !important; }
            body.logged-in .menu-toggle { display: flex; }
            .main { margin-left: 0; padding: 76px 16px 32px; max-width: 100%; }
            .grid-4, .grid-5 { grid-template-columns: 1fr 1fr; }
            .grid-3 { grid-template-columns: 1fr 1fr; }
            .form-row { grid-template-columns: 1fr; }
            .welcome-box { padding: 40px 30px; }
            .login-box { padding: 32px 24px; }
        }
        @media (max-width: 600px) {
            .grid-2, .grid-3, .grid-4, .grid-5 { grid-template-columns: 1fr; }
            .chart-row { flex-direction: column; }
            .pie-chart { width: 120px; height: 120px; }
            .donut-chart { width: 120px; height: 120px; }
        }
        @media (max-width: 480px) {
            .main { padding: 72px 12px 28px; }
            .card { padding: 18px 16px; }
            .modal { padding: 22px; border-radius: 20px; }
            h1 { font-size: 22px !important; }
        }
    </style>
</head>
<body>

<!-- ===== ANIMATED BG LOGO ===== -->
<div class="bg-logo">⬦</div>

<!-- ===== MOBILE MENU TOGGLE ===== -->
<button class="menu-toggle" id="menuToggle" onclick="toggleSidebar()" aria-label="Open menu"><i class="fas fa-bars"></i></button>
<div class="sidebar-overlay" id="sidebarOverlay" onclick="closeSidebar()"></div>

<!-- ===== TOAST CONTAINER ===== -->
<div class="toast-container" id="toastContainer"></div>

<!-- ===== WELCOME PAGE ===== -->
<div id="welcomePage" class="welcome-page">
    <div class="welcome-box">
        <span class="logo-big"><i class="fas fa-gem"></i></span>
        <h1>Sapphire Bank</h1>
        <p>Your trusted financial partner. Secure, fast, and reliable banking at your fingertips.</p>
        <button class="btn btn-gold" onclick="showLogin()"><i class="fas fa-arrow-right"></i> Get Started</button>
    </div>
</div>

<!-- ===== LOGIN PAGE ===== -->
<div id="loginPage" class="login-page">
    <div class="login-box">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <div style="animation:spinLogo 14s linear infinite; display:inline-block;"><i class="fas fa-gem" style="font-size:36px; color:var(--gold);"></i></div>
            <h1>Sapphire Bank</h1>
        </div>
        <p>Welcome back! Sign in to continue to your account.</p>
        <div class="form-group"><label>Username or Email</label><input type="text" id="loginEmail" value="john_doe" /></div>
        <div class="form-group"><label>PIN</label><input type="password" id="loginPassword" value="1234" /></div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <label style="font-size:14px; color:var(--muted);"><input type="checkbox" checked /> Remember Me</label>
            <a href="#" style="color:var(--gold); font-size:14px; text-decoration:none;">Forgot PIN?</a>
        </div>
        <button class="btn btn-gold" onclick="handleLogin()"><i class="fas fa-sign-in-alt"></i> Sign In</button>
        <div class="login-footer">
            Don't have an account? <a href="#" onclick="showRegister()">Create Account</a>
        </div>
        <div class="demo-accounts">
            <strong>Demo Accounts:</strong><br />
            <strong>Admin:</strong> admin / PIN 0000<br />
            <strong>John Doe:</strong> john_doe / PIN 1234 (Account: 1234567890)<br />
            <strong>Jane Smith:</strong> jane_smith / PIN 1234 (Account: 0987654321)<br />
            <strong>Mike Johnson:</strong> mike_johnson / PIN 1234 (Account: 1122334455)<br />
            <strong>Sarah Lee:</strong> sarah_lee / PIN 1234 (Account: 5566778899)<br />
            <strong>James Bond:</strong> james_bond / PIN 1234 (Account: 6677889900)
        </div>
    </div>
</div>

<!-- ===== REGISTER PAGE ===== -->
<div id="registerPage" class="register-page">
    <div class="register-container">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <div style="animation:spinLogo 14s linear infinite; display:inline-block;"><i class="fas fa-gem" style="font-size:36px; color:var(--gold);"></i></div>
            <h1>Sapphire Bank</h1>
        </div>
        <p>Create your account to get started.</p>
        <div class="form-group"><label>Username</label><input type="text" id="regUsername" placeholder="Choose a username" /></div>
        <div class="form-group"><label>Email</label><input type="email" id="regEmail" placeholder="you@example.com" /></div>
        <div class="form-group"><label>Full Name</label><input type="text" id="regFullName" placeholder="John Doe" /></div>
        <div class="form-group"><label>Phone Number</label><input type="tel" id="regPhone" placeholder="08012345678" /></div>
        <div class="form-group"><label>Account Type</label><select id="regAccountType"><option value="savings">Savings</option><option value="current">Current</option></select></div>
        <div class="form-group"><label>Account Category</label><select id="regAccountCategory"><option value="personal">Personal</option><option value="business">Business</option></select></div>
        <div class="form-group"><label>Login PIN (4 digits)</label><input type="password" id="regLoginPin" maxlength="4" placeholder="****" /></div>
        <div class="form-group"><label>Transaction PIN (4 digits)</label><input type="password" id="regTransPin" maxlength="4" placeholder="****" /></div>
        <div class="checkbox-group"><input type="checkbox" id="regTerms"><label>I agree to the <span onclick="alert('Terms and Conditions')" style="color:var(--gold);cursor:pointer;">Terms and Conditions</span></label></div>
        <button class="btn btn-gold" onclick="handleRegister()"><i class="fas fa-user-plus"></i> Create Account</button>
        <div class="login-footer">Already have an account? <a href="#" onclick="showLogin()">Sign In</a></div>
        <div id="registerError" style="display:none; background:#fee2e2; color:#dc2626; padding:12px; border-radius:10px; margin-top:15px;"></div>
        <div id="registerSuccess" style="display:none; background:#d1fae5; color:#059669; padding:12px; border-radius:10px; margin-top:15px;"></div>
    </div>
</div>

<!-- ===== SIDEBAR ===== -->
<aside class="sidebar" id="sidebar">
    <div class="brand">
        <div class="logo-wrap"><i class="fas fa-gem"></i></div>
        <span>Sapphire</span>
    </div>
    <nav class="nav">
        <a href="#" class="active" data-page="dashboard"><i class="fas fa-th-large"></i><span>Dashboard</span></a>
        <a href="#" data-page="payguard"><i class="fas fa-shield-alt"></i><span>PayGuard</span></a>
        <a href="#" data-page="transfers"><i class="fas fa-exchange-alt"></i><span>Transfers</span></a>
        <a href="#" data-page="cards"><i class="fas fa-credit-card"></i><span>Cards</span></a>
        <a href="#" data-page="beneficiaries"><i class="fas fa-users"></i><span>Beneficiaries</span></a>
        <a href="#" data-page="transactions"><i class="fas fa-list-ul"></i><span>Transactions</span></a>
        <a href="#" data-page="savings"><i class="fas fa-piggy-bank"></i><span>Savings</span></a>
        <a href="#" data-page="investments"><i class="fas fa-chart-line"></i><span>Investments</span></a>
        <a href="#" data-page="loans"><i class="fas fa-hand-holding-usd"></i><span>Loans</span></a>
        <a href="#" data-page="rewards"><i class="fas fa-gift"></i><span>Rewards</span></a>
        <a href="#" data-page="support"><i class="fas fa-headset"></i><span>Support</span></a>
        <a href="#" data-page="settings"><i class="fas fa-cog"></i><span>Settings</span></a>
        <a href="#" data-page="analytics"><i class="fas fa-chart-pie"></i><span>Analytics</span></a>
        <a href="#" data-page="profile"><i class="fas fa-user-circle"></i><span>Profile</span></a>
    </nav>
    <div class="bottom">
        <a onclick="handleLogout()"><i class="fas fa-sign-out-alt"></i><span>Logout</span></a>
    </div>
</aside>

<!-- ===== MAIN CONTENT ===== -->
<main class="main" id="mainContent">

    <!-- ==================== DASHBOARD ==================== -->
    <section id="page-dashboard" class="page active">
        <div class="flex-between" style="margin-bottom: 28px;">
            <div><h1 style="font-size: 28px; font-weight: 700;">Dashboard</h1><p style="color: var(--muted); font-size: 16px;">Welcome back, <span id="userName">Stacey</span>! 🎉 Here's what's happening with your finances today.</p></div>
            <div><button class="btn btn-gold" onclick="showToast('New transfer initiated!', 'success')"><i class="fas fa-plus"></i> New Transfer</button></div>
        </div>

        <div class="grid-4" style="margin-bottom: 30px;">
            <div class="card"><h3>Total Balance</h3><div class="value" id="totalBalance">₦4,256,789.50</div><div class="stat-row"><span class="change"><i class="fas fa-arrow-up"></i> 12.5%</span> <span style="color:#7a8aa0;">from last month</span></div></div>
            <div class="card"><h3>Income</h3><div class="value" id="totalIncome">₦1,230,000</div><div class="stat-row"><span class="change"><i class="fas fa-arrow-up"></i> 8.2%</span></div></div>
            <div class="card"><h3>Expenses</h3><div class="value" id="totalExpenses">₦540,000</div><div class="stat-row"><span class="change down"><i class="fas fa-arrow-down"></i> 2.1%</span></div></div>
            <div class="card"><h3>Scanner AI</h3><div class="value" style="font-size:22px;">Level 9 Guardian</div><div class="stat-row"><span class="change" style="background:#fef3c7; color:#92400e;">86% Accuracy</span> <span style="color:#7a8aa0;">12 catches</span></div></div>
        </div>

        <div class="chart-row" style="margin-bottom: 24px;">
            <div class="chart-box"><h4>📊 Monthly Spending (Bar)</h4><div class="chart-container" id="barChart"></div></div>
            <div class="chart-box"><h4>🧩 Expense Breakdown (Pie)</h4><div class="pie-chart" id="pieChart"></div><div class="pie-legend" id="pieLegend"></div></div>
            <div class="chart-box"><h4>📈 Balance Trend (Line)</h4><div class="line-chart-svg" id="lineChart"></div></div>
        </div>

        <div class="card" style="margin-bottom: 30px;">
            <h3 style="margin-bottom: 16px;">Quick Actions</h3>
            <div style="display:flex; flex-wrap:wrap; gap:12px;">
                <button class="btn btn-primary" onclick="showPage('page-transfers')"><i class="fas fa-paper-plane"></i> Transfer</button>
                <button class="btn btn-primary"><i class="fas fa-file-invoice"></i> Pay Bills</button>
                <button class="btn btn-primary"><i class="fas fa-mobile-alt"></i> Airtime</button>
                <button class="btn btn-primary"><i class="fas fa-qrcode"></i> Scan & Pay</button>
                <button class="btn btn-primary"><i class="fas fa-credit-card"></i> Virtual Card</button>
                <button class="btn btn-outline"><i class="fas fa-ellipsis-h"></i> More</button>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <div class="flex-between"><h3>Recent Transactions</h3><a href="#" style="color:var(--primary); font-weight:600; font-size:14px;" onclick="showPage('page-transactions')">See all</a></div>
                <div class="txn-list" id="recentTransactions"></div>
            </div>
            <div>
                <div class="card" style="margin-bottom:18px;">
                    <h3>Scanner AI Status</h3>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin:10px 0 8px;">
                        <span><strong>Level 9 Guardian</strong><br><span style="font-size:13px; color:#7a8aa0;">Experience · Fraud Catches</span></span>
                        <span class="stat-pill">86% Accuracy</span>
                    </div>
                    <div class="progress-bar"><div class="fill" style="width:86%; background:var(--gold);"></div></div>
                    <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
                        <button class="btn btn-outline" style="flex:1;" onclick="showPage('page-payguard')">View Scanner AI</button>
                        <button class="btn btn-primary" style="flex:1;">Security Center</button>
                    </div>
                </div>
                <div class="card" style="background:var(--primary); color:white; border:none;">
                    <h3 style="color:var(--gold);">Stay Secure, Always</h3>
                    <p style="color:#b0c4de; font-size:14px; margin:8px 0 14px;">Enable 2FA & transaction alerts for maximum protection.</p>
                    <div style="display:flex; gap:12px; flex-wrap:wrap;">
                        <button class="btn btn-gold" style="flex:1;" onclick="showToast('2FA enabled successfully!', 'success')">Enable 2FA</button>
                        <button class="btn" style="background:#1e3550; color:white; flex:1;">Learn More</button>
                    </div>
                </div>
                <div class="card" style="margin-top:18px;">
                    <h3>Spending Overview</h3>
                    <div style="margin-top:12px;">
                        <div style="display:flex; justify-content:space-between;"><span>Food & Dining</span><span>₦185,000</span></div><div class="progress-bar"><div class="fill" style="width:34%; background:#f59e0b;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Transport</span><span>₦72,000</span></div><div class="progress-bar"><div class="fill" style="width:13%; background:#3b82f6;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Utilities</span><span>₦98,000</span></div><div class="progress-bar"><div class="fill" style="width:18%; background:#8b5cf6;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Entertainment</span><span>₦45,000</span></div><div class="progress-bar"><div class="fill" style="width:8%; background:#ec4899;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Shopping</span><span>₦85,000</span></div><div class="progress-bar"><div class="fill" style="width:16%; background:#10b981;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Others</span><span>₦55,000</span></div><div class="progress-bar"><div class="fill" style="width:11%; background:#6366f1;"></div></div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- ==================== PAYGUARD ==================== -->
    <section id="page-payguard" class="page">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:24px;">
            <div><h1 style="font-size:28px; font-weight:700;"><i class="fas fa-shield-alt" style="color:var(--gold);"></i> PayGuard AI Command Center</h1><p style="color:var(--muted); font-size:16px;">Real-time Fraud Monitoring & Intelligence</p></div>
            <span class="stat-pill"><i class="fas fa-circle" style="color:#10b981; font-size:10px;"></i> Live</span>
        </div>
        <div class="grid-4" style="margin-bottom:26px;">
            <div class="card"><h3>Total Alerts</h3><div class="value">1,284</div><span class="change"><i class="fas fa-arrow-up"></i> 8%</span></div>
            <div class="card"><h3>Pending</h3><div class="value">342</div><span class="change" style="background:#fef3c7; color:#92400e;">High</span></div>
            <div class="card"><h3>Blocked</h3><div class="value">97</div><span class="change down"><i class="fas fa-arrow-down"></i> 2%</span></div>
            <div class="card"><h3>Accuracy</h3><div class="value">94%</div><span class="change"><i class="fas fa-arrow-up"></i> 1.2%</span></div>
        </div>
        <div class="grid-2">
            <div class="card">
                <div class="flex-between"><h3>Alerts Queue</h3><span style="font-size:13px; color:var(--muted);">Last 24 hours</span></div>
                <div class="alert-feed" id="alertFeed"></div>
            </div>
            <div>
                <div class="card" style="margin-bottom:18px;"><h3>Model Performance</h3>
                    <div style="margin-top:12px;">
                        <div style="display:flex; justify-content:space-between;"><span>Scanner AI</span><span>94%</span></div><div class="progress-bar"><div class="fill" style="width:94%;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Random Forest</span><span>88%</span></div><div class="progress-bar"><div class="fill" style="width:88%;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>XGBoost</span><span>91%</span></div><div class="progress-bar"><div class="fill" style="width:91%;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Neural Net</span><span>87%</span></div><div class="progress-bar"><div class="fill" style="width:87%;"></div></div>
                        <div style="display:flex; justify-content:space-between;"><span>Ensemble</span><span>93%</span></div><div class="progress-bar"><div class="fill" style="width:93%;"></div></div>
                    </div>
                </div>
                <div class="card"><h3>Fraud Map</h3>
                    <div style="display:flex; flex-wrap:wrap; gap:12px; margin-top:10px;">
                        <span class="stat-pill">Lagos · 23</span><span class="stat-pill">Abuja · 11</span>
                        <span class="stat-pill">Port Harcourt · 8</span><span class="stat-pill">Kano · 4</span>
                        <span class="stat-pill">Ibadan · 6</span><span class="stat-pill">Others · 15</span>
                        <span class="stat-pill">Enugu · 3</span><span class="stat-pill">Benin · 2</span>
                    </div>
                    <div style="margin-top:14px;"><button class="btn btn-primary">View Full Map</button></div>
                </div>
                <div class="card" style="margin-top:18px;">
                    <h3>Recent Cases</h3>
                    <div style="font-size:14px; margin-top:8px;" id="recentCases"></div>
                </div>
            </div>
        </div>
    </section>

    <!-- ==================== TRANSFERS ==================== -->
    <section id="page-transfers" class="page">
        <div class="card">
            <h1 style="font-size:26px; margin-bottom:6px;">Transfers</h1>
            <p style="color:var(--muted); margin-bottom:20px;">Send money to beneficiaries, banks, or cards.</p>
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px;">
                <button class="btn btn-primary" onclick="document.getElementById('internalAccount').scrollIntoView({behavior:'smooth', block:'center'})"><i class="fas fa-plus"></i> New Transfer</button>
                <button class="btn btn-outline"><i class="fas fa-history"></i> Recent</button>
                <button class="btn btn-outline"><i class="fas fa-star"></i> Favorites</button>
            </div>
            <div class="tabs">
                <button class="tab active" onclick="switchTab(this,'transfer-internal')">Internal</button>
                <button class="tab" onclick="switchTab(this,'transfer-bank')">Other Banks</button>
                <button class="tab" onclick="switchTab(this,'transfer-international')">International</button>
            </div>
            <div id="transfer-internal" class="tab-content active">
                <div class="form-row">
                    <div class="form-group"><label>From Account</label><select><option>Current Account · ₦4,256,789</option><option>Savings Account · ₦1,250,000</option></select></div>
                    <div class="form-group"><label>To Account</label><input placeholder="Enter account number or select beneficiary" id="internalAccount" /></div>
                </div>
                <div class="form-row">
                    <div class="form-group"><label>Amount (₦)</label><input type="number" placeholder="0.00" id="internalAmount" /></div>
                    <div class="form-group"><label>Description (Optional)</label><input placeholder="What's this for?" id="internalDesc" /></div>
                </div>
                <div class="form-group"><label>Transaction PIN</label><input type="password" maxlength="4" placeholder="****" id="internalPin" /></div>
                <button class="btn btn-gold w-full" style="justify-content:center;" onclick="validateTransfer('internal')">Send Money</button>
            </div>
            <div id="transfer-bank" class="tab-content">
                <div class="form-row"><div class="form-group"><label>From Account</label><select><option>Current Account · ₦4,256,789</option></select></div>
                <div class="form-group"><label>Bank Name</label><select><option>Access Bank</option><option>GTBank</option><option>First Bank</option><option>Zenith Bank</option><option>UBA</option></select></div></div>
                <div class="form-row"><div class="form-group"><label>Account Number</label><input placeholder="Enter 10-digit account number" id="otherAccount" /></div>
                <div class="form-group"><label>Account Name</label><input placeholder="Account holder name" id="otherName" disabled /></div></div>
                <div class="form-row"><div class="form-group"><label>Amount (₦)</label><input type="number" placeholder="0.00" id="otherAmount" /></div>
                <div class="form-group"><label>Description</label><input placeholder="Reference" id="otherDesc" /></div></div>
                <div class="form-group"><label>Transaction PIN</label><input type="password" maxlength="4" placeholder="****" id="otherPin" /></div>
                <button class="btn btn-gold w-full" style="justify-content:center;" onclick="validateTransfer('other')">Transfer to Other Bank</button>
            </div>
            <div id="transfer-international" class="tab-content">
                <div class="form-row"><div class="form-group"><label>From Account</label><select><option>Current Account · ₦4,256,789</option></select></div>
                <div class="form-group"><label>Destination Country</label><select><option>United States</option><option>United Kingdom</option><option>Canada</option></select></div></div>
                <div class="form-row"><div class="form-group"><label>Currency</label><select><option>USD</option><option>EUR</option><option>GBP</option></select></div>
                <div class="form-group"><label>Amount</label><input type="number" placeholder="0.00" id="intlAmount" /></div></div>
                <div class="form-row"><div class="form-group"><label>Beneficiary Name</label><input placeholder="Full name" id="intlName" /></div>
                <div class="form-group"><label>SWIFT / IBAN</label><input placeholder="SWIFT or IBAN code" id="intlSwift" /></div></div>
                <div class="form-group"><label>Transaction PIN</label><input type="password" maxlength="4" placeholder="****" id="intlPin" /></div>
                <button class="btn btn-gold w-full" style="justify-content:center;" onclick="validateTransfer('international')">Send International</button>
            </div>
            <div style="margin-top:24px; background:var(--light); border-radius:20px; padding:20px;">
                <h4 style="margin-bottom:12px;">Quick Beneficiaries</h4>
                <div style="display:flex; gap:16px; flex-wrap:wrap;" id="quickBeneficiaries"></div>
            </div>
        </div>
    </section>

    <!-- ==================== CARDS ==================== -->
    <section id="page-cards" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Cards</h1><p style="color:var(--muted); margin-bottom:20px;">Manage your debit, credit, and virtual cards.</p>
            <div style="display:flex; gap:14px; flex-wrap:wrap;">
                <button class="btn btn-primary" onclick="showToast('Virtual card requested!', 'success')"><i class="fas fa-credit-card"></i> Virtual Card</button>
                <button class="btn btn-outline"><i class="fas fa-id-card"></i> Physical Card</button>
                <button class="btn btn-outline"><i class="fas fa-lock"></i> Freeze Card</button>
                <button class="btn btn-outline"><i class="fas fa-eye"></i> View All</button>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:24px;" id="cardList"></div>
        </div>
    </section>

    <!-- ==================== BENEFICIARIES ==================== -->
    <section id="page-beneficiaries" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Beneficiaries</h1><p style="color:var(--muted); margin-bottom:20px;">Manage your saved recipients.</p>
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px;">
                <button class="btn btn-primary" onclick="showToast('Add beneficiary form opened', 'info')"><i class="fas fa-user-plus"></i> Add</button>
                <button class="btn btn-outline"><i class="fas fa-sync"></i> Sync</button>
                <button class="btn btn-outline"><i class="fas fa-trash"></i> Manage</button>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;" id="beneficiaryList"></div>
        </div>
    </section>

    <!-- ==================== TRANSACTIONS ==================== -->
    <section id="page-transactions" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Transaction History</h1><p style="color:var(--muted); margin-bottom:20px;">Full transaction history with filters.</p>
            <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px;">
                <button class="btn btn-outline btn-sm active" onclick="filterTransactions('all')">All</button>
                <button class="btn btn-outline btn-sm" onclick="filterTransactions('credit')">Credit</button>
                <button class="btn btn-outline btn-sm" onclick="filterTransactions('debit')">Debit</button>
                <button class="btn btn-outline btn-sm" onclick="filterTransactions('month')">This Month</button>
                <button class="btn btn-primary btn-sm"><i class="fas fa-download"></i> Export</button>
                <button class="btn btn-outline btn-sm"><i class="fas fa-print"></i> Print</button>
            </div>
            <div class="table-wrap"><table><thead><tr><th>Ref</th><th>Date</th><th>Description</th><th>Type</th><th>Amount</th><th>Status</th><th>Action</th></tr></thead>
            <tbody id="transactionTableBody"></tbody></table></div>
            <div class="pagination" id="transactionPagination">
                <button class="active" onclick="pageTransactions(1)">1</button>
                <button onclick="pageTransactions(2)">2</button>
                <button onclick="pageTransactions(3)">3</button>
                <button onclick="pageTransactions(4)">4</button>
                <button onclick="pageTransactions(5)">5</button>
            </div>
        </div>
    </section>

    <!-- ==================== SAVINGS ==================== -->
    <section id="page-savings" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Savings</h1><p style="color:var(--muted); margin-bottom:20px;">Earn up to 7% interest per annum.</p>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;" id="savingsList"></div>
            <button class="btn btn-gold" style="margin-top:20px;" onclick="showToast('Savings account opened!', 'success')"><i class="fas fa-plus"></i> Start Saving</button>
        </div>
    </section>

    <!-- ==================== INVESTMENTS ==================== -->
    <section id="page-investments" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Investments</h1><p style="color:var(--muted); margin-bottom:20px;">Grow your wealth with Sapphire.</p>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;" id="investmentList"></div>
            <button class="btn btn-primary" style="margin-top:20px;" onclick="showToast('Investment advisor connected!', 'info')"><i class="fas fa-plus"></i> Explore</button>
        </div>
    </section>

    <!-- ==================== LOANS ==================== -->
    <section id="page-loans" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Loans</h1><p style="color:var(--muted); margin-bottom:20px;">Apply for personal or business loans.</p>
            <div style="display:flex; gap:14px; flex-wrap:wrap;">
                <button class="btn btn-primary" onclick="showToast('Loan application form opened', 'info')"><i class="fas fa-hand-holding-usd"></i> Apply Now</button>
                <button class="btn btn-outline"><i class="fas fa-calculator"></i> Calculate</button>
                <button class="btn btn-outline"><i class="fas fa-history"></i> My Loans</button>
            </div>
            <div style="margin-top:20px; background:var(--light); border-radius:20px; padding:20px;" id="loanList"></div>
            <div style="margin-top:20px; background:#fef3c7; border-radius:20px; padding:16px; border:1px solid #fde68a;">
                <p style="font-size:14px;"><i class="fas fa-info-circle" style="color:#92400e;"></i> <strong>Special Offer:</strong> 0% interest on first ₦100,000 for new loan applicants.</p>
            </div>
        </div>
    </section>

    <!-- ==================== REWARDS ==================== -->
    <section id="page-rewards" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Rewards</h1><p style="color:var(--muted); margin-bottom:20px;">Cashback, points, and exclusive offers.</p>
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;" id="rewardsStats"></div>
            <div style="margin-top:20px; background:var(--light); border-radius:20px; padding:20px;">
                <h4 style="margin-bottom:12px;">Available Offers</h4><div style="display:flex; gap:12px; flex-wrap:wrap;" id="offersList"></div>
            </div>
            <div style="margin-top:16px; background:linear-gradient(135deg, var(--gold), #e09e2f); border-radius:20px; padding:20px; color:white;">
                <h4>🆕 Tier Rewards</h4><p style="font-size:14px; opacity:0.9;">You're 2,550 points away from Gold Tier.</p>
                <div class="progress-bar" style="background:rgba(255,255,255,0.3);"><div class="fill" style="width:74%; background:white;"></div></div>
                <div style="display:flex; justify-content:space-between; font-size:13px; margin-top:4px;"><span>Silver</span><span>Gold</span><span>Platinum</span></div>
            </div>
        </div>
    </section>

    <!-- ==================== SUPPORT ==================== -->
    <section id="page-support" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Support</h1><p style="color:var(--muted); margin-bottom:20px;">24/7 help center and live chat.</p>
            <div style="display:flex; gap:14px; flex-wrap:wrap;">
                <button class="btn btn-primary" onclick="showToast('Live chat connected!', 'success')"><i class="fas fa-comment"></i> Live Chat</button>
                <button class="btn btn-outline"><i class="fas fa-envelope"></i> Email</button>
                <button class="btn btn-outline"><i class="fas fa-phone"></i> Call</button>
                <button class="btn btn-outline"><i class="fas fa-book"></i> FAQ</button>
            </div>
            <div style="margin-top:20px; background:var(--light); border-radius:20px; padding:20px;">
                <p><strong>📞 Hotline:</strong> 0800-SAPPHIRE (0800-727-7447)</p>
                <p><strong>✉️ Email:</strong> support@sapphirebank.com</p>
                <p><strong>🕒 Hours:</strong> 24/7, including holidays</p>
                <p><strong>📍 Branches:</strong> Lagos, Abuja, Port Harcourt, Kano, Ibadan, Enugu, Benin</p>
            </div>
            <div style="margin-top:16px; display:grid; grid-template-columns:1fr 1fr; gap:16px;">
                <div style="background:var(--light); border-radius:16px; padding:16px; text-align:center;">
                    <i class="fas fa-robot" style="font-size:32px; color:var(--gold);"></i>
                    <h4 style="margin:8px 0;">AI Assistant</h4>
                    <p style="font-size:13px; color:var(--muted);">Get instant answers 24/7</p>
                    <button class="btn btn-primary btn-sm" style="margin-top:8px;" onclick="showToast('AI Assistant activated!', 'info')">Chat Now</button>
                </div>
                <div style="background:var(--light); border-radius:16px; padding:16px; text-align:center;">
                    <i class="fas fa-file-alt" style="font-size:32px; color:var(--gold);"></i>
                    <h4 style="margin:8px 0;">Knowledge Base</h4>
                    <p style="font-size:13px; color:var(--muted);">Articles & guides</p>
                    <button class="btn btn-outline btn-sm" style="margin-top:8px;">Browse</button>
                </div>
            </div>
        </div>
    </section>

    <!-- ==================== SETTINGS ==================== -->
    <section id="page-settings" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Settings</h1><p style="color:var(--muted); margin-bottom:20px;">Profile, security, notifications, and more.</p>
            <div style="display:flex; gap:14px; flex-wrap:wrap; margin-bottom:20px;">
                <button class="btn btn-outline" onclick="showPage('page-profile')"><i class="fas fa-user"></i> Profile</button>
                <button class="btn btn-outline"><i class="fas fa-lock"></i> Security</button>
                <button class="btn btn-outline"><i class="fas fa-bell"></i> Notifications</button>
                <button class="btn btn-primary" onclick="showToast('Password change form opened', 'info')"><i class="fas fa-key"></i> Change Password</button>
                <button class="btn btn-outline" onclick="showToast('2FA setup initiated', 'info')"><i class="fas fa-mobile-alt"></i> 2FA Setup</button>
                <button class="btn btn-outline"><i class="fas fa-language"></i> Language</button>
            </div>
            <div style="background:var(--light); border-radius:20px; padding:20px;" id="settingsList"></div>
            <div style="margin-top:20px;">
                <h4 style="margin-bottom:12px;">Account Preferences</h4>
                <div class="form-row"><div class="form-group"><label>Default Account</label><select><option>Current</option><option>Savings</option></select></div>
                <div class="form-group"><label>Currency</label><select><option>₦ NGN</option><option>$ USD</option></select></div></div>
                <div class="form-row"><div class="form-group"><label>Language</label><select><option>English</option><option>French</option><option>Yoruba</option></select></div>
                <div class="form-group"><label>Time Zone</label><select><option>Africa/Lagos (UTC+1)</option><option>Europe/London</option></select></div></div>
                <button class="btn btn-primary" onclick="showToast('Preferences saved!', 'success')"><i class="fas fa-save"></i> Save</button>
            </div>
        </div>
    </section>

    <!-- ==================== PROFILE ==================== -->
    <section id="page-profile" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">My Profile</h1><p style="color:var(--muted); margin-bottom:20px;">View and manage your personal information.</p>
            <div style="display:flex; gap:30px; flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="width:120px; height:120px; background:var(--primary); border-radius:60px; display:flex; align-items:center; justify-content:center; margin:0 auto; font-size:48px; color:white; cursor:pointer;" onclick="showToast('Profile picture updated!', 'success')"><i class="fas fa-user"></i></div>
                    <button class="btn btn-outline btn-sm" style="margin-top:8px;" onclick="showToast('Upload profile picture', 'info')">Change Photo</button>
                </div>
                <div style="flex:1; min-width:250px;">
                    <div class="form-row"><div class="form-group"><label>Full Name</label><input value="John Doe" id="profileName" /></div><div class="form-group"><label>Email</label><input value="john@sapphire.com" id="profileEmail" /></div></div>
                    <div class="form-row"><div class="form-group"><label>Phone</label><input value="08012345678" id="profilePhone" /></div><div class="form-group"><label>Date of Birth</label><input type="date" value="1990-01-15" id="profileDob" /></div></div>
                    <div class="form-row"><div class="form-group"><label>Address</label><input value="42 Bourdillon Road, Ikoyi, Lagos" id="profileAddress" /></div><div class="form-group"><label>Occupation</label><input value="Software Engineer" id="profileOccupation" /></div></div>
                    <div style="display:flex; gap:12px; margin-top:12px;">
                        <button class="btn btn-primary" onclick="showToast('Profile updated successfully!', 'success')"><i class="fas fa-save"></i> Save</button>
                        <button class="btn btn-outline" onclick="showPage('page-settings')">Cancel</button>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- ==================== ANALYTICS ==================== -->
    <section id="page-analytics" class="page">
        <div class="card"><h1 style="font-size:26px; margin-bottom:6px;">Analytics</h1><p style="color:var(--muted); margin-bottom:20px;">Deep insights into your financial behavior.</p>
            <div class="grid-3" style="margin-bottom:20px;" id="analyticsStats"></div>
            <div class="chart-row" style="margin-bottom:20px;">
                <div class="chart-box"><h4>📊 Monthly Spending (Bar)</h4><div class="chart-container" id="analyticsBarChart"></div></div>
                <div class="chart-box"><h4>🧩 Category Breakdown (Pie)</h4><div class="pie-chart" id="analyticsPieChart"></div><div class="pie-legend" id="analyticsPieLegend"></div></div>
                <div class="chart-box"><h4>📈 Income vs Expenses (Line)</h4><div class="line-chart-svg" id="analyticsLineChart"></div></div>
            </div>
            <div class="chart-row" style="margin-bottom:20px;">
                <div class="chart-box"><h4>🍩 Savings vs Spending (Donut)</h4><div class="donut-chart" id="analyticsDonutChart"><div class="center-text" id="analyticsDonutCenter">68%</div></div><div class="pie-legend" id="analyticsDonutLegend"></div></div>
                <div class="chart-box"><h4>📊 Category Comparison (Bar)</h4><div class="chart-container" id="categoryComparisonChart"></div></div>
                <div class="chart-box"><h4>📈 Monthly Growth (Area)</h4><div class="line-chart-svg" id="growthChart"></div></div>
            </div>
            <h4 style="margin-bottom:12px;">Monthly Spending Breakdown</h4>
            <div style="background:var(--light); border-radius:16px; padding:16px;" id="spendingBreakdown"></div>
            <div style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:20px;" id="analyticsDetails"></div>
        </div>
    </section>

</main>

<!-- ===== MODAL ===== -->
<div class="modal-overlay" id="modalOverlay">
    <div class="modal">
        <div class="modal-header"><h2 id="modalTitle">Confirm Transaction</h2><button class="modal-close" onclick="closeModal()">&times;</button></div>
        <p style="color:var(--muted); margin-bottom:16px;" id="modalDesc">Please verify the details before proceeding.</p>
        <div style="background:var(--light); border-radius:16px; padding:16px; margin-bottom:16px;" id="modalDetails">
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Amount</span><span style="font-weight:700;">₦120,000.00</span></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>To</span><span>John Okafor</span></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Account</span><span>1234567890</span></div>
        </div>
        <div class="form-group"><label>Transaction PIN</label><input type="password" maxlength="4" placeholder="****" id="modalPin" /></div>
        <div style="display:flex; gap:12px;">
            <button class="btn btn-gold" style="flex:1; justify-content:center;" onclick="confirmTransaction()">Confirm</button>
            <button class="btn btn-outline" style="flex:1; justify-content:center;" onclick="closeModal()">Cancel</button>
        </div>
        <div class="form-error" id="modalError">Please enter your PIN</div>
    </div>
</div>

<!-- ===== OTP VERIFICATION MODAL ===== -->
<div class="modal-overlay" id="otpModalOverlay">
    <div class="modal">
        <div class="modal-header"><h2><i class="fas fa-shield-alt" style="color:var(--gold);"></i> Verify This Transfer</h2><button class="modal-close" onclick="closeOtpModal()">&times;</button></div>
        <p style="color:var(--muted); margin-bottom:16px;" id="otpModalDesc">Our fraud detection engine flagged this transfer as unusual. Enter the OTP sent to your registered phone/email to continue.</p>
        <div style="background:var(--light); border-radius:16px; padding:16px; margin-bottom:16px;" id="otpModalDetails"></div>
        <div class="form-group"><label>One-Time Password (OTP)</label><input type="text" maxlength="6" placeholder="6-digit code" id="otpModalCode" /></div>
        <div style="display:flex; gap:12px;">
            <button class="btn btn-gold" style="flex:1; justify-content:center;" onclick="submitOtpVerification()">Verify & Send</button>
            <button class="btn btn-outline" style="flex:1; justify-content:center;" onclick="closeOtpModal()">Cancel</button>
        </div>
        <div class="form-error" id="otpModalError">Incorrect OTP, please try again.</div>
    </div>
</div>

<!-- ===== SCRIPTS ===== -->
<script>
    // ============================================================
    // BACKEND API CLIENT
    // ============================================================
    const API_BASE = '';

    function getToken() { return localStorage.getItem('stb_token'); }
    function setToken(t) { localStorage.setItem('stb_token', t); }
    function clearToken() { localStorage.removeItem('stb_token'); }

    async function api(path, options = {}) {
        const headers = Object.assign({ 'Content-Type': 'application/json' }, options.headers || {});
        const token = getToken();
        if (token) headers['Authorization'] = 'Bearer ' + token;

        let res;
        try {
            res = await fetch(API_BASE + path, Object.assign({}, options, { headers }));
        } catch (err) {
            throw new Error('Could not reach the server. Is sapphire_backend.py running?');
        }

        let data = null;
        try { data = await res.json(); } catch (_) {}

        if (!res.ok) {
            const detail = (data && data.detail) ? data.detail : ('Request failed (' + res.status + ')');
            const error = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
            error.status = res.status;
            error.data = data;
            throw error;
        }
        return data;
    }

    const apiGet = (path) => api(path, { method: 'GET' });
    const apiPost = (path, body) => api(path, { method: 'POST', body: body ? JSON.stringify(body) : undefined });

    // ============================================================
    // NAVIGATION
    // ============================================================
    function showLogin() {
        document.getElementById('welcomePage').style.display = 'none';
        document.getElementById('registerPage').classList.remove('active');
        document.getElementById('loginPage').classList.add('active');
    }

    function showRegister() {
        document.getElementById('welcomePage').style.display = 'none';
        document.getElementById('loginPage').classList.remove('active');
        document.getElementById('registerPage').classList.add('active');
    }

    function showWelcome() {
        document.getElementById('welcomePage').style.display = 'flex';
        document.getElementById('loginPage').classList.remove('active');
        document.getElementById('registerPage').classList.remove('active');
    }

    function showPage(pageId) {
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById(pageId).classList.add('active');
        document.querySelectorAll('.sidebar .nav a').forEach(l => {
            l.classList.remove('active');
            if (l.dataset.page === pageId.replace('page-', '')) l.classList.add('active');
        });
        if (typeof closeSidebar === 'function') closeSidebar();
    }

    // ============================================================
    // TOAST NOTIFICATIONS
    // ============================================================
    function showToast(message, type) {
        const container = document.getElementById('toastContainer');
        const icons = { success: 'fa-check-circle', error: 'fa-exclamation-circle', info: 'fa-info-circle' };
        const toast = document.createElement('div');
        toast.className = `toast ${type || 'info'}`;
        toast.innerHTML = `
            <span class="toast-icon"><i class="fas ${icons[type] || icons.info}" style="color:${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'}"></i></span>
            <span class="toast-msg">${message}</span>
        `;
        container.appendChild(toast);
        setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(100px)'; setTimeout(() => toast.remove(), 300); }, 3000);
    }

    // ============================================================
    // MODAL
    // ============================================================
    let modalCallback = null;

    function showModal(type) {
        const overlay = document.getElementById('modalOverlay');
        const title = document.getElementById('modalTitle');
        const desc = document.getElementById('modalDesc');
        const details = document.getElementById('modalDetails');

        document.getElementById('modalPin').closest('.form-group').style.display = '';

        if (type === 'transfer') {
            title.textContent = 'Confirm Transfer';
            desc.textContent = 'This transfer will be screened by our fraud detection engine.';
            details.innerHTML = `
                <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Amount</span><span style="font-weight:700;">₦${document.getElementById('internalAmount').value || '0'}</span></div>
                <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>To</span><span>${document.getElementById('internalAccount').value || 'Unknown'}</span></div>
                <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Bank</span><span>Sapphire Bank</span></div>
            `;
            modalCallback = function() { executeInternalTransfer(); };
        } else {
            title.textContent = 'Confirm Action';
            desc.textContent = 'Please verify the details before proceeding.';
            details.innerHTML = `<div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Action</span><span style="font-weight:700;">Confirm</span></div>`;
            modalCallback = function() { showToast('Action confirmed!', 'success'); };
        }

        document.getElementById('modalError').style.display = 'none';
        document.getElementById('modalPin').value = '';
        overlay.classList.add('show');
    }

    function closeModal() {
        document.getElementById('modalOverlay').classList.remove('show');
    }

    async function confirmTransaction() {
        const pin = document.getElementById('modalPin').value;
        if (!pin || pin.length !== 4) {
            document.getElementById('modalError').style.display = 'block';
            return;
        }
        document.getElementById('modalError').style.display = 'none';
        if (modalCallback) await modalCallback();
        closeModal();
    }

    document.getElementById('modalOverlay').addEventListener('click', function(e) {
        if (e.target === this) closeModal();
    });

    // ============================================================
    // VALIDATION & TRANSFER EXECUTION
    // ============================================================
    let pendingTransfer = null;

    async function validateTransfer(type) {
        let amount, pin;
        if (type === 'internal') {
            amount = document.getElementById('internalAmount').value;
            pin = document.getElementById('internalPin').value;
        } else if (type === 'other') {
            amount = document.getElementById('otherAmount').value;
            pin = document.getElementById('otherPin').value;
        } else {
            amount = document.getElementById('intlAmount').value;
            pin = document.getElementById('intlPin').value;
        }

        if (!amount || parseFloat(amount) <= 0) {
            showToast('Please enter a valid amount', 'error');
            return;
        }
        if (!pin || pin.length !== 4) {
            showToast('Please enter a valid 4-digit PIN', 'error');
            return;
        }

        if (type !== 'internal') {
            showToast('Other bank transfers are simulated in this version.', 'info');
            showModal('transfer');
            return;
        }

        const account = document.getElementById('internalAccount').value.trim();
        if (!account || account.length !== 10) {
            showToast('Please enter a valid 10-digit account number', 'error');
            return;
        }
        if (!getToken()) {
            showToast('Please sign in first', 'error');
            return;
        }

        try {
            showToast('Looking up account...', 'info');
            const recipient = await apiGet('/verify-account/' + encodeURIComponent(account));
            openInternalTransferConfirm({ account, amount: parseFloat(amount), pin, recipient });
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    function openInternalTransferConfirm({ account, amount, pin, recipient }) {
        const overlay = document.getElementById('modalOverlay');
        document.getElementById('modalTitle').textContent = 'Confirm Transfer';
        document.getElementById('modalDesc').textContent = 'This transfer will be screened by our fraud detection engine.';
        document.getElementById('modalDetails').innerHTML = `
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Amount</span><span style="font-weight:700;">₦${amount.toLocaleString()}</span></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>To</span><span>${recipient.account_name}</span></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Account</span><span>${recipient.account_number}</span></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Bank</span><span>${recipient.bank_name}</span></div>
        `;
        document.getElementById('modalPin').closest('.form-group').style.display = 'none';
        document.getElementById('modalError').style.display = 'none';

        modalCallback = async function() {
            await executeInternalTransfer(account, amount, pin);
        };
        overlay.classList.add('show');
    }

    async function executeInternalTransfer(account, amount, pin) {
        try {
            const result = await apiPost('/transfer', {
                recipient_account: account,
                amount: amount,
                transaction_pin: pin,
                description: document.getElementById('internalDesc').value || 'Transfer via Sapphire Bank app',
            });

            if (result.status === 'completed') {
                showToast(result.alert || 'Transfer successful!', 'success');
                await refreshAccountData();
            } else if (result.status === 'otp_required') {
                pendingTransfer = { reference: result.reference, amount, account };
                openOtpModal(result);
            } else if (result.status === 'blocked') {
                showToast(result.alert || 'Transfer blocked by fraud detection.', 'error');
            } else {
                showToast('Transfer submitted: ' + result.status, 'info');
            }
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    function openOtpModal(result) {
        document.getElementById('otpModalDetails').innerHTML = `
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Reference</span><span style="font-weight:700;">${result.reference}</span></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Risk classification</span><span>${result.classification || 'SUSPICIOUS'}</span></div>
            <div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Fraud score</span><span>${(result.ai_fraud_score ?? 0).toFixed ? (result.ai_fraud_score ?? 0).toFixed(3) : result.ai_fraud_score}</span></div>
            ${result.otp ? `<div style="display:flex; justify-content:space-between; padding:4px 0;"><span>Demo OTP</span><span style="font-weight:700; color:var(--gold);">${result.otp}</span></div>` : ''}
        `;
        document.getElementById('otpModalCode').value = '';
        document.getElementById('otpModalError').style.display = 'none';
        document.getElementById('otpModalOverlay').classList.add('show');
    }

    function closeOtpModal() {
        document.getElementById('otpModalOverlay').classList.remove('show');
        pendingTransfer = null;
    }

    async function submitOtpVerification() {
        const code = document.getElementById('otpModalCode').value.trim();
        if (!code) {
            document.getElementById('otpModalError').textContent = 'Please enter the OTP.';
            document.getElementById('otpModalError').style.display = 'block';
            return;
        }
        if (!pendingTransfer) {
            closeOtpModal();
            return;
        }
        try {
            const result = await apiPost('/transaction/verify-otp', {
                transaction_reference: pendingTransfer.reference,
                otp: code,
            });
            showToast(result.alert || 'Transfer completed!', 'success');
            closeOtpModal();
            await refreshAccountData();
        } catch (err) {
            document.getElementById('otpModalError').textContent = err.message;
            document.getElementById('otpModalError').style.display = 'block';
        }
    }

    // ============================================================
    // LOGIN / LOGOUT
    // ============================================================

    async function handleLogin() {
        const identifier = document.getElementById('loginEmail').value.trim();
        const pin = document.getElementById('loginPassword').value.trim();
        if (!identifier || !pin) {
            showToast('Enter your username/email and PIN', 'error');
            return;
        }

        try {
            const result = await apiPost('/auth/login', { username: identifier, login_pin: pin });
            setToken(result.access_token);

            document.getElementById('loginPage').classList.remove('active');
            document.getElementById('mainContent').style.display = 'block';
            document.getElementById('sidebar').style.display = 'flex';
            document.body.classList.add('logged-in');
            document.getElementById('userName').textContent = result.full_name || 'User';

            renderAllCharts();
            renderAlerts();
            renderCases();
            renderBeneficiaries();
            renderQuickBeneficiaries();
            renderCards();
            renderSavings();
            renderInvestments();
            renderLoans();
            renderRewards();
            renderSettings();
            renderAnalytics();

            showToast('Welcome back, ' + (result.full_name || '') + '!', 'success');
            await refreshAccountData();
            await loadSettingsData();
            await loadPayGuardData();
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    async function handleRegister() {
        const username = document.getElementById('regUsername').value.trim();
        const email = document.getElementById('regEmail').value.trim();
        const fullName = document.getElementById('regFullName').value.trim();
        const phone = document.getElementById('regPhone').value.trim();
        const accountType = document.getElementById('regAccountType').value;
        const accountCategory = document.getElementById('regAccountCategory').value;
        const loginPin = document.getElementById('regLoginPin').value;
        const transPin = document.getElementById('regTransPin').value;
        const terms = document.getElementById('regTerms').checked;
        
        const errorDiv = document.getElementById('registerError');
        const successDiv = document.getElementById('registerSuccess');
        errorDiv.style.display = 'none';
        successDiv.style.display = 'none';
        
        if (!username || !email || !fullName || !phone || !loginPin || !transPin) {
            errorDiv.textContent = 'Please fill all required fields';
            errorDiv.style.display = 'block';
            return;
        }
        if (loginPin.length !== 4 || !/^\\d+$/.test(loginPin)) {
            errorDiv.textContent = 'Login PIN must be exactly 4 digits';
            errorDiv.style.display = 'block';
            return;
        }
        if (transPin.length !== 4 || !/^\\d+$/.test(transPin)) {
            errorDiv.textContent = 'Transaction PIN must be exactly 4 digits';
            errorDiv.style.display = 'block';
            return;
        }
        if (!terms) {
            errorDiv.textContent = 'You must accept the Terms and Conditions';
            errorDiv.style.display = 'block';
            return;
        }
        
        try {
            const result = await apiPost('/auth/register', {
                username, email, full_name: fullName, phone_number: phone,
                login_pin: loginPin, transaction_pin: transPin,
                account_type: accountType, account_category: accountCategory,
                terms_accepted: terms
            });
            successDiv.textContent = result.alert;
            successDiv.style.display = 'block';
            showToast('Account created! Your account number: ' + result.account_number, 'success');
            setTimeout(() => { showLogin(); }, 3000);
        } catch (err) {
            errorDiv.textContent = err.message;
            errorDiv.style.display = 'block';
        }
    }

    function handleLogout() {
        clearToken();
        document.getElementById('welcomePage').style.display = 'flex';
        document.getElementById('loginPage').classList.remove('active');
        document.getElementById('mainContent').style.display = 'none';
        document.getElementById('sidebar').style.display = 'none';
        document.body.classList.remove('logged-in');
        closeSidebar();
        showToast('Logged out successfully', 'info');
    }

    // ============================================================
    // DATA RENDER FUNCTIONS
    // ============================================================
    let transactions = [];
    let currentPage = 1;
    const itemsPerPage = 8;

    const categoryData = [
        { name: 'Food & Dining', amount: 185000, pct: 34, color: '#f59e0b' },
        { name: 'Transport', amount: 72000, pct: 13, color: '#3b82f6' },
        { name: 'Utilities', amount: 98000, pct: 18, color: '#8b5cf6' },
        { name: 'Entertainment', amount: 45000, pct: 8, color: '#ec4899' },
        { name: 'Shopping', amount: 85000, pct: 16, color: '#10b981' },
        { name: 'Others', amount: 55000, pct: 11, color: '#6366f1' }
    ];

    const monthlyData = [
        { month: 'Jan', spend: 420000, income: 580000 },
        { month: 'Feb', spend: 450000, income: 560000 },
        { month: 'Mar', spend: 480000, income: 620000 },
        { month: 'Apr', spend: 510000, income: 650000 },
        { month: 'May', spend: 540000, income: 710000 },
        { month: 'Jun', spend: 560000, income: 680000 }
    ];

    const beneficiaries = [
        'John Okafor · 1234567890', 'Mary Jane · 0987654321', 'Tech Corp · 1122334455',
        'Sarah Lee · 5566778899', 'James Bond · 6677889900', 'Grace Adeyemi · 7788990011',
        'David Okonkwo · 8899001122', 'Chioma Eze · 9900112233', 'Kofi Mensah · 0011223344'
    ];

    const cards = [
        { type: 'Virtual', last4: '1234', exp: '12/28', limit: '₦5,000,000', status: 'Active' },
        { type: 'Debit', last4: '5678', exp: '06/29', limit: '₦3,000,000', status: 'Active' },
        { type: 'Credit', last4: '9012', exp: '03/30', limit: '₦2,000,000', status: 'Pending' },
        { type: 'Corporate', last4: '3456', exp: '09/27', limit: '₦10,000,000', status: 'Active' }
    ];

    const savings = [
        { name: 'Flexible Savings', amount: 1250000, rate: '4.5% p.a.', status: 'Active', interest: '₦4,687' },
        { name: 'Fixed Deposit', amount: 2000000, rate: '7.0% p.a.', status: 'Pending', interest: '₦11,667' },
        { name: 'Target Savings', amount: 500000, rate: '5.0% p.a.', status: 'On Track', interest: '50% complete' }
    ];

    const investments = [
        { name: 'Stocks', amount: 850000, change: '+5.2%', detail: '12 holdings' },
        { name: 'Bonds', amount: 1200000, change: '+3.8%', detail: '8 bonds' },
        { name: 'Real Estate', amount: 3100000, change: '+7.1%', detail: '2 properties' },
        { name: 'Mutual Funds', amount: 680000, change: '+4.3%', detail: '5 funds' },
        { name: 'Crypto', amount: 320000, change: '-2.1%', detail: 'BTC, ETH, SOL' },
        { name: 'Precious Metals', amount: 450000, change: '+6.7%', detail: 'Gold, Silver' }
    ];

    const loans = [
        { name: 'Personal Loan', max: '₦5,000,000', rate: '12% p.a.' },
        { name: 'Business Loan', max: '₦20,000,000', rate: '9% p.a.' },
        { name: 'Home Loan', max: '₦50,000,000', rate: '7.5% p.a.' },
        { name: 'Auto Loan', max: '₦8,000,000', rate: '10% p.a.' }
    ];

    const offers = [
        '🛒 5% off at ShopRite', '⛽ ₦200 off fuel', '🍕 10% off Domino\'s',
        '✈️ 2x points on travel', '🎬 15% off Netflix', '☕ Buy 1 Get 1 at Starbucks'
    ];

    function renderBarChart(containerId, data, colors) {
        const container = document.getElementById(containerId);
        if (!container) return;
        const max = Math.max(...data.map(d => d.value), 1);
        container.innerHTML = data.map((d, i) => {
            const height = (d.value / max) * 160;
            const color = colors ? colors[i % colors.length] : '#2563eb';
            return `<div class="chart-bar" style="height:${height}px; background:${color}; flex:1; min-width:20px;">
                <span class="val">${d.value.toLocaleString()}</span>
                <span class="label">${d.label}</span>
            </div>`;
        }).join('');
    }

    function renderPieChart(containerId, legendId, data) {
        const container = document.getElementById(containerId);
        const legend = document.getElementById(legendId);
        if (!container || !legend) return;
        const total = data.reduce((sum, d) => sum + d.value, 0);
        if (total === 0) { container.style.background = '#e9edf2'; legend.innerHTML = '<span>No data</span>'; return; }
        let gradient = '', start = 0;
        data.forEach((d) => {
            const pct = (d.value / total) * 100;
            gradient += `${d.color} ${start}% ${start + pct}%, `;
            start += pct;
        });
        gradient = gradient.slice(0, -2);
        container.style.background = `conic-gradient(${gradient})`;
        legend.innerHTML = data.map(d =>
            `<span><span class="dot" style="background:${d.color};"></span> ${d.label} (${d.value.toLocaleString()})</span>`
        ).join('');
    }

    function renderLineChart(containerId, data, label, color) {
        const container = document.getElementById(containerId);
        if (!container) return;
        const max = Math.max(...data.map(d => d.value), 1);
        const points = data.map((d, i) => {
            const x = (i / (data.length - 1)) * 100;
            const y = 100 - (d.value / max) * 90;
            return `${x},${y}`;
        }).join(' ');
        const polyPoints = data.map((d, i) => {
            const x = (i / (data.length - 1)) * 100;
            const y = 100 - (d.value / max) * 90;
            return `${x},${y}`;
        }).join(' ');
        container.innerHTML = `
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" style="width:100%; height:100%;">
                <polyline points="${points}" fill="none" stroke="${color || '#2563eb'}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <polygon points="${polyPoints} 100,100 0,100" fill="${color || '#2563eb'}20" opacity="0.3"/>
                ${data.map((d, i) => {
                    const x = (i / (data.length - 1)) * 100;
                    const y = 100 - (d.value / max) * 90;
                    return `<circle cx="${x}" cy="${y}" r="2.5" fill="${color || '#2563eb'}" stroke="white" stroke-width="1"/>`;
                }).join('')}
            </svg>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:4px;">
                ${data.map(d => `<span>${d.label}</span>`).join('')}
            </div>
        `;
    }

    function renderDonutChart(containerId, centerId, legendId, data, total) {
        const container = document.getElementById(containerId);
        const center = document.getElementById(centerId);
        const legend = document.getElementById(legendId);
        if (!container) return;
        if (total === 0) { container.style.background = '#e9edf2'; if (center) center.textContent = '0%'; return; }
        let gradient = '', start = 0;
        data.forEach((d) => {
            const pct = (d.value / total) * 100;
            gradient += `${d.color} ${start}% ${start + pct}%, `;
            start += pct;
        });
        gradient = gradient.slice(0, -2);
        container.style.background = `conic-gradient(${gradient})`;
        if (center) center.textContent = Math.round((data[0].value / total) * 100) + '%';
        if (legend) {
            legend.innerHTML = data.slice(0, 3).map(d =>
                `<span><span class="dot" style="background:${d.color};"></span> ${d.label}</span>`
            ).join('');
        }
    }

    function renderAreaChart(containerId, data, color) {
        const container = document.getElementById(containerId);
        if (!container) return;
        const max = Math.max(...data.map(d => d.value), 1);
        const points = data.map((d, i) => {
            const x = (i / (data.length - 1)) * 100;
            const y = 100 - (d.value / max) * 90;
            return `${x},${y}`;
        }).join(' ');
        container.innerHTML = `
            <svg viewBox="0 0 100 100" preserveAspectRatio="none" style="width:100%; height:100%;">
                <defs><linearGradient id="areaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:${color || '#3b82f6'};stop-opacity:0.4" />
                    <stop offset="100%" style="stop-color:${color || '#3b82f6'};stop-opacity:0.05" />
                </linearGradient></defs>
                <polyline points="${points}" fill="none" stroke="${color || '#3b82f6'}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <polygon points="${points} 100,100 0,100" fill="url(#areaGrad)"/>
                ${data.map((d, i) => {
                    const x = (i / (data.length - 1)) * 100;
                    const y = 100 - (d.value / max) * 90;
                    return `<circle cx="${x}" cy="${y}" r="2" fill="${color || '#3b82f6'}" stroke="white" stroke-width="1"/>`;
                }).join('')}
            </svg>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:4px;">
                ${data.map(d => `<span>${d.label}</span>`).join('')}
            </div>
        `;
    }

    function renderAllCharts() {
        renderBarChart('barChart', monthlyData.map(d => ({ label: d.month, value: d.spend })), ['#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#6366f1']);
        renderPieChart('pieChart', 'pieLegend', categoryData.map(d => ({ label: d.name, value: d.amount, color: d.color })));
        const balanceData = monthlyData.map(d => ({ label: d.month, value: d.income - d.spend + 200000 }));
        renderLineChart('lineChart', balanceData, 'Balance', '#10b981');
        renderBarChart('analyticsBarChart', monthlyData.map(d => ({ label: d.month, value: d.spend })), ['#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#6366f1']);
        renderPieChart('analyticsPieChart', 'analyticsPieLegend', categoryData.map(d => ({ label: d.name, value: d.amount, color: d.color })));
        const analyticsDonutData = [
            { label: 'Needs', value: 280000, color: '#3b82f6' },
            { label: 'Wants', value: 160000, color: '#f59e0b' },
            { label: 'Savings', value: 100000, color: '#10b981' }
        ];
        renderDonutChart('analyticsDonutChart', 'analyticsDonutCenter', 'analyticsDonutLegend', analyticsDonutData, 540000);
        renderBarChart('categoryComparisonChart', categoryData.map(d => ({ label: d.name, value: d.amount })), categoryData.map(d => d.color));
        const growthData = monthlyData.map(d => ({ label: d.month, value: d.income - d.spend }));
        renderAreaChart('growthChart', growthData, '#10b981');
        const incomeData = monthlyData.map(d => ({ label: d.month, value: d.income }));
        const expenseData = monthlyData.map(d => ({ label: d.month, value: d.spend }));
        const container = document.getElementById('analyticsLineChart');
        if (container) {
            const max = Math.max(...monthlyData.map(d => Math.max(d.income, d.spend)), 1);
            const points1 = incomeData.map((d, i) => {
                const x = (i / (incomeData.length - 1)) * 100;
                const y = 100 - (d.value / max) * 90;
                return `${x},${y}`;
            }).join(' ');
            const points2 = expenseData.map((d, i) => {
                const x = (i / (expenseData.length - 1)) * 100;
                const y = 100 - (d.value / max) * 90;
                return `${x},${y}`;
            }).join(' ');
            container.innerHTML = `
                <svg viewBox="0 0 100 100" preserveAspectRatio="none" style="width:100%; height:100%;">
                    <polyline points="${points1}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    <polyline points="${points2}" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                    ${incomeData.map((d, i) => {
                        const x = (i / (incomeData.length - 1)) * 100;
                        const y = 100 - (d.value / max) * 90;
                        return `<circle cx="${x}" cy="${y}" r="2.5" fill="#10b981" stroke="white" stroke-width="1"/>`;
                    }).join('')}
                    ${expenseData.map((d, i) => {
                        const x = (i / (expenseData.length - 1)) * 100;
                        const y = 100 - (d.value / max) * 90;
                        return `<circle cx="${x}" cy="${y}" r="2.5" fill="#ef4444" stroke="white" stroke-width="1"/>`;
                    }).join('')}
                </svg>
                <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:4px;">
                    ${incomeData.map(d => `<span>${d.label}</span>`).join('')}
                </div>
                <div style="display:flex; gap:16px; justify-content:center; font-size:11px; margin-top:4px;">
                    <span><span style="display:inline-block; width:12px; height:3px; background:#10b981; margin-right:4px;"></span> Income</span>
                    <span><span style="display:inline-block; width:12px; height:3px; background:#ef4444; margin-right:4px;"></span> Expenses</span>
                </div>
            `;
        }
    }

    function renderRecentTransactions() {
        const container = document.getElementById('recentTransactions');
        container.innerHTML = transactions.slice(0, 8).map(t => {
            const isCredit = t.type === 'Credit';
            return `<div class="txn-item">
                <div class="left">
                    <div class="icon"><i class="fas ${isCredit ? 'fa-arrow-down' : 'fa-arrow-up'}"></i></div>
                    <div class="info">
                        <div class="name">${t.desc || 'Transaction'}</div>
                        <div class="meta">${t.date || new Date().toLocaleDateString()}</div>
                    </div>
                </div>
                <div class="amount ${isCredit ? 'credit' : 'debit'}">${isCredit ? '+' : '-'}₦${Math.abs(t.amount || 0).toLocaleString()}</div>
            </div>`;
        }).join('');
    }

    function renderAlerts() {
        const container = document.getElementById('alertFeed');
        const alerts = [
            { id: 'TXN-001', amount: 450000, type: 'Fraud' },
            { id: 'TXN-002', amount: 120000, type: 'Suspicious' },
            { id: 'TXN-003', amount: 50000, type: 'Safe' },
        ];
        container.innerHTML = alerts.map(a => {
            const icon = a.type === 'Fraud' ? 'fa-exclamation-triangle' : a.type === 'Suspicious' ? 'fa-exclamation-circle' : 'fa-check-circle';
            const color = a.type === 'Fraud' ? '#ef4444' : a.type === 'Suspicious' ? '#f59e0b' : '#10b981';
            const badgeClass = a.type === 'Fraud' ? 'badge' : a.type === 'Suspicious' ? 'badge yellow' : 'badge green';
            return `<div class="alert-item">
                <span><i class="fas ${icon}" style="color:${color};"></i> ${a.id} · ₦${a.amount.toLocaleString()}</span>
                <span class="${badgeClass}">${a.type}</span>
            </div>`;
        }).join('');
    }

    function renderCases() {
        const container = document.getElementById('recentCases');
        const cases = [
            { id: '#C-001', status: 'Fraud', amount: 450000, time: '2h ago' },
            { id: '#C-002', status: 'Suspicious', amount: 120000, time: '4h ago' },
        ];
        container.innerHTML = cases.map(c => {
            const statusClass = c.status === 'Fraud' ? 'fraud' : c.status === 'Suspicious' ? 'suspicious' : 'safe';
            return `<div style="display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #f0f2f6;">
                <span>${c.id}</span>
                <span class="status ${statusClass}">${c.status}</span>
                <span>₦${c.amount.toLocaleString()}</span>
                <span class="text-muted">${c.time}</span>
            </div>`;
        }).join('');
    }

    function renderBeneficiaries() {
        const container = document.getElementById('beneficiaryList');
        container.innerHTML = beneficiaries.map(b => {
            return `<div class="stat-pill" style="padding:16px; background:var(--light); width:100%;"><i class="fas fa-user"></i> ${b}</div>`;
        }).join('');
    }

    function renderQuickBeneficiaries() {
        const container = document.getElementById('quickBeneficiaries');
        container.innerHTML = beneficiaries.slice(0, 7).map(b => {
            return `<span class="stat-pill"><i class="fas fa-user"></i> ${b.split('·')[0].trim()}</span>`;
        }).join('');
    }

    function renderTransactions(page) {
        const container = document.getElementById('transactionTableBody');
        const start = (page - 1) * itemsPerPage;
        const end = start + itemsPerPage;
        const pageData = transactions.slice(start, end);
        container.innerHTML = pageData.map(t => {
            const statusClass = t.status === 'Completed' ? 'success' : t.status === 'Pending' ? 'pending' : 'failed';
            const isCredit = t.type === 'Credit';
            const amountClass = isCredit ? 'credit' : 'debit';
            return `<tr>
                <td>${t.ref || 'N/A'}</td>
                <td>${t.date || '-'}</td>
                <td>${t.desc || '-'}</td>
                <td>${t.type || '-'}</td>
                <td class="${amountClass}">${isCredit ? '+' : '-'}₦${Math.abs(t.amount || 0).toLocaleString()}</td>
                <td><span class="status ${statusClass}">${t.status || 'Completed'}</span></td>
                <td><button class="btn btn-sm btn-outline" onclick="showToast('Viewing transaction', 'info')">View</button></td>
            </tr>`;
        }).join('');
        currentPage = page;
    }

    function pageTransactions(page) {
        const totalPages = Math.ceil(transactions.length / itemsPerPage);
        if (page < 1 || page > totalPages) return;
        renderTransactions(page);
        document.querySelectorAll('#transactionPagination button').forEach((btn, i) => {
            btn.classList.toggle('active', i === page - 1);
        });
    }

    function filterTransactions(type) {
        const btn = event.target;
        document.querySelectorAll('#page-transactions .btn-sm').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        let filtered = transactions;
        if (type === 'credit') filtered = transactions.filter(t => t.type === 'Credit');
        else if (type === 'debit') filtered = transactions.filter(t => t.type === 'Debit');
        else if (type === 'month') filtered = transactions.filter(t => t.date && t.date.startsWith('2026-06'));
        const container = document.getElementById('transactionTableBody');
        container.innerHTML = filtered.slice(0, 8).map(t => {
            const statusClass = t.status === 'Completed' ? 'success' : t.status === 'Pending' ? 'pending' : 'failed';
            const isCredit = t.type === 'Credit';
            const amountClass = isCredit ? 'credit' : 'debit';
            return `<tr>
                <td>${t.ref || 'N/A'}</td>
                <td>${t.date || '-'}</td>
                <td>${t.desc || '-'}</td>
                <td>${t.type || '-'}</td>
                <td class="${amountClass}">${isCredit ? '+' : '-'}₦${Math.abs(t.amount || 0).toLocaleString()}</td>
                <td><span class="status ${statusClass}">${t.status || 'Completed'}</span></td>
                <td><button class="btn btn-sm btn-outline" onclick="showToast('Viewing transaction', 'info')">View</button></td>
            </tr>`;
        }).join('');
    }

    function renderCards() {
        const container = document.getElementById('cardList');
        container.innerHTML = cards.map(c => {
            const statusClass = c.status === 'Active' ? 'safe' : 'pending';
            const bgColors = ['var(--primary)', '#1e3550', '#2a1e3c', '#1a2a1f'];
            const bg = bgColors[cards.indexOf(c) % bgColors.length];
            return `<div style="background:${bg}; border-radius:24px; padding:24px; color:white; position:relative; overflow:hidden;">
                <div style="position:absolute; top:-20px; right:-20px; font-size:80px; opacity:0.05;"><i class="fas fa-credit-card"></i></div>
                <div style="display:flex; justify-content:space-between;"><span>💳 ${c.type}</span><span style="color:var(--gold);">● ● ● ●</span></div>
                <div style="font-size:22px; font-weight:600; margin:20px 0 8px;">**** ${c.last4}</div>
                <div style="display:flex; justify-content:space-between;"><span>Exp ${c.exp}</span><span>CVV ***</span></div>
                <div style="margin-top:12px;"><span class="status ${statusClass}">${c.status}</span> <span style="color:var(--gold);margin-left:12px;">${c.limit}</span></div>
            </div>`;
        }).join('');
    }

    function renderSavings() {
        const container = document.getElementById('savingsList');
        container.innerHTML = savings.map(s => {
            const statusClass = s.status === 'Active' ? 'success' : 'pending';
            return `<div style="background:var(--light); border-radius:20px; padding:20px;">
                <h4>${s.name}</h4>
                <div class="value" style="font-size:28px;">₦${s.amount.toLocaleString()}</div>
                <div class="stat-row"><span class="change">${s.rate}</span></div>
                <div style="margin-top:12px;"><span class="status ${statusClass}">${s.status}</span></div>
                <div style="margin-top:8px; font-size:13px; color:var(--muted);">Interest earned: ${s.interest}</div>
                ${s.name === 'Target Savings' ? `<div class="progress-bar"><div class="fill" style="width:50%;"></div></div>` : ''}
            </div>`;
        }).join('');
    }

    function renderInvestments() {
        const container = document.getElementById('investmentList');
        container.innerHTML = investments.map(i => {
            const changeClass = i.change.startsWith('+') ? 'change' : 'change down';
            return `<div style="background:var(--light); border-radius:20px; padding:18px; text-align:center;">
                <h4>${i.name}</h4>
                <div class="value" style="font-size:22px;">₦${i.amount.toLocaleString()}</div>
                <span class="${changeClass}">${i.change}</span>
                <div style="font-size:12px; color:var(--muted); margin-top:4px;">${i.detail}</div>
            </div>`;
        }).join('');
    }

    function renderLoans() {
        const container = document.getElementById('loanList');
        container.innerHTML = loans.map(l => {
            return `<div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border); align-items:center; flex-wrap:wrap; gap:8px;">
                <span><strong>${l.name}</strong><br><span style="font-size:13px; color:var(--muted);">Up to ${l.max}</span></span>
                <span style="font-weight:600;">${l.rate}</span>
                <button class="btn btn-primary btn-sm" onclick="showToast('${l.name} application started', 'info')">Apply</button>
            </div>`;
        }).join('');
    }

    function renderRewards() {
        const container = document.getElementById('rewardsStats');
        container.innerHTML = `
            <div style="background:var(--light); border-radius:20px; padding:18px; text-align:center;">
                <i class="fas fa-star" style="color:var(--gold); font-size:28px;"></i>
                <h4 style="margin:8px 0;">Points</h4>
                <div class="value" style="font-size:24px;">12,450</div>
                <span class="change">+120 this month</span>
            </div>
            <div style="background:var(--light); border-radius:20px; padding:18px; text-align:center;">
                <i class="fas fa-gift" style="color:var(--gold); font-size:28px;"></i>
                <h4 style="margin:8px 0;">Offers</h4>
                <div class="value" style="font-size:24px;">8</div>
                <span class="change">2 new</span>
            </div>
            <div style="background:var(--light); border-radius:20px; padding:18px; text-align:center;">
                <i class="fas fa-coins" style="color:var(--gold); font-size:28px;"></i>
                <h4 style="margin:8px 0;">Cashback</h4>
                <div class="value" style="font-size:24px;">₦2,340</div>
                <span class="change">+₦340</span>
            </div>
        `;
        const offersContainer = document.getElementById('offersList');
        offersContainer.innerHTML = offers.map(o => `<span class="stat-pill" onclick="showToast('Offer redeemed: ${o}', 'success')" style="cursor:pointer;">${o}</span>`).join('');
    }

    function renderSettings() {
        const container = document.getElementById('settingsList');
        container.innerHTML = `
            <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border); align-items:center; flex-wrap:wrap; gap:8px;">
                <span><strong>Two-Factor Authentication</strong><br><span style="font-size:13px; color:var(--muted);">Add extra security layer</span></span>
                <span class="stat-pill" style="background:#10b981; color:white;">Enabled</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border); align-items:center; flex-wrap:wrap; gap:8px;">
                <span><strong>Transaction Alerts</strong><br><span style="font-size:13px; color:var(--muted);">Receive SMS & email alerts</span></span>
                <span class="stat-pill" style="background:#10b981; color:white;">On</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border); align-items:center; flex-wrap:wrap; gap:8px;">
                <span><strong>Device Management</strong><br><span style="font-size:13px; color:var(--muted);">Manage trusted devices</span></span>
                <span class="stat-pill">2 active</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border); align-items:center; flex-wrap:wrap; gap:8px;">
                <span><strong>Biometric Login</strong><br><span style="font-size:13px; color:var(--muted);">Face ID / Fingerprint</span></span>
                <span class="stat-pill" style="background:#10b981; color:white;">Enabled</span>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 0; align-items:center; flex-wrap:wrap; gap:8px;">
                <span><strong>Session Timeout</strong><br><span style="font-size:13px; color:var(--muted);">Auto-logout after inactivity</span></span>
                <span class="stat-pill">15 minutes</span>
            </div>
        `;
    }

    function renderAnalytics() {
        const statsContainer = document.getElementById('analyticsStats');
        statsContainer.innerHTML = `
            <div style="background:var(--light); border-radius:16px; padding:16px; text-align:center;">
                <h4>Total Spent</h4>
                <div class="value" style="font-size:24px;">₦540,000</div>
                <span class="change down"><i class="fas fa-arrow-down"></i> 2.1%</span>
                <div style="font-size:13px; color:var(--muted);">vs last month</div>
            </div>
            <div style="background:var(--light); border-radius:16px; padding:16px; text-align:center;">
                <h4>Avg. Daily Spend</h4>
                <div class="value" style="font-size:24px;">₦18,000</div>
                <span class="change"><i class="fas fa-arrow-up"></i> 3.5%</span>
                <div style="font-size:13px; color:var(--muted);">vs last month</div>
            </div>
            <div style="background:var(--light); border-radius:16px; padding:16px; text-align:center;">
                <h4>Savings Rate</h4>
                <div class="value" style="font-size:24px;">32%</div>
                <span class="change"><i class="fas fa-arrow-up"></i> 5.2%</span>
                <div style="font-size:13px; color:var(--muted);">of income saved</div>
            </div>
        `;
        const breakdown = document.getElementById('spendingBreakdown');
        breakdown.innerHTML = categoryData.map(c => `
            <div style="display:flex; justify-content:space-between; padding:6px 0;">
                <span>${c.name}</span>
                <span>₦${c.amount.toLocaleString()} (${c.pct}%)</span>
            </div>
            <div class="progress-bar"><div class="fill" style="width:${c.pct}%; background:${c.color};"></div></div>
        `).join('');
        const details = document.getElementById('analyticsDetails');
        details.innerHTML = `
            <div style="background:var(--light); border-radius:16px; padding:16px;">
                <h4>Top Spending Categories</h4>
                <div style="margin-top:8px;">
                    ${categoryData.slice(0, 5).map((c, i) => `
                        <div style="display:flex; justify-content:space-between; padding:4px 0;">
                            <span>${i+1}. ${c.name}</span>
                            <span style="font-weight:600;">₦${c.amount.toLocaleString()}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            <div style="background:var(--light); border-radius:16px; padding:16px;">
                <h4>Monthly Trends</h4>
                <div style="margin-top:8px;">
                    ${monthlyData.map(m => `
                        <div style="display:flex; justify-content:space-between; padding:4px 0;">
                            <span>${m.month}</span>
                            <span style="font-weight:600;">₦${m.spend.toLocaleString()}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    async function refreshAccountData() {
        try {
            const balanceRes = await apiGet('/balance');
            const txRes = await apiGet('/transactions?limit=50');
            const profileRes = await apiGet('/profile');
            const accountRes = await apiGet('/account-number');

            const now = new Date();
            let income = 0, expenses = 0;
            (txRes.transactions || []).forEach(t => {
                if (!t.created_at) return;
                const d = new Date(t.created_at);
                if (d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear()) {
                    if (t.direction === 'received') income += t.amount; else expenses += t.amount;
                }
            });

            document.getElementById('totalBalance').textContent = `₦${(balanceRes.balance || 0).toLocaleString()}`;
            document.getElementById('totalIncome').textContent = `₦${income.toLocaleString()}`;
            document.getElementById('totalExpenses').textContent = `₦${expenses.toLocaleString()}`;
            document.getElementById('accountNumber').textContent = `Account: ${accountRes.account_number || 'N/A'}`;

            transactions = (txRes.transactions || []).map(t => ({
                ref: t.reference,
                date: t.created_at ? t.created_at.split('T')[0] : '',
                desc: t.description || t.counterparty || (t.type || 'Transaction'),
                type: t.direction === 'received' ? 'Credit' : 'Debit',
                amount: t.amount || 0,
                status: (t.status || 'completed').replace(/_/g, ' ').charAt(0).toUpperCase() + (t.status || 'completed').replace(/_/g, ' ').slice(1),
            }));

            renderRecentTransactions();
            renderTransactions(1);
        } catch (err) {
            // Use demo data if backend not available
            transactions = [
                { ref: 'TXN-001', date: '2026-06-30', desc: 'Transfer to John Okafor', type: 'Debit', amount: -120000, status: 'Completed' },
                { ref: 'TXN-002', date: '2026-06-30', desc: 'Salary Credit', type: 'Credit', amount: 350000, status: 'Completed' },
                { ref: 'TXN-003', date: '2026-06-30', desc: 'Netflix Payment', type: 'Debit', amount: -5800, status: 'Completed' },
            ];
            renderRecentTransactions();
            renderTransactions(1);
        }
    }

    async function loadSettingsData() {
        try {
            const settings = await apiGet('/settings/get');
            const twoFa = await apiGet('/security/get-2fa');
            const alerts = await apiGet('/security/get-alerts');
            document.getElementById('settingsList').innerHTML = `
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border);">
                    <span><strong>Two-Factor Authentication</strong><br><span style="font-size:13px; color:var(--muted);">Add extra security layer</span></span>
                    <span class="stat-pill" style="background:${twoFa.enabled ? '#10b981' : '#ef4444'}; color:white;">${twoFa.enabled ? 'Enabled' : 'Disabled'}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid var(--border);">
                    <span><strong>SMS Alerts</strong><br><span style="font-size:13px; color:var(--muted);">Transaction alerts via SMS</span></span>
                    <span class="stat-pill" style="background:${alerts.sms ? '#10b981' : '#94a3b8'}; color:white;">${alerts.sms ? 'On' : 'Off'}</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding:8px 0;">
                    <span><strong>Theme</strong><br><span style="font-size:13px; color:var(--muted);">Display preference</span></span>
                    <span class="stat-pill">${settings.theme || 'light'}</span>
                </div>
            `;
        } catch (err) {
            // Use default settings
        }
    }

    async function loadPayGuardData() {
        try {
            const res = await apiGet('/admin/quarantine-list');
            const list = res.quarantine || [];
            const feedContainer = document.getElementById('alertFeed');
            if (feedContainer) {
                feedContainer.innerHTML = list.slice(0, 10).map(q => {
                    const cls = q.classification || 'SUSPICIOUS';
                    const icon = cls === 'FRAUDULENT' ? 'fa-exclamation-triangle' : cls === 'SUSPICIOUS' ? 'fa-exclamation-circle' : 'fa-check-circle';
                    const color = cls === 'FRAUDULENT' ? '#ef4444' : cls === 'SUSPICIOUS' ? '#f59e0b' : '#10b981';
                    const badgeClass = cls === 'FRAUDULENT' ? 'badge' : cls === 'SUSPICIOUS' ? 'badge yellow' : 'badge green';
                    return `<div class="alert-item">
                        <span><i class="fas ${icon}" style="color:${color};"></i> ${q.user_name || 'User'} · ₦${(q.amount || 0).toLocaleString()}</span>
                        <span class="${badgeClass}">${cls}</span>
                    </div>`;
                }).join('') || '<p style="color:var(--muted); font-size:14px;">No flagged transactions yet.</p>';
            }
        } catch (err) {
            // Use demo alerts
            renderAlerts();
        }
    }

    // ============================================================
    // MOBILE SIDEBAR DRAWER
    // ============================================================
    function toggleSidebar() {
        document.getElementById('sidebar').classList.toggle('mobile-open');
        document.getElementById('sidebarOverlay').classList.toggle('show');
    }

    function closeSidebar() {
        document.getElementById('sidebar').classList.remove('mobile-open');
        document.getElementById('sidebarOverlay').classList.remove('show');
    }

    // ============================================================
    // NAVIGATION
    // ============================================================
    const links = document.querySelectorAll('.sidebar .nav a');
    const pages = document.querySelectorAll('.page');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            links.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            const pageId = this.dataset.page;
            pages.forEach(p => p.classList.remove('active'));
            const target = document.getElementById('page-' + pageId);
            if (target) target.classList.add('active');
            closeSidebar();
        });
    });

    function switchTab(el, tabId) {
        const parent = el.closest('.card');
        const tabs = parent.querySelectorAll('.tab');
        const contents = parent.querySelectorAll('.tab-content');
        tabs.forEach(t => t.classList.remove('active'));
        el.classList.add('active');
        contents.forEach(c => c.classList.remove('active'));
        const target = document.getElementById(tabId);
        if (target) target.classList.add('active');
    }

    // ============================================================
    // INITIALIZATION - Check if backend is available
    // ============================================================
    async function initApp() {
        try {
            const test = await fetch('/balance');
            if (test.ok) {
                console.log('✅ Backend connected');
                // If we have a token, load data
                if (getToken()) {
                    document.getElementById('mainContent').style.display = 'block';
                    document.getElementById('sidebar').style.display = 'flex';
                    document.body.classList.add('logged-in');
                    await refreshAccountData();
                    await loadSettingsData();
                    await loadPayGuardData();
                    renderAllCharts();
                }
            }
        } catch (err) {
            console.log('ℹ️ Running with demo data (backend not available)');
            // Show the app with demo data anyway
            document.getElementById('loginPage').classList.remove('active');
            document.getElementById('welcomePage').style.display = 'none';
            document.getElementById('mainContent').style.display = 'block';
            document.getElementById('sidebar').style.display = 'flex';
            document.body.classList.add('logged-in');
            document.getElementById('userName').textContent = 'Demo User';
            
            renderAllCharts();
            renderAlerts();
            renderCases();
            renderBeneficiaries();
            renderQuickBeneficiaries();
            renderCards();
            renderSavings();
            renderInvestments();
            renderLoans();
            renderRewards();
            renderSettings();
            renderAnalytics();
            
            transactions = [
                { ref: 'TXN-001', date: '2026-06-30', desc: 'Transfer to John Okafor', type: 'Debit', amount: -120000, status: 'Completed' },
                { ref: 'TXN-002', date: '2026-06-30', desc: 'Salary Credit', type: 'Credit', amount: 350000, status: 'Completed' },
                { ref: 'TXN-003', date: '2026-06-30', desc: 'Netflix Payment', type: 'Debit', amount: -5800, status: 'Completed' },
                { ref: 'TXN-004', date: '2026-06-28', desc: 'Mary Jane', type: 'Credit', amount: 25000, status: 'Completed' },
            ];
            renderRecentTransactions();
            renderTransactions(1);
        }
    }

    initApp();
    console.log('💎 Sapphire Bank · All-in-One App');
</script>
</body>
</html>
"""

# ============================================================
# SERVE FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(FRONTEND_HTML)

@app.get("/{path:path}")
async def serve_frontend_path(path: str):
    return HTMLResponse(FRONTEND_HTML)

# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("💎 SAPPHIRE BANK - ALL-IN-ONE APPLICATION")
    print("="*70)
    print("\n📌 FEATURES:")
    print("   • AI Fraud Detection (Scanner AI + Random Forest + XGBoost + Neural Network)")
    print("   • SHAP Explainability for AI predictions")
    print("   • Fuzzy Logic Aggregator for final decision")
    print("   • OTP Verification for suspicious transactions")
    print("   • Quarantine System with Admin Dashboard")
    print("   • Tier System (1→2→3 with transaction limits)")
    print("   • Complete Banking API (Transfers, Airtime, Data)")
    print("   • PDF Receipt Generation")
    print("   • 2FA & Security Center")
    print("   • Dark/Light Theme Support")
    print("="*70)
    print(f"📍 Server: http://localhost:8000")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print("\n👑 ADMIN ACCOUNT:")
    print("   👤 Username: admin")
    print("   📞 Phone: 08000000000")
    print("   🔐 PIN: 0000")
    print("\n📱 DEMO USERS:")
    print("   👤 john_doe | PIN: 1234 | Account: 1234567890")
    print("   👤 jane_smith | PIN: 1234 | Account: 0987654321")
    print("   👤 mike_johnson | PIN: 1234 | Account: 1122334455")
    print("   👤 sarah_lee | PIN: 1234 | Account: 5566778899")
    print("   👤 james_bond | PIN: 1234 | Account: 6677889900")
    print("\n🔍 SHAP EXPLANATIONS:")
    print("   Every transaction includes SHAP values explaining why AI flagged it")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
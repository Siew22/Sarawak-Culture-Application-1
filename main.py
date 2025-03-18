import os
import time
import torch
import requests
import pandas as pd
import logging
import re
import traceback
import random
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from datasets import Dataset as HF_Dataset
from fastapi import FastAPI, HTTPException, Query, Depends
from typing import Optional
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pyodbc
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pathlib import Path
from urllib.parse import unquote
from fastapi.middleware.cors import CORSMiddleware

# ---------------------- 环境设置与日志 ----------------------
load_dotenv(dotenv_path=".env")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"CUDA available, using device: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available, falling back to CPU")

# ---------------------- Azure SQL 数据库配置 ----------------------
conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={os.getenv('DB_SERVER')};"
    f"DATABASE={os.getenv('DB_NAME')};"
    f"UID={os.getenv('DB_USER')};"
    f"PWD={os.getenv('DB_PASSWORD')}"
)

def get_db_connection():
    try:
        conn = pyodbc.connect(conn_str)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to Azure SQL database: {str(e)}")
        raise

# ---------------------- 数据模型 ----------------------
class ItineraryItem(BaseModel):
    user_id: int
    day: int
    time_slot: str
    category: str  # FOOD, ATTRACTION, EXPERIENCE
    name: str
    address: str

# ---------------------- JWT 认证配置 ----------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"user_id": user_id}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------------------- 数据定义 ----------------------
ATTRACTIONS = {
    "Kuching": [
        {"name": "猫博物馆", "address": "Jalan Tun Ahmad Zaidi Adruce, 93400 Kuching, Sarawak, 马来西亚"},
        {"name": "沙捞越文化村", "address": "Pantai Damai, 93752 Kuching, Sarawak, 马来西亚"},
        {"name": "古晋旧法院", "address": "Jalan Tun Abang Haji Openg, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "猫城广场", "address": "Jalan Main Bazaar, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "古晋滨水区", "address": "Kuching Waterfront, 93000 Kuching, Sarawak, 马来西亚"}
    ]
}
FOODS = {
    "Kuching": [
        {"name": "沙捞越叻沙", "address": "Jalan Padungan, 93100 Kuching, Sarawak, 马来西亚"},
        {"name": "马来西亚肉骨茶", "address": "Jalan Song, 93350 Kuching, Sarawak, 马来西亚"},
        {"name": "沙捞越层糕", "address": "Jalan India, 93100 Kuching, Sarawak, 马来西亚"},
        {"name": "三层肉饭", "address": "Main Bazaar, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "古早味面", "address": "Jalan Carpenter, 93000 Kuching, Sarawak, 马来西亚"}
    ]
}
EXPERIENCES = {
    "Kuching": [
        {"name": "拜访伊班族长屋", "address": "Batang Ai, Sarawak, 马来西亚"},
        {"name": "婆罗洲雨林徒步", "address": "Bako National Park, 93050 Kuching, Sarawak, 马来西亚"},
        {"name": "游览砂拉越河", "address": "Kuching Waterfront, 93000 Kuching, Sarawak, 马来西亚"},
        {"name": "探索风洞国家公园", "address": "Gunung Mulu National Park, Sarawak, 马来西亚"},
        {"name": "夜市探险", "address": "Jalan Satok, 93400 Kuching, Sarawak, 马来西亚"}
    ]
}

DEFAULT_SAMPLE_FILE_PATH = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\itinerary_samples.txt"

# ---------------------- BERT Preference Prediction Module ----------------------
bert_model_path = os.path.join("models", "bert_classifier")
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
if os.path.exists(bert_model_path):
    logger.info(f"Loading BERT model from: {bert_model_path}")
    model_bert = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=2).to(device)
else:
    logger.info("No trained BERT model found, loading pre-trained model for training...")
    model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
model_bert.eval()

def predict_preference(text: str) -> int:
    with torch.no_grad():
        inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        outputs = model_bert(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

def get_user_preferences(excel_input: str) -> Tuple[Dict, Dict]:
    try:
        logger.debug(f"Attempting to load preferences file: {excel_input}")
        if not os.path.exists(excel_input):
            raise FileNotFoundError(f"File not found: {excel_input}")

        df = pd.read_excel(excel_input, header=None, dtype=str)
        df = df.fillna("")
        logger.debug(f"Excel file content:\n{df.to_string()}")

        preferences_by_day = {}
        extra_preferences = {}
        current_day = None
        current_prefs = []
        in_extra_section = False

        for index, row in df.iterrows():
            if all(val == "" for val in row):
                continue
            if str(row[0]).strip().lower().startswith("day"):
                if current_day is not None and current_prefs:
                    preferences_by_day[current_day] = current_prefs
                day_value = str(row[0]).strip().lower().replace("day", "").strip()
                if day_value and day_value.isdigit():
                    current_day = int(day_value)
                else:
                    raise ValueError(f"Row {index}: Invalid 'Day' value, expected a number, got {day_value}")
                current_prefs = []
                in_extra_section = False
                logger.debug(f"Found Day: {current_day}")
                continue

            if str(row[0]).strip().lower() == "location" and str(row[1]).strip().lower() == "days":
                continue

            if str(row[1]).strip().lower().startswith("experiences"):
                if current_day is not None and current_prefs:
                    preferences_by_day[current_day] = current_prefs
                current_day = None
                current_prefs = []
                in_extra_section = True
                logger.debug("Entering extra preferences section")
                continue

            if row[1] and row[2]:
                category = str(row[1]).strip().lower().replace("s", "")
                try:
                    rate = float(row[2])
                    if rate < 0:
                        rate = 0
                        logger.warning(f"Row {index}: negative rate found, set to 0")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Row {index}: 'rate' must be numeric, got {row[2]}")
                logger.debug(f"Parsing category: {category}, rate: {rate}")
                if in_extra_section:
                    extra_preferences[category] = rate
                elif current_day is not None:
                    current_prefs.append((category, rate))

        if current_day is not None and current_prefs:
            preferences_by_day[current_day] = current_prefs

        logger.debug(f"Extracted preferences by day: {preferences_by_day}")
        logger.debug(f"Extracted extra preferences: {extra_preferences}")
        if not preferences_by_day and not extra_preferences:
            raise ValueError("No valid preferences extracted from the file.")
        logger.info(f"Successfully extracted preferences: {preferences_by_day}, extra: {extra_preferences}")
        return preferences_by_day, extra_preferences

    except Exception as e:
        logger.error(f"Error in get_user_preferences: {str(e)}\n{traceback.format_exc()}")
        raise

def train_bert(excel_path: str, epochs: int = 15, batch_size: int = 4):
    try:
        preferences_by_day, extra_preferences = get_user_preferences(excel_path)
        texts = []
        labels = []
        threshold = 2.5

        for day, prefs in preferences_by_day.items():
            for category, rate in prefs:
                texts.append(f"{category} (Day {day})")
                labels.append(1 if rate > threshold else 0)

        for category, rate in extra_preferences.items():
            texts.append(f"{category} (Extra Preference)")
            labels.append(1 if rate > threshold else 0)

        logger.debug(f"Number of preference data: {len(texts)}")
        logger.debug(f"Texts: {texts}")
        logger.debug(f"Labels: {labels}")
        if len(texts) == 0:
            raise ValueError("No preference data available for training.")
        effective_batch_size = min(batch_size, len(texts))

        class BertDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=256):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)
                input_ids = encoding["input_ids"].squeeze()
                attention_mask = encoding["attention_mask"].squeeze()
                label = torch.tensor(self.labels[idx], dtype=torch.long)
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}

        dataset = BertDataset(texts, labels, tokenizer_bert)
        if len(dataset) < effective_batch_size:
            effective_batch_size = 1
            logger.warning(f"Data size {len(dataset)} is less than batch_size {batch_size}, setting batch_size to 1")

        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        optimizer = AdamW(model_bert.parameters(), lr=2e-5)
        model_bert.train().to(device)

        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            for i, batch in enumerate(dataloader):
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model_bert(**batch)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_bert.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    logger.error(f"Runtime error during training (epoch {epoch+1}, batch {i}): {str(e)}\n{traceback.format_exc()}")
                    torch.cuda.empty_cache()
                    continue
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        os.makedirs(bert_model_path, exist_ok=True)
        model_bert.save_pretrained(bert_model_path)
        tokenizer_bert.save_pretrained(bert_model_path)
        logger.info(f"BERT model saved to {bert_model_path}")

    except Exception as e:
        logger.error(f"Error in train_bert: {str(e)}\n{traceback.format_exc()}")
        raise

# ---------------------- GPT-2 Itinerary Generation Module ----------------------
BASE_DIR = Path(__file__).resolve().parent
custom_gpt2_model_path = BASE_DIR / "models" / "gpt2_finetuned"
#custom_gpt2_model_path = r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\models\gpt2_finetuned"
try:
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(custom_gpt2_model_path)
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token
except Exception as e:
    logger.error(f"Failed to load tokenizer from {custom_gpt2_model_path}: {str(e)}")
    raise

try:
    if os.path.exists(custom_gpt2_model_path):
        logger.info(f"Loading fine-tuned GPT-2 model from: {custom_gpt2_model_path}")
        model_gpt2 = GPT2LMHeadModel.from_pretrained(custom_gpt2_model_path).to(device)
    else:
        raise FileNotFoundError(f"Fine-tuned model not found at {custom_gpt2_model_path}")
except Exception as e:
    logger.error(f"Failed to load fine-tuned GPT-2 model: {str(e)}")
    raise

class GPT2Dataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        input_ids = encoding["input_ids"].squeeze()
        return input_ids

def fine_tune_gpt2(data_file: str, epochs: int = 5, batch_size: int = 2):
    try:
        logger.debug(f"Attempting to fine-tune GPT-2 with data file: {data_file}")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Sample file not found: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            text = f.read()
        examples = text.split("Generate a 5-day travel itinerary:")
        examples = [("Generate a 5-day travel itinerary:" + ex).strip() for ex in examples if ex.strip()]
        if not examples:
            raise ValueError("No training examples found in the data file.")

        data_dict = {"text": examples}
        hf_dataset = HF_Dataset.from_dict(data_dict)
        hf_dataset = hf_dataset.shuffle(seed=42)
        split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        def tokenize_function(examples):
            return tokenizer_gpt2(examples["text"], truncation=True, max_length=512)

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_gpt2, mlm=False)

        training_args = TrainingArguments(
            output_dir="models/gpt2_finetuned",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            logging_steps=50,
            learning_rate=5e-5,
            warmup_steps=100,
            fp16=True if torch.cuda.is_available() else False,
        )

        trainer = Trainer(
            model=model_gpt2,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model("models/gpt2_finetuned")
        tokenizer_gpt2.save_pretrained("models/gpt2_finetuned")
        logger.info("GPT-2 fine-tuning completed and saved to models/gpt2_finetuned")

    except Exception as e:
        logger.error(f"Error in fine_tune_gpt2: {str(e)}\n{traceback.format_exc()}")
        raise

# ---------------------- 辅助函数 ----------------------
def normalize_text(text: str) -> str:
    text = text.replace("Address(s):", "address:")
    text = text.replace("Address:", "address:")
    return text.lower()

def extract_locations(itinerary: str) -> Dict[str, str]:
    logger.debug(f"Extracting locations, input itinerary: {itinerary}")
    locations = {}
    normalized_itinerary = normalize_text(itinerary)
    pattern = re.compile(r"-\s*(food|experience|attraction):\s*([^,]+),\s*address:\s*(.+)", re.IGNORECASE)
    for line in normalized_itinerary.split("\n"):
        match = pattern.search(line.strip())
        if match:
            category, name, address = match.groups()
            locations[name.strip()] = address.strip()
    logger.debug(f"Extracted locations: {locations}")
    return locations

def geocode_location(location: str) -> Tuple[float, float]:
    cache_file = "location_cache.txt"
    cache = {}
    manual_coords = {
        "7 jalan legoland, 79100 nusajaya, johor, malaysia": (1.5028, 103.6314),
        "jalan balik pulau, 11500 air itam, penang, malaysia": (5.4044, 100.2762),
        "pantai damai santubong, 93050 kuching, sarawak, malaysia": (1.7167, 110.3167),
        "siam road char koay teow, 82 jalan siam, 10400 george town, penang, malaysia": (5.4226, 100.3251),
        "gua tempurung, 31600 gopeng, perak, malaysia": (4.4149, 101.1879),
        "top spot food court, jalan padungan, 93100 kuching, sarawak, malaysia": (1.5593, 110.3442),
        "kuching waterfront, 93000 kuching, sarawak, malaysia": (1.5595, 110.3467),
        "pantai damai santubong, 93050 kuching, sarawak, malaysia": (1.7167, 110.3167),
        "jalan puncak borneo, 93250 kuching, sarawak, malaysia": (1.4131, 110.2847),
        "kuala lumpur city centre, 50088 kuala lumpur, malaysia": (3.1579, 101.7123),
        "5, jalan ss 21/37, damansara utama, 47400 petaling jaya, malaysia": (3.1353, 101.6235),
        "gombak, 68100 batu caves, selangor, malaysia": (3.2379, 101.6811),
        "jalan puncak, 50250 kuala lumpur, malaysia": (3.1488, 101.7051),
        "gunung gading national park": (1.69, 109.85),
        "semenggoh wildlife centre": (1.39, 110.31),
        "sin lian shin, jalan sekama, 93300 kuching, sarawak, 马来西亚": (1.5532, 110.3645),
        "sin lian shin": (1.5532, 110.3645),
    }
    norm_location = location.strip().lower()
    if norm_location == "to be added":
        return None
    if norm_location in manual_coords:
        logger.debug(f"Using manual coordinates for {norm_location}: {manual_coords[norm_location]}")
        return manual_coords[norm_location]
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            for line in f:
                key, lat, lon = line.strip().split("|")
                cache[key] = [float(lat), float(lon)]
    if norm_location in cache:
        logger.debug(f"Using cached coordinates for {norm_location}: {cache[norm_location]}")
        return tuple(cache[norm_location])
    
    query_location = norm_location
    if "malaysia" not in norm_location:
        query_location = f"{norm_location}, malaysia"
    full_address = None
    for category_dict in [FOODS, ATTRACTIONS, EXPERIENCES]:
        if location in category_dict:
            for item in category_dict[location]:
                if item["name"].lower() == norm_location:
                    full_address = item["address"].lower()
                    break
    if full_address:
        query_location = full_address
        logger.debug(f"Found full address for {norm_location}: {query_location}")

    params = {"q": query_location, "format": "json", "limit": 1}
    try:
        response = requests.get("https://nominatim.openstreetmap.org/search", params=params,
                               headers={'User-Agent': 'TravelAssistant/2.0'}, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            cache[norm_location] = [lat, lon]
            with open(cache_file, "a") as f:
                f.write(f"{norm_location}|{lat}|{lon}\n")
            logger.debug(f"Geocoded {norm_location} to ({lat}, {lon})")
            return (lat, lon)
        else:
            logger.warning(f"Geocoding failed for: {query_location}")
            return None
    except Exception as e:
        logger.error(f"Geocoding error for {query_location}: {str(e)}")
        return None

def get_navigation(start: str, end: str) -> Dict:
    api_key = os.getenv("ORS_API_KEY")
    if not api_key:
        logger.warning("ORS_API_KEY not set in .env file")
        return {"error": "Please set ORS_API_KEY in .env file"}

    start_coord = geocode_location(start)
    end_coord = geocode_location(end)
    if not start_coord or not end_coord:
        error_detail = "Geocoding failed for"
        if not start_coord:
            error_detail += f" start: {start}"
        if not end_coord:
            error_detail += f" end: {end}" if not start_coord else f", end: {end}"
        logger.error(error_detail)
        return {"error": error_detail}

    start_lon, start_lat = start_coord[1], start_coord[0]
    end_lon, end_lat = end_coord[1], end_coord[0]

    is_borneo = 109 <= start_lon <= 115 and 0.5 <= start_lat <= 5
    is_peninsula = 100 <= end_lon <= 103 and 1 <= end_lat <= 7
    is_cross_island = (is_borneo and is_peninsula) or (is_peninsula and is_borneo)

    profiles = ["foot-walking", "driving-car"]
    recommendations = {}

    for profile in profiles:
        url = (f"https://api.openrouteservice.org/v2/directions/{profile}?"
               f"api_key={api_key}&start={start_lon},{start_lat}&end={end_lon},{end_lat}&format=json")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Navigation response for {profile}: {data}")

            if "features" in data and data["features"]:
                route = data["features"][0]["properties"]["segments"][0]
                duration_minutes = route["duration"] / 60
                distance_km = route["distance"] / 1000

                if profile == "foot-walking" and (duration_minutes > 1000 or distance_km > 500):
                    recommendations[profile] = {"error": "Walking not feasible for this distance"}
                else:
                    recommendations[profile] = {
                        "duration_minutes": round(duration_minutes, 2),
                        "distance_km": round(distance_km, 2)
                    }
            else:
                recommendations[profile] = {"error": "No route found for this mode"}
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                recommendations[profile] = {"error": f"Mode not supported in this region: {str(e)}"}
            else:
                logger.error(f"Navigation request failed for {profile}: {str(e)}")
                recommendations[profile] = {"error": f"Request failed: {str(e)}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Navigation request failed for {profile}: {str(e)}")
            recommendations[profile] = {"error": f"Request failed: {str(e)}"}

    if is_cross_island or (start.lower() == "kuching" and end.lower() == "kuala lumpur"):
        recommendations = {
            "message": "Direct travel between the locations is not possible by walking, driving, or public transport due to geographic separation. Consider taking a flight or other transport."
        }

    return {"recommendations": recommendations}

def format_navigation(nav_data: Dict) -> str:
    if "error" in nav_data:
        return f"Error: {nav_data['error']}"
    
    recommendations = nav_data.get("recommendations", {})
    if "message" in recommendations:
        return recommendations["message"]

    formatted_output = "Navigation Recommendations:\n"
    for profile, details in recommendations.items():
        if "error" in details:
            formatted_output += f"- {profile.replace('-', ' ').title()}: {details['error']}\n"
        else:
            mode = profile.replace('-', ' ').title()
            duration = details["duration_minutes"]
            distance = details["distance_km"]
            formatted_output += f"- {mode}: {duration:.1f} mins, {distance:.2f} km\n"
    
    return formatted_output.strip()

def build_prompt(location: str, day: int, food_count: int, attraction_count: int, experience_count: int, sample_file: str) -> str:
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            samples = [s.strip() for s in content.split("20-DAY TRAVEL ITINERARY:") if s.strip()]
            logger.debug(f"Total samples extracted: {len(samples)}")

            if not samples:
                logger.warning(f"No valid samples found in {sample_file}, using default prompt")
                sample_context = (
                    "DAY 1\n"
                    "MORNING\n"
                    "ATTRACTION: SEMENGGOH WILDLIFE CENTRE, observe orangutans, Address: Semenggoh, 93250 Kuching, Sarawak, Malaysia\n"
                    "FOOD: SARAWAK LAKSA, Address: Choon Hui Cafe, Jalan Ban Hock, 93100 Kuching, Sarawak, Malaysia\n"
                    "NOON\n"
                    "EXPERIENCE: Orangutan feeding session, Address: Semenggoh Wildlife Centre, 93250 Kuching, Sarawak, Malaysia\n"
                    "ATTRACTION: KUCHING WATERFRONT, explore the riverside, Address: 93000 Kuching, Sarawak, Malaysia\n"
                    "AFTERNOON\n"
                    "FOOD: KOLO MEE, Address: Sin Lian Shin, Jalan Sekama, 93300 Kuching, Sarawak, Malaysia\n"
                    "ATTRACTION: SARAWAK STATE MUSEUM, delve into local history, Address: Jalan Tun Abang Haji Openg, 93000 Kuching, Sarawak, Malaysia\n"
                    "EVENING\n"
                    "EXPERIENCE: Sunset cruise on Sarawak River, Address: Kuching Waterfront, 93000 Kuching, Sarawak, Malaysia\n"
                    "FOOD: GRILLED FISH, Address: Top Spot Food Court, Jalan Padungan, 93100 Kuching, Sarawak, Malaysia"
                )
            else:
                relevant_samples = []
                normalized_location = location.split(",")[0].strip().upper()
                for sample in samples:
                    if normalized_location in sample:
                        relevant_samples.append(sample)
                        if len(relevant_samples) >= 2:
                            break
                
                if not relevant_samples:
                    relevant_samples = random.sample(samples, min(2, len(samples)))
                
                sample_context = "\n\n".join(relevant_samples[:2])[:1000]  # 限制前 1000 字符
                logger.debug(f"Selected sample context for {location}:\n{sample_context}")
    except Exception as e:
        logger.error(f"Error reading sample file {sample_file}: {str(e)}")
        sample_context = (
            "DAY 1\n"
            "MORNING\n"
            "ATTRACTION: SEMENGGOH WILDLIFE CENTRE, observe orangutans, Address: Semenggoh, 93250 Kuching, Sarawak, Malaysia\n"
            "FOOD: SARAWAK LAKSA, Address: Choon Hui Cafe, Jalan Ban Hock, 93100 Kuching, Sarawak, Malaysia\n"
            "NOON\n"
            "EXPERIENCE: Orangutan feeding session, Address: Semenggoh Wildlife Centre, 93250 Kuching, Sarawak, Malaysia\n"
            "ATTRACTION: KUCHING WATERFRONT, explore the riverside, Address: 93000 Kuching, Sarawak, Malaysia\n"
            "AFTERNOON\n"
            "FOOD: KOLO MEE, Address: Sin Lian Shin, Jalan Sekama, 93300 Kuching, Sarawak, Malaysia\n"
            "ATTRACTION: SARAWAK STATE MUSEUM, delve into local history, Address: Jalan Tun Abang Haji Openg, 93000 Kuching, Sarawak, Malaysia\n"
            "EVENING\n"
            "EXPERIENCE: Sunset cruise on Sarawak River, Address: Kuching Waterfront, 93000 Kuching, Sarawak, Malaysia\n"
            "FOOD: GRILLED FISH, Address: Top Spot Food Court, Jalan Padungan, 93100 Kuching, Sarawak, Malaysia"
        )

    prompt = (
        f"Generate a 1-day travel itinerary for {location.upper()} with the following requirements:\n"
        f"- FOOD: {food_count} recommendations\n"
        f"- ATTRACTION: {attraction_count} recommendations\n"
        f"- EXPERIENCE: {experience_count} recommendations\n"
        "Format:\n"
        "DAY X\n"
        "MORNING\n"
        "ATTRACTION: [Name], [description], Address: [Full Address in Malaysia]\n"
        "FOOD: [Name], Address: [Full Address in Malaysia]\n"
        "NOON\n"
        "EXPERIENCE: [Name], Address: [Full Address in Malaysia]\n"
        "ATTRACTION: [Name], [description], Address: [Full Address in Malaysia]\n"
        "AFTERNOON\n"
        "FOOD: [Name], Address: [Full Address in Malaysia]\n"
        "ATTRACTION: [Name], [description], Address: [Full Address in Malaysia]\n"
        "EVENING\n"
        "EXPERIENCE: [Name], Address: [Full Address in Malaysia]\n"
        "FOOD: [Name], Address: [Full Address in Malaysia]\n"
        "Rules:\n"
        "1. Distribute activities evenly across MORNING, NOON, AFTERNOON, and EVENING.\n"
        "2. Use complete and realistic addresses (e.g., 'Jalan Sultan, 50000 Kuala Lumpur, Malaysia').\n"
        "3. Do NOT repeat locations or use placeholders.\n"
        "4. Use English for all text, and capitalize section names (e.g., MORNING, FOOD).\n\n"
        "Examples:\n"
        f"{sample_context}\n\n"
        f"Now generate a 1-day itinerary for DAY {day} in {location.upper()}:\n"
    )
    logger.debug(f"Prompt for Day {day} (first 2000 chars):\n{prompt[:2000]}...")
    return prompt

def distribute_recommendations(available_foods, available_attractions, available_experiences, days, food_value, attraction_value, experience_value):
    daily_foods = [food_value] * days
    daily_attractions = [attraction_value] * days
    daily_experiences = [experience_value] * days
    return daily_foods, daily_attractions, daily_experiences

def enforce_exact_recommendations(
    day_text: str,
    food_count: int,
    attraction_count: int,
    experience_count: int,
    day: int,
    location: str,
    all_days_counts: Tuple[List, List, List],
    day_index: int,
    available_attractions: List[Dict] = None,
    available_foods: List[Dict] = None,
    available_experiences: List[Dict] = None
) -> str:
    available_attractions = available_attractions or []
    available_foods = available_foods or []
    available_experiences = available_experiences or []
    
    if len(available_attractions) < attraction_count:
        logger.warning(f"Insufficient attractions for DAY {day}: {len(available_attractions)} available, {attraction_count} needed.")
        available_attractions.extend(random.sample(available_attractions, attraction_count - len(available_attractions)) if available_attractions else [{"name": f"Placeholder Attraction {i}", "address": f"Unknown Address, {location}"} for i in range(attraction_count - len(available_attractions))])
    if len(available_foods) < food_count:
        logger.warning(f"Insufficient foods for DAY {day}: {len(available_foods)} available, {food_count} needed.")
        available_foods.extend(random.sample(available_foods, food_count - len(available_foods)) if available_foods else [{"name": f"Placeholder Food {i}", "address": f"Unknown Address, {location}"} for i in range(food_count - len(available_foods))])
    if len(available_experiences) < experience_count:
        logger.warning(f"Insufficient experiences for DAY {day}: {len(available_experiences)} available, {experience_count} needed.")
        available_experiences.extend(random.sample(available_experiences, experience_count - len(available_experiences)) if available_experiences else [{"name": f"Placeholder Experience {i}", "address": f"Unknown Address, {location}"} for i in range(experience_count - len(available_experiences))])
    
    daily_foods, daily_attractions, daily_experiences = all_days_counts
    this_day_food = daily_foods[day_index]
    this_day_attraction = daily_attractions[day_index]
    this_day_experience = daily_experiences[day_index]
    
    selected_attractions = random.sample(available_attractions, this_day_attraction)
    selected_foods = random.sample(available_foods, this_day_food)
    selected_experiences = random.sample(available_experiences, this_day_experience)
    
    attraction_activities = [f"ATTRACTION: {item['name']}, Address: {item['address']}" for item in selected_attractions]
    food_activities = [f"FOOD: {item['name']}, Address: {item['address']}" for item in selected_foods]
    experience_activities = [f"EXPERIENCE: {item['name']}, Address: {item['address']}" for item in selected_experiences]
    
    time_slots = ["MORNING", "NOON", "AFTERNOON", "EVENING"]
    slot_allocations = {slot: [] for slot in time_slots}
    total_activities = this_day_food + this_day_attraction + this_day_experience
    activities_per_slot = max(3, total_activities // len(time_slots))
    
    remaining_foods = food_activities.copy()
    remaining_attractions = attraction_activities.copy()
    remaining_experiences = experience_activities.copy()
    
    for slot in time_slots:
        for _ in range(min(2, this_day_food // len(time_slots) + (1 if len(remaining_foods) > 0 else 0))):
            if remaining_foods:
                slot_allocations[slot].append(remaining_foods.pop(0))
        for _ in range(min(2, this_day_attraction // len(time_slots) + (1 if len(remaining_attractions) > 0 else 0))):
            if remaining_attractions:
                slot_allocations[slot].append(remaining_attractions.pop(0))
        for _ in range(min(2, this_day_experience // len(time_slots) + (1 if len(remaining_experiences) > 0 else 0))):
            if remaining_experiences:
                slot_allocations[slot].append(remaining_experiences.pop(0))
    
    remaining_activities = remaining_foods + remaining_attractions + remaining_experiences
    random.shuffle(remaining_activities)
    for slot in time_slots:
        while remaining_activities and len(slot_allocations[slot]) < 5:
            slot_allocations[slot].append(remaining_activities.pop(0))
    
    while remaining_activities:
        for slot in time_slots:
            if remaining_activities:
                slot_allocations[slot].append(remaining_activities.pop(0))
    
    total_allocated = sum(len(activities) for activities in slot_allocations.values())
    if total_allocated < total_activities:
        logger.warning(f"Insufficient activities allocated for DAY {day}: {total_allocated} allocated, {total_activities} needed.")
        while total_allocated < total_activities:
            for slot in time_slots:
                if len(slot_allocations[slot]) < 5:
                    slot_allocations[slot].append(f"FOOD: Placeholder, Address: {location}")
                    total_allocated += 1
                    if total_allocated >= total_activities:
                        break
    
    new_day = f"DAY {day}\n"
    for slot, slot_activities in slot_allocations.items():
        new_day += f"{slot}\n"
        if slot_activities:
            for activity in slot_activities:
                new_day += f"{activity}\n"
        else:
            new_day += "to be added\n"
    
    logger.debug(f"Generated DAY {day} itinerary:\n{new_day}")
    return new_day

def correct_itinerary_with_bert(day_text: str) -> str:
    logger.debug("Starting BERT correction...")
    try:
        return day_text  # 简化处理，直接返回
    except Exception as e:
        logger.error(f"Error in BERT correction: {str(e)}")
        return day_text

def validate_itinerary(itinerary: str, days: int = 1) -> bool:
    required_days = [f"DAY {i}" for i in range(1, days + 1)]
    missing_days = [day for day in required_days if day not in itinerary]
    if missing_days:
        logger.warning(f"Missing days: {missing_days}")
        return False
    logger.debug(f"Itinerary validation passed for {days} days")
    return True

def format_itinerary(raw_text: str, days: int = 5) -> str:
    logger.debug(f"Raw itinerary text: {raw_text}")
    try:
        content = raw_text.strip()
        day_sections = re.split(r"##\s*Day\s*\d+", content)[1:]
        if len(day_sections) < days:
            logger.warning(f"Detected {len(day_sections)} days, less than {days}, forcing supplement")
            while len(day_sections) < days:
                day_sections.append(
                    "\n### Morning\n- to be added\n"
                    "### Noon\n- to be added\n"
                    "### Afternoon\n- to be added\n"
                    "### Evening\n- to be added"
                )
        md_output = "## Personalized Travel Itinerary\n\n"
        for i, day_text in enumerate(day_sections[:days], start=1):
            day_text = day_text.strip()
            md_output += f"## Day {i}\n{day_text}\n\n"
        logger.debug(f"Formatted itinerary:\n{md_output}")
        return md_output
    except Exception as e:
        logger.error(f"Itinerary formatting failed: {str(e)}\n{traceback.format_exc()}")
        return raw_text

async def save_itinerary_to_db(itinerary: Dict, user_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for day_data in itinerary["itinerary"]:
            day = day_data["day"]
            for time_slot, activities in day_data["schedule"].items():
                for category in ["food", "attraction", "experience"]:
                    if category in activities:
                        for item in activities[category]:
                            name_address = item.split(", Address: ")
                            name = name_address[0].split(": ")[1]
                            address = name_address[1] if len(name_address) > 1 else "Unknown"
                            query = """
                                INSERT INTO Itinerary (user_id, day, time_slot, category, name, address)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """
                            cursor.execute(query, (user_id, day, time_slot, category.upper(), name, address))
        conn.commit()
        logger.info(f"Itinerary saved to database for user_id: {user_id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error saving itinerary to DB: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save itinerary")
    finally:
        cursor.close()
        conn.close()

async def generate_itinerary(location: str, days: int, food_value: int, attraction_value: int, experience_value: int, sample_file: str, use_gpt2: bool = False, user_id: int = None) -> Dict:
    try:
        logger.info(f"Generating itinerary for location: {location}, days: {days}, food_value: {food_value}, attraction_value: {attraction_value}, experience_value: {experience_value}")
        
        available_attractions = []
        available_foods = []
        available_experiences = []
        seen_foods = set()
        seen_attractions = set()
        seen_experiences = set()
        
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            samples = [s.strip() for s in re.split(r"20-DAY TRAVEL ITINERARY:", content, flags=re.IGNORECASE) if s.strip()]
            logger.debug(f"Extracted samples: {len(samples)}")
            
            if not samples:
                logger.warning("No samples found in file. Using default activities.")
            else:
                normalized_location = location.split(",")[0].strip().upper()
                matched_samples = []
                for sample in samples:
                    if normalized_location in sample.upper():
                        matched_samples.append(sample)
                
                if not matched_samples:
                    logger.warning(f"No samples matched for location {normalized_location}. Using all available samples.")
                    matched_samples = samples
                
                for sample in matched_samples:
                    for line in sample.split("\n"):
                        line = line.strip()
                        if re.match(r"ATTRACTION:", line, re.IGNORECASE):
                            parts = re.split(r", Address:", line, maxsplit=1)
                            if len(parts) >= 2 and parts[1].strip():
                                name_desc = parts[0].replace("ATTRACTION:", "").strip()
                                address = parts[1].strip()
                                if normalized_location not in address.upper():
                                    continue
                                if name_desc not in seen_attractions:
                                    seen_attractions.add(name_desc)
                                    available_attractions.append({"name": name_desc, "address": address})
                        elif re.match(r"FOOD:", line, re.IGNORECASE):
                            parts = re.split(r", Address:", line, maxsplit=1)
                            if len(parts) >= 2 and parts[1].strip():
                                name = parts[0].replace("FOOD:", "").strip()
                                address = parts[1].strip()
                                if normalized_location not in address.upper():
                                    continue
                                if name not in seen_foods:
                                    seen_foods.add(name)
                                    available_foods.append({"name": name, "address": address})
                        elif re.match(r"EXPERIENCE:", line, re.IGNORECASE):
                            parts = re.split(r", Address:", line, maxsplit=1)
                            if len(parts) >= 2 and parts[1].strip():
                                name = parts[0].replace("EXPERIENCE:", "").strip()
                                address = parts[1].strip()
                                if normalized_location not in address.upper():
                                    continue
                                if name not in seen_experiences:
                                    seen_experiences.add(name)
                                    available_experiences.append({"name": name, "address": address})
        
        total_food_needed = food_value * days
        total_attractions_needed = attraction_value * days
        total_experiences_needed = experience_value * days
        
        if len(available_foods) < total_food_needed:
            available_foods.extend(random.sample(available_foods, total_food_needed - len(available_foods)) if available_foods else [{"name": f"Placeholder Food {i}", "address": f"Unknown Address, {location}"} for i in range(total_food_needed - len(available_foods))])
        if len(available_attractions) < total_attractions_needed:
            available_attractions.extend(random.sample(available_attractions, total_attractions_needed - len(available_attractions)) if available_attractions else [{"name": f"Placeholder Attraction {i}", "address": f"Unknown Address, {location}"} for i in range(total_attractions_needed - len(available_attractions))])
        if len(available_experiences) < total_experiences_needed:
            available_experiences.extend(random.sample(available_experiences, total_experiences_needed - len(available_experiences)) if available_experiences else [{"name": f"Placeholder Experience {i}", "address": f"Unknown Address, {location}"} for i in range(total_experiences_needed - len(available_experiences))])
        
        daily_foods, daily_attractions, daily_experiences = distribute_recommendations(
            available_foods, available_attractions, available_experiences, days, food_value, attraction_value, experience_value
        )
        
        itinerary_list = []
        for day in range(1, days + 1):
            day_index = day - 1
            day_text = ""
            
            all_days_counts = (daily_foods, daily_attractions, daily_experiences)
            day_text = enforce_exact_recommendations(
                day_text=day_text,
                food_count=daily_foods[day_index],
                attraction_count=daily_attractions[day_index],
                experience_count=daily_experiences[day_index],
                day=day,
                location=location,
                all_days_counts=all_days_counts,
                day_index=day_index,
                available_attractions=available_attractions,
                available_foods=available_foods,
                available_experiences=available_experiences
            )
            
            day_schedule = {"MORNING": {}, "NOON": {}, "AFTERNOON": {}, "EVENING": {}}
            current_slot = None
            for line in day_text.split("\n"):
                line = line.strip()
                if line in ["MORNING", "NOON", "AFTERNOON", "EVENING"]:
                    current_slot = line
                elif line.startswith("FOOD:") and current_slot:
                    if "food" not in day_schedule[current_slot]:
                        day_schedule[current_slot]["food"] = []
                    day_schedule[current_slot]["food"].append(line)
                elif line.startswith("ATTRACTION:") and current_slot:
                    if "attraction" not in day_schedule[current_slot]:
                        day_schedule[current_slot]["attraction"] = []
                    day_schedule[current_slot]["attraction"].append(line)
                elif line.startswith("EXPERIENCE:") and current_slot:
                    if "experience" not in day_schedule[current_slot]:
                        day_schedule[current_slot]["experience"] = []
                    day_schedule[current_slot]["experience"].append(line)
            
            itinerary_list.append({"day": day, "schedule": day_schedule})
        
        result = {
            "itinerary": itinerary_list,
            "location": location.upper(),
            "generated_at": datetime.now().isoformat()
        }
        
        if user_id:
            await save_itinerary_to_db(result, user_id)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in generate_itinerary: {str(e)}\n{traceback.format_exc()}")
        raise

# ---------------------- FastAPI Application ----------------------
app = FastAPI()

# 添加 CORS 中间件，允许 Flutter 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发时允许所有来源，生产环境限制特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        logger.info("Successfully connected to Azure SQL database")
        cursor.close()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to connect to Azure SQL database: {str(e)}. Proceeding without DB connection.")
        logger.info(f"Registered routes: {[route.path for route in app.routes]}")
        # 不抛出异常，让服务器继续启动

@app.on_event("shutdown")
def shutdown():
    pass  # pyodbc 不需要显式关闭全局连接，因为我们每次都新建连接

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Travel Assistant API"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # 这里假设有一个简单的用户验证逻辑，实际应从数据库验证
    user_id = 1  # 模拟用户ID，应从数据库获取
    token_data = {"sub": str(user_id), "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}
    access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/generate_itinerary_json")
async def generate_itinerary_json(
    location: str = Query(..., description="Travel destination"),
    days: int = Query(1, description="Number of days", ge=1),
    food_value: int = Query(2, description="Number of food recommendations (max 5)", ge=0, le=5),
    attraction_value: int = Query(2, description="Number of attraction recommendations (max 5)", ge=0, le=5),
    experience_value: int = Query(1, description="Number of experience recommendations (max 5)", ge=0, le=5),
    sample_file: str = Query("", description="Path to sample itinerary file (optional, uses default if not provided)"),  # 改为可选
    use_gpt2: bool = Query(False, description="Use GPT-2 for generation")
):
    logger.debug(f"Received request: location={location}, days={days}, food_value={food_value}, attraction_value={attraction_value}, experience_value={experience_value}, sample_file={sample_file}, use_gpt2={use_gpt2}")
    try:
        # 如果未提供 sample_file，使用默认路径
        sample_file = unquote(sample_file).strip('"') or DEFAULT_SAMPLE_FILE_PATH
        sample_file_path = Path(sample_file)
        logger.debug(f"Resolved file path: {sample_file_path}, exists: {sample_file_path.exists()}")

        if not sample_file_path.exists():
            logger.error(f"Sample file not found: {sample_file_path}")
            raise HTTPException(status_code=400, detail=f"Sample file not found at {sample_file_path}")

        # 调用 generate_itinerary
        itinerary = await generate_itinerary(location, days, food_value, attraction_value, experience_value, str(sample_file_path), use_gpt2=use_gpt2)
        logger.debug(f"Generated itinerary: {itinerary}")
        return itinerary
    except HTTPException as e:
        logger.error(f"HTTP Exception: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.get("/itineraries/{user_id}")
async def get_itineraries(user_id: int, current_user: dict = Depends(get_current_user)):
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this user's itineraries")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM Itinerary WHERE user_id = ? ORDER BY day, time_slot"
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        itineraries = {}
        for row in rows:
            day = row.day
            if day not in itineraries:
                itineraries[day] = {"day": day, "schedule": {"MORNING": {}, "NOON": {}, "AFTERNOON": {}, "EVENING": {}}}
            time_slot = row.time_slot
            category = row.category.lower()
            if category not in itineraries[day]["schedule"][time_slot]:
                itineraries[day]["schedule"][time_slot][category] = []
            itineraries[day]["schedule"][time_slot][category].append(f"{row.category}: {row.name}, Address: {row.address}")
        conn.close()
        return {"itineraries": list(itineraries.values())}
    except Exception as e:
        logger.error(f"Error fetching itineraries: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch itineraries")

@app.get("/navigate")
async def navigate_endpoint(
    start: str = Query(..., description="Starting location"),
    end: str = Query(..., description="Ending location"),
):
    try:
        logger.debug(f"Navigating from {start} to {end}")
        nav_data = get_navigation(start, end)
        return nav_data
    except Exception as e:
        logger.error(f"Error in navigate_endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating navigation: {str(e)}")

# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    sample_file = os.getenv("ITINERARY_SAMPLES_FILE", r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\sample_itineraries.txt")
    preference_file = os.getenv("USER_PREFERENCES_FILE", r"C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\data\user_preferences.xlsx")

    try:
        if os.path.exists(preference_file):
            train_bert(preference_file, epochs=15, batch_size=4)
        else:
            logger.warning(f"Preference file not found, skipping BERT training: {preference_file}")
    except Exception as e:
        logger.error(f"Failed to train BERT: {str(e)}\n{traceback.format_exc()}")

    try:
        if os.path.exists(sample_file):
            fine_tune_gpt2(sample_file, epochs=5, batch_size=2)
        else:
            logger.warning(f"Sample file not found, skipping GPT-2 fine-tuning: {sample_file}")
    except Exception as e:
        logger.error(f"Failed to fine-tune GPT-2: {str(e)}\n{traceback.format_exc()}")

    import uvicorn
    logger.info("Starting FastAPI server on http:// 0.0.0.0:8801")
    uvicorn.run(app, host="0.0.0.0", port=8801, log_level="debug")
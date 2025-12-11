import os
import gc
import csv
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import ClapModel, ClapProcessor
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPModel, CLIPProcessor as HFCLIPProcessor
import serial
import time

# ==========================================
# [설정] 환경 변수 및 경로
#  - 사용자의 환경에 맞게 수정해주세요.
# ==========================================
DATA_ROOT = "/mnt/data"
TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_image.jpg")

# 음악 파일, 경로 및 임베딩 파일이 필요합니다.
MUSIC_EMB_PATH = os.path.join(DATA_ROOT, "jamendo_clap_filtered2.npy")
META_PATH = os.path.join(DATA_ROOT, "jamendo_clap_filtered2.csv")

# 사전 학습된 CLIP 감정 분류기 모델이 필요합니다.
CLIP_EMOTION_CKPT = os.path.join(DATA_ROOT, "clip_emotion_classifier.pt")

CLAP_CKPT = "laion/clap-htsat-unfused"
BLIP_CKPT = "Salesforce/blip-image-captioning-base"
CLIP_CKPT = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMO_LABELS = ["amusement","anger","awe","contentment","disgust","excitement","fear","sadness"]

print(f"System Info: Running on {DEVICE}")

# ==========================================
# [아두이노 연결 설정]
# ==========================================
ser = None
try:
    ARDUINO_PORT = '/dev/ttyACM0'
    print(f"Arduino 연결 시도 중... (포트: {ARDUINO_PORT})")
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    time.sleep(2)
    if ser.is_open:
        print(f"✅ Arduino 연결 성공! 포트가 열렸습니다.")
    else:
        ser = None
        print(f"⚠️ Arduino 포트를 열 수 없습니다. 시뮬레이션 모드로 실행됩니다.")
except serial.SerialException as e:
    ser = None
    print(f"❌ Arduino 연결 실패: {e}")
    print("시뮬레이션 모드로 실행됩니다. (LED 제어 안 함)")

# ==========================================
# [기능 0] 메모리 정리
# ==========================================
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# [아두이노 LED 제어 기능]
# ==========================================
def control_light(emotion: str):
    if not ser:
        print("[시뮬레이션] LED 제어 신호를 보내야 하지만, 아두이노가 연결되지 않았습니다.")
        return
    command = 'O'
    if emotion == "excitement":    command = 'E'
    elif emotion == "amusement":   command = 'A'
    elif emotion == "awe":         command = 'W'
    elif emotion == "contentment": command = 'C'
    elif emotion == "anger":       command = 'R'
    elif emotion == "fear":        command = 'F'
    elif emotion == "sadness":     command = 'S'
    elif emotion == "disgust":     command = 'D'
    try:
        ser.write(command.encode('utf-8'))
        print(f"[IoT 제어] 감정 '{emotion}'에 대한 신호 '{command}'를 아두이노로 전송했습니다.")
    except Exception as e:
        print(f"❌ 아두이노로 신호 전송 중 오류 발생: {e}")

# ==========================================
# [기능 1] CLIP 감정 분석
# ==========================================
class CLIPEmotionHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=8):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.classifier(x)

def get_emotion(pil_img):
    print("\n-------------------------------------------")
    print("[Step 1] Loading CLIP for Emotion Analysis...")
    emotion = "neutral" # 기본값 설정
    try:
        clip_backbone = CLIPModel.from_pretrained(CLIP_CKPT).to(DEVICE).eval()
        clip_proc = HFCLIPProcessor.from_pretrained(CLIP_CKPT)
        emotion_head = CLIPEmotionHead(num_classes=len(EMO_LABELS)).to(DEVICE)

        if os.path.exists(CLIP_EMOTION_CKPT):
            ckpt = torch.load(CLIP_EMOTION_CKPT, map_location=DEVICE)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            emotion_head.load_state_dict(state_dict, strict=False)
            emotion_head.eval()
            print("✅ Custom Emotion Checkpoint Loaded.")
        else:
            print("⚠️ Checkpoint not found. Using random weights.")

        inputs = clip_proc(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_feats = clip_backbone.get_image_features(**inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = emotion_head(img_feats)
            probs = torch.softmax(logits, dim=-1)[0]

        top_id = int(torch.argmax(probs).item())
        emotion = EMO_LABELS[top_id]
        prob = float(probs[top_id].item())

        print(f"✅ Detected: {emotion.upper()} ({prob*100:.1f}%)")

    except Exception as e:
        print(f"❌ Error in CLIP: {e}")
    finally:
        # 오류 발생 여부와 상관없이 메모리 정리 실행
        del clip_backbone, emotion_head, clip_proc
        clear_memory()
        return emotion

# ==========================================
# [기능 2] BLIP 캡션 생성
# ==========================================
def get_caption(pil_img):
    print("\n-------------------------------------------")
    print("[Step 2] Loading BLIP for Captioning...")
    caption = "an image" # 기본값 설정
    try:
        processor = BlipProcessor.from_pretrained(BLIP_CKPT)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_CKPT).to(DEVICE).eval()
        inputs = processor(pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
        print(f"✅ Caption: \"{caption}\"")
    except Exception as e:
        print(f"❌ Error in BLIP: {e}")
    finally:
        del model, processor
        clear_memory()
        return caption

# ==========================================
# [기능 3] CLAP 음악 추천
# ==========================================
def search_music(query_text):
    print("\n-------------------------------------------")
    print("[Step 3] Loading CLAP & DB for Music Search...")

    if not os.path.exists(MUSIC_EMB_PATH) or not os.path.exists(META_PATH):
        print("❌ Error: DB Files missing!")
        return

    try:
        # 1. DB 로드 (CPU 메모리 사용)
        song_embeds_cpu = torch.from_numpy(np.load(MUSIC_EMB_PATH)).float()
        song_embeds_cpu = song_embeds_cpu / (song_embeds_cpu.norm(dim=-1, keepdim=True) + 1e-8)

        song_meta = []
        with open(META_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                song_meta.append(row)
        print(f"✅ Loaded {len(song_meta)} songs from DB.")

        # 2. 모델 로드 및 텍스트 임베딩 (GPU 사용)
        model = ClapModel.from_pretrained(CLAP_CKPT).to(DEVICE).eval()
        processor = ClapProcessor.from_pretrained(CLAP_CKPT)
        print(f"Final Query: '{query_text}'")
        inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # 3. 유사도 계산
        sims = text_emb @ song_embeds_cpu.to(DEVICE).T

        # 4. Top 3 결과 출력
        scores, idx = sims[0].topk(3)

        print("\n🎵 Top 3 Recommended Songs:")
        for i, score in zip(idx.cpu().numpy(), scores.cpu().numpy()):
            row = song_meta[i]
            # CSV 파일의 'path' 컬럼을 제목처럼 사용
            path_title = row.get("path") or "Unknown Path"
            print(f"  - [{score:.4f}] {path_title}")

    except Exception as e:
        print(f"❌ Error in CLAP/Music-Search: {e}")
    finally:
        del model, processor, text_emb, sims, song_embeds_cpu
        clear_memory()

# ==========================================
# [Main] 메인 실행 로직
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: '{TEST_IMAGE_PATH}' not found.")
    else:
        try:
            print(f"✅ Processing image: {TEST_IMAGE_PATH}")
            image = Image.open(TEST_IMAGE_PATH).convert("RGB")

            emotion_result = get_emotion(image)
            control_light(emotion_result)
            caption_result = get_caption(image)
            
            search_music(f"{emotion_result} mood. {caption_result}")

            print("\n✅ All Pipeline Finished Successfully!")

        except KeyboardInterrupt:
             print("\n\n🛑 사용자에 의해 프로그램이 중단되었습니다.")
        finally:
             if ser and ser.is_open:
                 ser.close()
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def control_light(emotion: str):
    if not ser: return
    command = 'O'
    if emotion == "excitement":    command = 'E'
    elif emotion == "amusement":   command = 'A'
    elif emotion == "awe":         command = 'W'
    elif emotion == "contentment": command = 'C'
    elif emotion == "anger":       command = 'R'
    elif emotion == "fear":        command = 'F'
    elif emotion == "sadness":     command = 'S'
    elif emotion == "disgust":     command = 'D'
    try:
        ser.write(command.encode('utf-8'))
        print(f"💡 [IoT 제어] 감정 '{emotion}'에 대한 신호 '{command}'를 아두이노로 전송했습니다.")
    except Exception as e:
        print(f"❌ 아두이노로 신호 전송 중 오류 발생: {e}")

class CLIPEmotionHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=8):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.classifier(x)

def get_emotion(pil_img):
    print("\n-------------------------------------------")
    print("[Step 1] Loading CLIP for Emotion Analysis...")
    emotion = "neutral"
    try:
        clip_backbone = CLIPModel.from_pretrained(CLIP_CKPT).to(DEVICE).eval()
        clip_proc = HFCLIPProcessor.from_pretrained(CLIP_CKPT)
        emotion_head = CLIPEmotionHead(num_classes=len(EMO_LABELS)).to(DEVICE)
        if os.path.exists(CLIP_EMOTION_CKPT):
            ckpt = torch.load(CLIP_EMOTION_CKPT, map_location=DEVICE)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            emotion_head.load_state_dict(state_dict, strict=False)
            emotion_head.eval()
        inputs = clip_proc(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            img_feats = clip_backbone.get_image_features(**inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = emotion_head(img_feats)
            probs = torch.softmax(logits, dim=-1)[0]
        top_id = int(torch.argmax(probs).item())
        emotion = EMO_LABELS[top_id]
    except Exception as e:
        print(f"❌ Error in CLIP: {e}")
    finally:
        del clip_backbone, emotion_head, clip_proc
        clear_memory()
        return emotion

def get_caption(pil_img):
    print("\n-------------------------------------------")
    print("[Step 2] Loading BLIP for Captioning...")
    caption = "an image"
    try:
        processor = BlipProcessor.from_pretrained(BLIP_CKPT)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_CKPT).to(DEVICE).eval()
        inputs = processor(pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        print(f"❌ Error in BLIP: {e}")
    finally:
        del model, processor
        clear_memory()
        return caption

def search_music(query_text):
    print("\n-------------------------------------------")
    print("[Step 3] Loading CLAP & DB for Music Search...")
    if not os.path.exists(MUSIC_EMB_PATH) or not os.path.exists(META_PATH):
        print("❌ Error: DB Files missing!")
        return
    try:
        song_embeds_cpu = torch.from_numpy(np.load(MUSIC_EMB_PATH)).float()
        song_embeds_cpu = song_embeds_cpu / (song_embeds_cpu.norm(dim=-1, keepdim=True) + 1e-8)
        song_meta = []
        with open(META_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                song_meta.append(row)
        model = ClapModel.from_pretrained(CLAP_CKPT).to(DEVICE).eval()
        processor = ClapProcessor.from_pretrained(CLAP_CKPT)
        inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        sims = text_emb @ song_embeds_cpu.to(DEVICE).T
        scores, idx = sims[0].topk(3)
        print("\n🎵 Top 3 Recommended Songs:")
        for i, score in zip(idx.cpu().numpy(), scores.cpu().numpy()):
            row = song_meta[i]
            path_title = row.get("path") or "Unknown Path"
            print(f"  - [{score:.4f}] {path_title}")
    except Exception as e:
        print(f"❌ Error in CLAP/Music-Search: {e}")
    finally:
        del model, processor, text_emb, sims, song_embeds_cpu
        clear_memory()

if __name__ == "__main__":
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: '{TEST_IMAGE_PATH}' not found.")
    else:
        try:
            image = Image.open(TEST_IMAGE_PATH).convert("RGB")
            emotion_result = get_emotion(image)
            control_light(emotion_result)
            caption_result = get_caption(image)
            search_music(f"{emotion_result} mood. {caption_result}")
            print("\n✅ All Pipeline Finished Successfully!")
        except KeyboardInterrupt:
             print("\n\n🛑 User interrupted the program.")
        finally:
             if ser and ser.is_open:
                 ser.close()

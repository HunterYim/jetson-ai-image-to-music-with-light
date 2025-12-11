/*
  AIoT Mood Light - Final Version (v15)
  - Jetson으로부터 감정 신호를 받아 NeoPixel LED로 표현하는 무드등 제어 코드
*/

#include <Adafruit_NeoPixel.h>

// --- 설정 (Settings) ---
#define LED_PIN     6  // NeoPixel 데이터 핀
#define LED_COUNT   16 // 연결된 LED 개수

// NeoPixel 객체 생성
Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

// 현재 감정 상태를 저장하는 변수. 'O'(Off)로 시작.
char currentSignal = 'O'; 

void setup() {
  // 시리얼 통신 시작 (Jetson과 속도를 맞춰야 함)
  Serial.begin(9600);

  // NeoPixel 초기화
  strip.begin();
  strip.setBrightness(120); // 전체 밝기 (0-255)
  strip.show(); // 모든 LED를 끈 상태로 시작
  
  Serial.println("Arduino Mood Light Ready.");
}

void loop() {
  // Jetson으로부터 새로운 신호가 들어왔는지 확인
  if (Serial.available() > 0) {
    char newSignal = Serial.read();
    // 이전과 다른 신호일 경우에만 현재 상태 업데이트
    if (newSignal != currentSignal) {
      currentSignal = newSignal;
      Serial.print("New State: ");
      Serial.println(currentSignal);
    }
  }

  // 현재 감정 상태에 맞는 애니메이션 함수를 계속해서 호출
  switch (currentSignal) {
    case 'E': uniformHueWave(0, 65535, 0.8); break;
    case 'A': auroraWave(strip.Color(255, 215, 0), strip.Color(255, 250, 5), strip.Color(173, 255, 47), 0.6); break;
    case 'C': auroraWave(strip.Color(139, 0, 0), strip.Color(255, 140, 0), strip.Color(0, 0, 0), 0.3); break;
    case 'W': auroraWave(strip.Color(255, 105, 180), strip.Color(255, 165, 0), strip.Color(40, 0, 80), 0.1); break;
    case 'R': auroraWave(strip.Color(255, 0, 0), strip.Color(139, 0, 0), strip.Color(0, 0, 0), 0.9); break;
    case 'F': auroraWave(strip.Color(0, 0, 255), strip.Color(255, 0, 0), strip.Color(0, 0, 0), 0.7); break;
    case 'S': auroraWave(strip.Color(0, 0, 80), strip.Color(0, 0, 200), strip.Color(0, 0, 0), 0.08); break;
    case 'D': auroraWave(strip.Color(85, 107, 47), strip.Color(154, 205, 50), strip.Color(20, 20, 20), 0.4); break;
    case 'O':
    default:
      auroraWave(strip.Color(0,0,0), strip.Color(0,0,0), strip.Color(0,0,0), 0.5); // 서서히 꺼짐
      break;
  }
}


// --- 애니메이션 엔진 ---

// 엔진 1: '균일 패턴 (Hue)' - 무지개 색상환 기반 그라데이션 파도
void uniformHueWave(uint16_t startHue, uint16_t endHue, float speed) {
  float progress = (sin(millis() * speed / 1000.0) + 1) / 2.0;
  uint16_t currentHue = startHue + (uint16_t)((float)(endHue - startHue) * progress);
  for (int i = 0; i < strip.numPixels(); i++) {
    int pixelHue = currentHue + (i * 65536L / strip.numPixels());
    strip.setPixelColor(i, strip.gamma32(strip.ColorHSV(pixelHue)));
  }
  strip.show();
}

// 엔진 2: '오로라 패턴 (3색)' - 3가지 색상이 섞이며 일렁이는 효과
void auroraWave(uint32_t color1, uint32_t color2, uint32_t color3, float speed) {
  uint8_t r1 = (color1 >> 16) & 0xFF, g1 = (color1 >> 8) & 0xFF, b1 = color1 & 0xFF;
  uint8_t r2 = (color2 >> 16) & 0xFF, g2 = (color2 >> 8) & 0xFF, b2 = color2 & 0xFF;
  uint8_t r3 = (color3 >> 16) & 0xFF, g3 = (color3 >> 8) & 0xFF, b3 = color3 & 0xFF;

  for (int i = 0; i < strip.numPixels(); i++) {
    // 서로 다른 주기를 가진 3개의 sin 파동을 혼합하여 각 픽셀의 색상을 결정
    float blend1 = (sin(millis() * speed * 1.0 / 1000.0 + i * 0.2) + 1) / 2.0;
    float blend2 = (sin(millis() * speed * 0.66 / 1000.0 + i * 0.35) + 1) / 2.0;
    float blend3 = (sin(millis() * speed * 0.33 / 1000.0 + i * 0.5) + 1) / 2.0;
    float totalBlend = blend1 + blend2 + blend3;

    // 각 색상 채널별로 가중 평균을 계산
    uint8_t r = (r1 * blend1 + r2 * blend2 + r3 * blend3) / totalBlend;
    uint8_t g = (g1 * blend1 + g2 * blend2 + g3 * blend3) / totalBlend;
    uint8_t b = (b1 * blend1 + b2 * blend2 + b3 * blend3) / totalBlend;
    
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();
}

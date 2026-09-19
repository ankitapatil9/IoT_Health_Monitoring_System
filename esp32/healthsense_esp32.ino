#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <MAX30105.h>
#include <spo2_algorithm.h>

// ============================================================
// USER CONFIGURATION
// ============================================================

// Wi-Fi
const char* WIFI_SSID = "Miss Khan";
const char* WIFI_PASS = "maliha.03";

// Backend
const char* SERVER_URL = "http://192.168.83.177:5000/api/telemetry";
const char* DEVICE_API_KEY = "dev-api-key-123";
const char* DEVICE_ID = "ESP32_HEALTH_NODE_01";
const char* PATIENT_ID = "PATIENT_DEMO_2026";

// Future temperature sensor:
// Change to true after connecting a DS18B20 and installing the libraries.
#define ENABLE_DS18B20 false

// ============================================================
// GPIO CONFIGURATION
// ============================================================

#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22

#define AD8232_OUTPUT_PIN 34
#define AD8232_LO_PLUS_PIN 32
#define AD8232_LO_MINUS_PIN 33

#define DS18B20_DATA_PIN 4

// ============================================================
// OLED
// ============================================================

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define OLED_ADDRESS 0x3C

Adafruit_SSD1306 display(
  SCREEN_WIDTH,
  SCREEN_HEIGHT,
  &Wire,
  OLED_RESET
);

// ============================================================
// MAX30102
// ============================================================

MAX30105 particleSensor;

#define MAX30102_ADDRESS 0x57
#define BUFFER_SIZE 100
#define FINGER_THRESHOLD 50000

uint32_t irBuffer[BUFFER_SIZE];
uint32_t redBuffer[BUFFER_SIZE];
int bufferIndex = 0;

int32_t spo2 = 0;
int8_t validSpO2 = 0;
int32_t heartRate = 0;
int8_t validHeartRate = 0;

uint32_t latestIR = 0;
uint32_t latestRED = 0;

// ============================================================
// ECG
// ============================================================

#define ECG_SAMPLE_RATE 250
#define ECG_SAMPLE_INTERVAL_US (1000000UL / ECG_SAMPLE_RATE)

int waveform[SCREEN_WIDTH];

float baseline = 2048.0f;
float ECG_GAIN = 0.035f;

bool currentLeadOff = true;

// Small telemetry buffer: 187 samples = about 0.75 sec at 250 Hz
#define ECG_TELEMETRY_SIZE 187
float ecgTelemetryBuffer[ECG_TELEMETRY_SIZE];
int ecgTelemetryIndex = 0;

// ============================================================
// OPTIONAL DS18B20
// ============================================================

#if ENABLE_DS18B20
  #include <OneWire.h>
  #include <DallasTemperature.h>

  OneWire oneWire(DS18B20_DATA_PIN);
  DallasTemperature tempSensor(&oneWire);

  float currentTemperature = NAN;
  bool temperatureValid = false;
#endif

// ============================================================
// TIMERS
// ============================================================

unsigned long lastDisplay = 0;
unsigned long lastTelemetrySend = 0;
unsigned long lastWiFiCheck = 0;
unsigned long lastECGSample = 0;

// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("========================================");
  Serial.println(" ESP32 HEALTH MONITOR");
  Serial.println(" AD8232 + MAX30102 + OLED");
  Serial.println(" Optional DS18B20 + Wi-Fi Backend");
  Serial.println("========================================");

  // ---------------- I2C ----------------
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);

  // ---------------- AD8232 ----------------
  pinMode(AD8232_LO_PLUS_PIN, INPUT);
  pinMode(AD8232_LO_MINUS_PIN, INPUT);

  analogReadResolution(12);
  analogSetPinAttenuation(AD8232_OUTPUT_PIN, ADC_11db);

  // ---------------- OLED ----------------
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("[ERROR] OLED not found. Check VCC/GND/SDA/SCL.");
    while (true) {
      delay(1000);
    }
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(20, 15);
  display.println("ESP32 HEALTH");
  display.setCursor(20, 32);
  display.println("Starting...");
  display.display();
  delay(1500);

  // ---------------- MAX30102 ----------------
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST, MAX30102_ADDRESS)) {
    Serial.println("[ERROR] MAX30102 not found. Check VCC/GND/SDA/SCL.");

    display.clearDisplay();
    display.setCursor(5, 20);
    display.println("MAX30102 NOT FOUND");
    display.setCursor(5, 35);
    display.println("Check I2C wiring");
    display.display();

    while (true) {
      delay(1000);
    }
  }

  // MAX30102 configuration
  byte ledBrightness = 60;
  byte sampleAverage = 4;
  byte ledMode = 2;       // RED + IR
  int sampleRate = 100;   // 100 samples/sec
  int pulseWidth = 411;
  int adcRange = 4096;

  particleSensor.setup(
    ledBrightness,
    sampleAverage,
    ledMode,
    sampleRate,
    pulseWidth,
    adcRange
  );

  // ---------------- Optional DS18B20 ----------------
#if ENABLE_DS18B20
  tempSensor.begin();
  Serial.println("[TEMP] DS18B20 enabled");
#else
  Serial.println("[TEMP] DS18B20 disabled - ready for future use");
#endif

  // ---------------- Buffers ----------------
  for (int i = 0; i < SCREEN_WIDTH; i++) {
    waveform[i] = 38;
  }

  for (int i = 0; i < ECG_TELEMETRY_SIZE; i++) {
    ecgTelemetryBuffer[i] = 0.0f;
  }

  // ---------------- Wi-Fi ----------------
  connectWiFi();

  lastECGSample = micros();

  Serial.println("[SYSTEM] Initialization complete.");
  Serial.println();
}

// ============================================================
// WIFI
// ============================================================

void connectWiFi() {
  if (String(WIFI_SSID) == "YOUR_WIFI_SSID") {
    Serial.println("[WiFi] Configure WIFI_SSID/WIFI_PASS first.");
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("[WiFi] Connecting");

  int retries = 0;

  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print(".");
    retries++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("[WiFi] Connected. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[WiFi] Not connected. Device will continue offline.");
  }
}

// ============================================================
// ECG
// ============================================================

void sampleECG() {
  int rawECG = analogRead(AD8232_OUTPUT_PIN);

  currentLeadOff =
    digitalRead(AD8232_LO_PLUS_PIN) ||
    digitalRead(AD8232_LO_MINUS_PIN);

  // Slowly track DC baseline
  baseline = baseline * 0.995f + rawECG * 0.005f;

  float ecgAC = rawECG - baseline;

  // OLED waveform
  int y = 38 - (int)(ecgAC * ECG_GAIN);
  y = constrain(y, 14, 50);

  for (int i = 0; i < SCREEN_WIDTH - 1; i++) {
    waveform[i] = waveform[i + 1];
  }

  waveform[SCREEN_WIDTH - 1] = currentLeadOff ? 38 : y;

  // Telemetry
  ecgTelemetryBuffer[ecgTelemetryIndex] = ecgAC;
  ecgTelemetryIndex =
    (ecgTelemetryIndex + 1) % ECG_TELEMETRY_SIZE;
}

// ============================================================
// MAX30102
// ============================================================

void processMAX30102() {
  particleSensor.check();

  while (particleSensor.available()) {
    latestIR = particleSensor.getFIFOIR();
    latestRED = particleSensor.getFIFORed();

    if (latestIR >= FINGER_THRESHOLD) {

      // Store EVERY available sample.
      // The MAX30102 is configured for 100 Hz.
      if (bufferIndex < BUFFER_SIZE) {
        irBuffer[bufferIndex] = latestIR;
        redBuffer[bufferIndex] = latestRED;
        bufferIndex++;
      }

      // Calculate after 100 samples.
      if (bufferIndex >= BUFFER_SIZE) {

        maxim_heart_rate_and_oxygen_saturation(
          irBuffer,
          BUFFER_SIZE,
          redBuffer,
          &spo2,
          &validSpO2,
          &heartRate,
          &validHeartRate
        );

        Serial.print("[MAX30102] HR: ");
        if (validHeartRate) {
          Serial.print(heartRate);
          Serial.print(" BPM");
        } else {
          Serial.print("INVALID");
        }

        Serial.print(" | SpO2: ");

        if (validSpO2) {
          Serial.print(spo2);
          Serial.println("%");
        } else {
          Serial.println("INVALID");
        }

        // Keep the latest 75 samples and collect 25 new ones.
        for (int i = 25; i < BUFFER_SIZE; i++) {
          irBuffer[i - 25] = irBuffer[i];
          redBuffer[i - 25] = redBuffer[i];
        }

        bufferIndex = 75;
      }

    } else {
      // No finger
      bufferIndex = 0;
      spo2 = 0;
      validSpO2 = 0;
      heartRate = 0;
      validHeartRate = 0;
    }

    particleSensor.nextSample();
  }
}

// ============================================================
// OLED
// ============================================================

void updateDisplay() {
  display.clearDisplay();

  // Header
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("ECG:");
  display.print(currentLeadOff ? "OFF" : "OK");

  display.setCursor(60, 0);
  display.print("SpO2:");

  if (latestIR < FINGER_THRESHOLD) {
    display.print("--");
  } else if (validSpO2) {
    display.print(spo2);
    display.print("%");
  } else {
    display.print("...");
  }

  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  // ECG waveform
  for (int x = 1; x < SCREEN_WIDTH; x++) {
    display.drawLine(
      x - 1,
      waveform[x - 1],
      x,
      waveform[x],
      SSD1306_WHITE
    );
  }

  // Bottom information
  display.fillRect(0, 52, 128, 12, BLACK);
  display.setCursor(0, 54);

  if (currentLeadOff) {
    display.print("ECG: Attach electrodes");
  }
  else if (latestIR < FINGER_THRESHOLD) {
    display.print("Place finger on MAX");
  }
  else if (!validSpO2) {
    display.print("Measuring HR/SpO2...");
  }
  else {
    display.print("HR:");
    if (validHeartRate) {
      display.print(heartRate);
    } else {
      display.print("--");
    }

    display.print(" SpO2:");
    display.print(spo2);
    display.print("%");
  }

  display.display();
}

// ============================================================
// JSON TELEMETRY
// ============================================================

void sendTelemetry() {
  JsonDocument doc;

  doc["deviceId"] = DEVICE_ID;
  doc["patientId"] = PATIENT_ID;
  doc["timestampMillis"] = millis();

  bool fingerDetected = latestIR >= FINGER_THRESHOLD;

  // HR
  if (fingerDetected && validHeartRate) {
    doc["heartRate"] = heartRate;
    doc["heartRateValid"] = true;
  } else {
    doc["heartRate"] = nullptr;
    doc["heartRateValid"] = false;
  }

  // SpO2
  if (fingerDetected && validSpO2) {
    doc["spo2"] = spo2;
    doc["spo2Valid"] = true;
  } else {
    doc["spo2"] = nullptr;
    doc["spo2Valid"] = false;
  }

  // ECG
  JsonArray ecgArray = doc["ecg"].to<JsonArray>();

  if (!currentLeadOff) {
    for (int i = 0; i < ECG_TELEMETRY_SIZE; i++) {
      ecgArray.add(
        ecgTelemetryBuffer[
          (ecgTelemetryIndex + i) % ECG_TELEMETRY_SIZE
        ]
      );
    }
  }

  doc["ecgLeadOff"] = currentLeadOff;
  doc["ecgSignalQuality"] =
    currentLeadOff ? "LEAD_OFF" : "SIGNAL_PRESENT";

  doc["ppgSignalQuality"] =
    fingerDetected
      ? (validSpO2 ? "GOOD" : "MEASURING")
      : "NO_FINGER";

  doc["fingerDetected"] = fingerDetected;

#if ENABLE_DS18B20
  if (temperatureValid) {
    doc["temperature"] = currentTemperature;
    doc["temperatureValid"] = true;
  } else {
    doc["temperature"] = nullptr;
    doc["temperatureValid"] = false;
  }
#else
  doc["temperature"] = nullptr;
  doc["temperatureValid"] = false;
  doc["temperatureSensor"] = "NOT_INSTALLED";
#endif

  doc["wifiRssi"] =
    WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : -127;

  doc["esp32Status"] = "ONLINE";

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  // Always print telemetry over Serial
  Serial.print("[TELEMETRY_JSON]");
  Serial.println(jsonPayload);

  // Send to backend
  if (WiFi.status() == WL_CONNECTED &&
      String(SERVER_URL) != "http://192.168.83.177:5000/api/telemetry") {

    HTTPClient http;

    http.begin(SERVER_URL);
    http.setTimeout(3000);
    http.addHeader("Content-Type", "application/json");
    http.addHeader("x-device-api-key", DEVICE_API_KEY);

    int httpCode = http.POST(jsonPayload);

    if (httpCode > 0) {
      Serial.printf("[HTTP] POST response: %d\n", httpCode);
    } else {
      Serial.printf(
        "[HTTP] POST failed: %s\n",
        http.errorToString(httpCode).c_str()
      );
    }

    http.end();
  }
}

// ============================================================
// OPTIONAL TEMPERATURE
// ============================================================

#if ENABLE_DS18B20
void updateTemperature() {
  static unsigned long lastTemperatureRead = 0;

  if (millis() - lastTemperatureRead >= 2000) {
    lastTemperatureRead = millis();

    tempSensor.requestTemperatures();

    float t = tempSensor.getTempCByIndex(0);

    if (t != DEVICE_DISCONNECTED_C && t > -55.0f && t < 125.0f) {
      currentTemperature = t;
      temperatureValid = true;
    } else {
      temperatureValid = false;
    }
  }
}
#endif

// ============================================================
// LOOP
// ============================================================

void loop() {

  unsigned long nowMicros = micros();
  unsigned long nowMillis = millis();

  // ---------------- ECG @ 250 Hz ----------------
  while ((unsigned long)(nowMicros - lastECGSample)
         >= ECG_SAMPLE_INTERVAL_US) {

    lastECGSample += ECG_SAMPLE_INTERVAL_US;
    sampleECG();

    nowMicros = micros();
  }

  // ---------------- MAX30102 ----------------
  processMAX30102();

#if ENABLE_DS18B20
  // ---------------- Temperature ----------------
  updateTemperature();
#endif

  // ---------------- OLED every 50 ms ----------------
  if (nowMillis - lastDisplay >= 50) {
    lastDisplay = nowMillis;
    updateDisplay();
  }

  // ---------------- Backend every 3 sec ----------------
  if (nowMillis - lastTelemetrySend >= 3000) {
    lastTelemetrySend = nowMillis;
    sendTelemetry();
  }

  // ---------------- Wi-Fi reconnect ----------------
  if (nowMillis - lastWiFiCheck >= 10000) {
    lastWiFiCheck = nowMillis;

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WiFi] Disconnected. Reconnecting...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  }
}

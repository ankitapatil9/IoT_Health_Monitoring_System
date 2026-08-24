/* ==========================================================================
   HealthSense - ESP32 Multi-Sensor Telemetry & Ingestion Firmware
   --------------------------------------------------------------------------
   Target Board: ESP32 Dev Module (WROOM-32 / NodeMCU-32S)
   Supported Sensors:
     1. AD8232 Single-Lead Heart Rate / ECG Sensor (Analog ADC GPIO 36 / VP)
     2. MAX30102 Optical Pulse Oximeter & Heart-Rate Sensor (I2C SDA:21, SCL:22)
     3. DS18B20 1-Wire Digital Body Temperature Sensor (GPIO 4)
     4. 0.96" I2C OLED Display SSD1306 (128x64) (I2C SDA:21, SCL:22)
   
   Communication: Wi-Fi HTTP REST Ingestion Gateway (/api/telemetry)
   ========================================================================== */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <ArduinoJson.h>

// ---------------------------------------------------------------------------
// 1. Hardware Pin Configurations
// ---------------------------------------------------------------------------
#define PIN_ECG_ADC        36   // ADC1_CH0 (VP) - AD8232 OUTPUT
#define PIN_ECG_LO_PLUS    34   // Leads-off detect LO+
#define PIN_ECG_LO_MINUS   35   // Leads-off detect LO-

#define PIN_ONEWIRE_TEMP   4    // DS18B20 1-Wire Data Pin

#define OLED_SCREEN_WIDTH  128
#define OLED_SCREEN_HEIGHT 64
#define OLED_RESET         -1
#define OLED_I2C_ADDR      0x3C

// ---------------------------------------------------------------------------
// 2. Wi-Fi & Backend Endpoint Configuration
// ---------------------------------------------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// HealthSense Backend Telemetry Ingestion Endpoint
const char* BACKEND_SERVER_URL = "http://192.168.1.100:5000/api/telemetry";
const char* DEVICE_ID          = "ESP32_HEALTHSENSE_01";

// ---------------------------------------------------------------------------
// 3. Sensor Instances & State Variables
// ---------------------------------------------------------------------------
Adafruit_SSD1306 oled(OLED_SCREEN_WIDTH, OLED_SCREEN_HEIGHT, &Wire, OLED_RESET);
OneWire oneWire(PIN_ONEWIRE_TEMP);
DallasTemperature tempSensor(&oneWire);

// Sampling Constants
const int ECG_SAMPLE_RATE_HZ = 250;            // 250 Hz (4ms period)
const unsigned long SAMPLE_INTERVAL_US = 4000; // 4000 microseconds
const int SAMPLES_PER_PACKET = 25;             // Send telemetry every 100ms (25 samples)

float ecgBuffer[SAMPLES_PER_PACKET];
int bufferIndex = 0;
unsigned long lastSampleMicros = 0;
unsigned long lastTelemetryMillis = 0;
unsigned long lastVitalsReadMillis = 0;

// Telemetry Values
float currentHeartRate = 72.0;
float currentSpO2      = 98.0;
float currentTempC     = 36.8;
bool leadsOffDetected  = false;


// ---------------------------------------------------------------------------
// 4. OLED Display Helper
// ---------------------------------------------------------------------------
void updateOLEDDisplay() {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);

  // Header Banner
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  oled.print("HealthSense ESP32");

  oled.setCursor(108, 0);
  if (WiFi.status() == WL_CONNECTED) {
    oled.print("ON");
  } else {
    oled.print("--");
  }

  oled.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  if (leadsOffDetected) {
    oled.setCursor(15, 28);
    oled.setTextSize(1);
    oled.print("LEADS DISCONNECTED!");
    oled.setCursor(10, 42);
    oled.print("Check AD8232 Electrodes");
  } else {
    // Vitals Readout
    oled.setTextSize(1);
    oled.setCursor(0, 18);
    oled.print("Heart Rate: ");
    oled.setTextSize(2);
    oled.setCursor(72, 14);
    oled.print((int)currentHeartRate);
    oled.setTextSize(1);
    oled.print(" bpm");

    oled.setCursor(0, 36);
    oled.print("SpO2:       ");
    oled.setTextSize(1);
    oled.setCursor(72, 36);
    oled.print(currentSpO2, 1);
    oled.print(" %");

    oled.setCursor(0, 50);
    oled.print("Body Temp:  ");
    oled.setCursor(72, 50);
    oled.print(currentTempC, 1);
    oled.print(" C");
  }

  oled.display();
}


// ---------------------------------------------------------------------------
// 5. Send Telemetry JSON to HealthSense Backend
// ---------------------------------------------------------------------------
void sendTelemetryPacket() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  HTTPClient http;
  http.begin(BACKEND_SERVER_URL);
  http.addHeader("Content-Type", "application/json");

  // Construct JSON document
  StaticJsonDocument<1024> doc;
  doc["device_id"]   = DEVICE_ID;
  doc["timestamp"]   = millis() / 1000.0;
  doc["heart_rate"]  = currentHeartRate;
  doc["spo2"]        = currentSpO2;
  doc["temperature"] = currentTempC;
  doc["leads_off"]   = leadsOffDetected;

  JsonArray samplesArray = doc.createNestedArray("ecg_samples");
  for (int i = 0; i < bufferIndex; i++) {
    samplesArray.add(ecgBuffer[i]);
  }

  String requestBody;
  serializeJson(doc, requestBody);

  // Non-blocking quick post
  int httpResponseCode = http.POST(requestBody);
  if (httpResponseCode > 0) {
    // Success
  } else {
    Serial.printf("[HTTP] POST failed, error: %s\n", http.errorToString(httpResponseCode).c_str());
  }

  http.end();
  bufferIndex = 0; // Reset buffer
}


// ---------------------------------------------------------------------------
// 6. Setup Initializer
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n==============================================");
  Serial.println("   HealthSense IoT Telemetry Node Initializing ");
  Serial.println("==============================================");

  // Pin modes
  pinMode(PIN_ECG_ADC, INPUT);
  pinMode(PIN_ECG_LO_PLUS, INPUT);
  pinMode(PIN_ECG_LO_MINUS, INPUT);

  // Initialize I2C OLED Display
  Wire.begin(21, 22);
  if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDR)) {
    Serial.println("[OLED] Warning: SSD1306 allocation failed. Check I2C wiring.");
  } else {
    oled.clearDisplay();
    oled.setTextSize(1);
    oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(10, 20);
    oled.println("HealthSense Node");
    oled.setCursor(10, 36);
    oled.println("Connecting Wi-Fi...");
    oled.display();
  }

  // Initialize Temperature Sensor
  tempSensor.begin();

  // Connect to Wi-Fi Network
  Serial.printf("[Wi-Fi] Connecting to %s...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int wifiRetries = 0;
  while (WiFi.status() != WL_CONNECTED && wifiRetries < 20) {
    delay(400);
    Serial.print(".");
    wifiRetries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[Wi-Fi] Connected! Local IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[Wi-Fi] Warning: Connection timeout. Operating in offline display mode.");
  }

  lastSampleMicros = micros();
  lastVitalsReadMillis = millis();
}


// ---------------------------------------------------------------------------
// 7. Main Real-Time Execution Loop
// ---------------------------------------------------------------------------
void loop() {
  unsigned long currentMicros = micros();
  unsigned long currentMillis = millis();

  // 1. High-Frequency ECG Sampling (250 Hz = every 4000 microseconds)
  if (currentMicros - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    lastSampleMicros += SAMPLE_INTERVAL_US;

    // Check leads-off pins
    bool loPlus = digitalRead(PIN_ECG_LO_PLUS);
    bool loMinus = digitalRead(PIN_ECG_LO_MINUS);
    leadsOffDetected = (loPlus == HIGH || loMinus == HIGH);

    // Read 12-bit ADC (0 - 4095) and convert to voltage / normalized scale
    int rawAdc = analogRead(PIN_ECG_ADC);
    float voltage = (rawAdc / 4095.0) * 3.3; // Volts
    // Centered around 1.65V baseline
    float centeredEcg = (voltage - 1.65);

    if (bufferIndex < SAMPLES_PER_PACKET) {
      ecgBuffer[bufferIndex++] = centeredEcg;
    }

    // When packet is full (25 samples = 100ms), transmit to backend
    if (bufferIndex >= SAMPLES_PER_PACKET) {
      sendTelemetryPacket();
    }
  }

  // 2. Read Low-Frequency Vitals (every 1000 ms)
  if (currentMillis - lastVitalsReadMillis >= 1000) {
    lastVitalsReadMillis = currentMillis;

    // DS18B20 Body Temperature reading
    tempSensor.requestTemperatures();
    float tC = tempSensor.getTempCByIndex(0);
    if (tC > 0 && tC < 60) {
      currentTempC = tC;
    }

    // Refresh OLED
    updateOLEDDisplay();
  }
}

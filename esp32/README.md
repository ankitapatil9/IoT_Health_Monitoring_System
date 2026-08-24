# HealthSense — ESP32 Firmware & Hardware Integration Guide

## 1. Hardware Bill of Materials (BOM)

| Component | Function | Interface / Pins | Operating Voltage |
| :--- | :--- | :--- | :--- |
| **ESP32 Dev Module (WROOM-32)** | Master Microcontroller & Wi-Fi Node | ADC / I2C / 1-Wire / Wi-Fi | 3.3V / 5V (USB) |
| **AD8232 ECG Sensor** | Single-Lead Electrocardiogram Monitor | Analog Output (GPIO 36 / VP), LO+/LO- | 3.3V |
| **MAX30102 Sensor** | Optical PPG Heart Rate & Pulse Oximetry | I2C (SDA: GPIO 21, SCL: GPIO 22) | 3.3V |
| **DS18B20 Sensor** | Digital Precision Body Temperature Sensor | 1-Wire (Data: GPIO 4) with 4.7kΩ pullup | 3.3V - 5V |
| **0.96" SSD1306 OLED** | Local Real-Time Display (128x64) | I2C (SDA: GPIO 21, SCL: GPIO 22, Addr: `0x3C`) | 3.3V |
| **TP4056 + 18650 Battery** | Li-ion Charging & 3.7V - 5V Power Supply | Power rails | 3.7V - 5.0V |

---

## 2. Wiring & Pin Connection Table

```
+------------------+-----------------+-----------------------------------+
| Sensor / Module  | ESP32 Pin       | Notes / Descriptions              |
+------------------+-----------------+-----------------------------------+
| AD8232 OUTPUT    | GPIO 36 (VP)    | Analog ECG Voltage (ADC1_CH0)     |
| AD8232 LO+       | GPIO 34         | Leads-Off Detection (+)           |
| AD8232 LO-       | GPIO 35         | Leads-Off Detection (-)           |
| AD8232 3.3V / GND| 3V3 / GND       | Power Supply                      |
+------------------+-----------------+-----------------------------------+
| MAX30102 SDA     | GPIO 21         | I2C Data                          |
| MAX30102 SCL     | GPIO 22         | I2C Clock                         |
| MAX30102 VIN/GND | 3V3 / GND       | Power Supply                      |
+------------------+-----------------+-----------------------------------+
| DS18B20 DATA     | GPIO 4          | 1-Wire (with 4.7kΩ pull-up to 3V3)|
| DS18B20 VCC/GND  | 3V3 / GND       | Power Supply                      |
+------------------+-----------------+-----------------------------------+
| OLED SSD1306 SDA | GPIO 21         | Shared I2C Data Line              |
| OLED SSD1306 SCL | GPIO 22         | Shared I2C Clock Line             |
| OLED VCC / GND   | 3V3 / GND       | Power Supply                      |
+------------------+-----------------+-----------------------------------+
```

---

## 3. Required Arduino IDE Libraries

In the Arduino IDE Library Manager (`Ctrl+Shift+I`), install:
1. **`Adafruit SSD1306`** & **`Adafruit GFX Library`** (for OLED display)
2. **`OneWire`** (by Paul Stoffregen)
3. **`DallasTemperature`** (by Miles Burton)
4. **`ArduinoJson`** (v6 or v7 by Benoit Blanchon)
5. **`SparkFun MAX3010x Pulse and Proximity Sensor Library`**

---

## 4. Configuration & Flashing

1. Open [`healthsense_esp32.ino`](file:///e:/HealthSense/esp32/healthsense_esp32.ino) in Arduino IDE.
2. Edit your Wi-Fi credentials:
   ```cpp
   const char* WIFI_SSID     = "YOUR_WIFI_NAME";
   const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   ```
3. Set your backend IP address:
   ```cpp
   const char* BACKEND_SERVER_URL = "http://<YOUR_COMPUTER_IP>:5000/api/telemetry";
   ```
4. Select Board: **`ESP32 Dev Module`**, Port: Select your COM port.
5. Click **Upload**.

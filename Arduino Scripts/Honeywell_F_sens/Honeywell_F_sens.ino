#include <Wire.h>

uint8_t mlx0_i2c = 0x28;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Wire.begin();
  //Wire.setClock(400000);
  delay(1);
  Serial.println("initialized");
}

void loop() {
  // put your main code here, to run repeatedly:
  Wire.requestFrom(0,12);
  Serial.print("Started");
  while(Wire.available()){
    char c = Wire.read();
    Serial.print(c);
  }
  Serial.print("Done");
  delay(500);
}

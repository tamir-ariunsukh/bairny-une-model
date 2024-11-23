// Arduino код
int ledPin = 13; // LED холбогдсон зүү (зөвхөн жишээ)

void setup() {
  pinMode(ledPin, OUTPUT);  // LED-г гаралт болгох
  Serial.begin(9600);       // Serial холболтыг эхлүүлэх
}

void loop() {
  if (Serial.available() > 0) {   // Python-аас өгөгдөл ирсэн эсэхийг шалгах
    char data = Serial.read();   // Мэдээлэл унших
    if (data == '1') {           // '1' - Унтаж байна
      digitalWrite(ledPin, HIGH); // LED-г асаах
    } else if (data == '0') {    // '0' - Сэрүүн байна
      digitalWrite(ledPin, LOW);  // LED-г унтраах
    }
  }
}

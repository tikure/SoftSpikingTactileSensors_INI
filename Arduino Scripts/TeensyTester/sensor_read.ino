/*
  5X ReSkin Board Example Code
  By: Tess Hellebrekers
  Date: October 22, 2021
  License: This code is public domain but you buy me a beer if you use this and we meet someday (Beerware license).

  Library: Heavily based on original MLX90393 library from Theodore Yapo (https://github.com/tedyapo/arduino-MLX90393)
  Use this fork (https://github.com/tesshellebrekers/arduino-MLX90393) to access additional burst mode commands
  
  Read the XYZ magnetic flux fields and temperature across all five chips on the 5X ReSkin board
  Print binary data over serial port
*/

void read_sensor()
{
  //Serial.println("Starting Loop");
  
  //continuously read the most recent data from the data registers and save to data
  mlx0.readBurstData(data0); //Read the values from the sensor
  mlx1.readBurstData(data1); 
  mlx2.readBurstData(data2); 
  mlx3.readBurstData(data3); 
  mlx4.readBurstData(data4); 

  //write string data over serial
  if(mode == 1){
    Serial.print(data0.x);
    Serial.print("\t");
    Serial.print(data0.y);
    Serial.print("\t");
    Serial.print(data0.z);
    Serial.print("\t");
    Serial.print(data0.t);
    Serial.print("\t");
    
    Serial.print(data1.x);
    Serial.print("\t");
    Serial.print(data1.y);
    Serial.print("\t");
    Serial.print(data1.z);
    Serial.print("\t");
    Serial.print(data1.t);
    Serial.print("\t");
    
    Serial.print(data2.x);
    Serial.print("\t");
    Serial.print(data2.y);
    Serial.print("\t");
    Serial.print(data2.z);
    Serial.print("\t");     
    Serial.print(data2.t);
    Serial.print("\t");
    
    Serial.print(data3.x);
    Serial.print("\t");
    Serial.print(data3.y);
    Serial.print("\t");
    Serial.print(data3.z);
    Serial.print("\t");
    Serial.print(data3.t);
    Serial.print("\t");
  
    Serial.print(data4.x);
    Serial.print("\t");
    Serial.print(data4.y);
    Serial.print("\t");
    Serial.print(data4.z);
    Serial.print("\t");
    Serial.print(data4.t);
    Serial.print("\t");
    Serial.print('\n');
    //Serial.println("Loop Complete");  
  }
  

  //adjust delay to achieve desired sampling rate
  delayMicroseconds(500);
  
}

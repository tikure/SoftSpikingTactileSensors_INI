//Modes
int mode = 0;


//Mechanorec Setup *************************************************
int led_pins[] = {29,30};
int butt_pins[] = {31,32};
int pot_pins[] = {A10,A11,A12};

//Sensor Parameters
const int dim = 4; //Dimension of Sensor Pixels
int rf = 2; //Receptive field
const int dim_neuron = 2; //dim - rf needs to be set constant to initialize array
int noize = 2;
//Setup Array
int sensor_pixels[dim][dim] =  {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
float neuron_voltages[dim_neuron][dim_neuron] = {};
float neuron_recovery[dim_neuron][dim_neuron] = {};

//Izhekevich Default Params
float timestep_ms_default = 0.1;
float timestep_ms = 0;
float a = 0.02; // time scale of recovery variable u. Smaller a gives slower recovery
float b = 0.2; // recovery variable associated with u. greater b coules it more strongly (basically sensitivity) !Do not change!
float c = -65; // after spike reset value !Do not change!
float d = 2; // after spike reset of recovery variable
float u = 0; //Initial Adaptation
float v = -65; //Initial Voltage
float I;
float I_DC = 0;
int IGain = 30; //Gain, sensor_pixels is multiplied by this



//Sensor Setup **********************************************
#include <Wire.h>
#include <MLX90393.h> 

//#define Serial SERIAL_PORT_USBVIRTUAL

MLX90393 mlx0;
MLX90393 mlx1;
MLX90393 mlx2;
MLX90393 mlx3;
MLX90393 mlx4;

MLX90393::txyz data0 = {0,0,0,0}; //Create a structure, called data, of four floats (t, x, y, and z)
MLX90393::txyz data1 = {0,0,0,0};
MLX90393::txyz data2 = {0,0,0,0};
MLX90393::txyz data3 = {0,0,0,0};
MLX90393::txyz data4 = {0,0,0,0};

/*
uint8_t mlx0_i2c = 0x0C; // these are the I2C addresses of the five chips that share one I2C bus
uint8_t mlx1_i2c = 0x13;
uint8_t mlx2_i2c = 0x12;
uint8_t mlx3_i2c = 0x10;
uint8_t mlx4_i2c = 0x11;
*/

uint8_t mlx0_i2c = 0x10; // these are the I2C addresses of the five chips that share one I2C bus
uint8_t mlx1_i2c = 0x18; //Blue = SDA, yellow = SCL
uint8_t mlx2_i2c = 0x19;
uint8_t mlx3_i2c = 0x1A;
uint8_t mlx4_i2c = 0x1B;





void setup() {
  // Mechanorec Setup ******************
  Serial.begin(115200);
  for (int i = 0; i < sizeof(led_pins);i++){
    pinMode(led_pins[i], OUTPUT);
  }
  pinMode(butt_pins[0], INPUT);
  pinMode(butt_pins[1], INPUT);
  pinMode(pot_pins[0], INPUT);
  pinMode(pot_pins[1], INPUT);
  pinMode(13,OUTPUT);digitalWrite(13, HIGH);


  // Sensor Setup ****************
  //Start serial port and wait until user opens it
  
  //while (!Serial) {delay(5);}
 
  //Serial.println("Booting");
  //Start default I2C bus for your board, set to fast mode (400kHz)
  
  Wire.begin();
  Wire.setClock(400000);
  delay(10);
  
  //start chips given address, -1 for no DRDY pin, and I2C bus object to use
  byte status = mlx0.begin(mlx0_i2c, -1, Wire);
  status = mlx1.begin(mlx1_i2c, -1, Wire);
  status = mlx2.begin(mlx2_i2c, -1, Wire);
  status = mlx3.begin(mlx3_i2c, -1, Wire);
  status = mlx4.begin(mlx4_i2c, -1, Wire);

  //default gain and digital filtering set up in the begin() function of library. Adjust here is you want to change them
  //mlx0.setGain(5); //accepts [0,7]
  //mlx0.setDigitalFiltering(5); // accepts [2,7]. refer to datasheet for hall configurations
  
  //Start burst mode for temp, x, y, and z for all chips
  //Burst mode: continuously sample temp, x, y, and z, at regular intervals without polling
  mlx0.startBurst(0xF);
  mlx1.startBurst(0xF);
  mlx2.startBurst(0xF);
  mlx3.startBurst(0xF);
  mlx4.startBurst(0xF);
  //Serial.println("Initialized");
  timestep_ms = millis();
}

void loop() {
  mode = map(analogRead(pot_pins[2]), 1, 700, 0, 1);
  //Serial.println(map(analogRead(pot_pins[2]), 1, 700, 0, 1)); 
  mechano_loop();     
  read_sensor(); 
}

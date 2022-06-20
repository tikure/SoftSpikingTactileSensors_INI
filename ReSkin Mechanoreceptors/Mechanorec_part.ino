// todo: Comments!, implement timing for ODE, implement frequency print, implement receptive field, implement rasterplot
//Define Variables

void mechano_loop() {
  read_inputs(); //Reads all inputs, sets sensor_pixels to button press and sets a,b
  for (int i = 0; i < dim_neuron ; i++) { //Loop through neuron array
    for (int j = 0; j < dim_neuron ; j++) {
      v = neuron_voltages[i][j]; //Get previous parameters for neuron
      u = neuron_recovery[i][j];
      I = random(-noize, noize); //Add noise
      for (int k = 0; k <= rf; k++) { //Loop through receptive field arround neuron, i+0 = i-1 since sensor_pixel array is larger than neuron array
        for (int q = 0; q <= rf; q++) {
          I = I+ I_DC + sensor_pixels[i + k][j + q];

        }
      }
      //Serial.print("I:");
      //Serial.println(I);
      timestep_ms = millis() - timestep_ms;
      v = v + timestep_ms_default * (0.04 * v * v + 5 * v + 140 - u + I); //Default izhekevich equations
      u = u + timestep_ms_default * (a * (b * v - u));
      if (v >= 30.0) { //SPike
        v = c; u += d;
        if(i == 1 && j == 0){
          digitalWrite(led_pins[0], HIGH);
          delay(1);
          digitalWrite(led_pins[0], LOW);
        }
        if(i == 1 && j == 1){
          digitalWrite(led_pins[1], HIGH);
          delay(1);
          digitalWrite(led_pins[1], LOW);
        }
        
      }

      neuron_voltages[i][j] = v; //Update neuron array with new v and u
      neuron_recovery[i][j] = u;

      //Serial.print(String(i) + String(j) + ": ");
      if(mode == 0){
        Serial.print(v);
        Serial.print("\t");
      }
    }
  }
  if(mode == 0){
        Serial.print("a");
        Serial.print(a);
        Serial.print("\t");
        Serial.print("d");
        Serial.print(d);
        Serial.print("\t");
        Serial.println("");
        }
  delay(10);
}


void read_inputs() {
  a = map(analogRead(pot_pins[0]), 1, 1023, 5, 80);
  a = a / 1000;
  d = map(analogRead(pot_pins[1]), 1, 1023, 0, 12);

  sensor_pixels[1][0] = map(digitalRead(butt_pins[0]),1,0,0,1) * IGain;
  sensor_pixels[1][1] = map(digitalRead(butt_pins[1]),1,0,0,1) * IGain;

  /*
  Serial.println(a);
  Serial.println(d);
  Serial.println(sensor_pixels[1][0]);
  Serial.println(sensor_pixels[1][1]);
  Serial.println("------------");
  delay(1000);
  */
}


// From Iziekevich.org - see also https://www.izhikevich.org/publications/figure1.pdf:
//      a         b       c       d        ???
//      0.02      0.2     -65      6       14 ;...    % tonic spiking
//      0.02      0.25    -65      6       0.5 ;...   % phasic spiking
//      0.02      0.2     -50      2       15 ;...    % tonic bursting
//      0.02      0.25    -55     0.05     0.6 ;...   % phasic bursting
//      0.02      0.2     -55     4        10 ;...    % mixed mode
//      0.01      0.2     -65     8        30 ;...    % spike frequency adaptation
//      0.02      -0.1    -55     6        0  ;...    % Class 1
//      0.2       0.26    -65     0        0  ;...    % Class 2
//      0.02      0.2     -65     6        7  ;...    % spike latency
//      0.05      0.26    -60     0        0  ;...    % subthreshold oscillations
//      0.1       0.26    -60     -1       0  ;...    % resonator
//      0.02      -0.1    -55     6        0  ;...    % integrator
//      0.03      0.25    -60     4        0;...      % rebound spike
//      0.03      0.25    -52     0        0;...      % rebound burst
//      0.03      0.25    -60     4        0  ;...    % threshold variability
//      1         1.5     -60     0      -65  ;...    % bistability
//        1       0.2     -60     -21      0  ;...    % DAP
//      0.02      1       -55     4        0  ;...    % accomodation
//     -0.02      -1      -60     8        80 ;...    % inhibition-induced spiking
//     -0.026     -1      -45     0        80];       % inhibition-induced bursting

* **P** : The proportional term produces an output value that is proportional to the current error value.
* **I** : The integral term is the sum of the instantaneous error over time and gives the accumulated offset that should have been corrected previously. 
The integral term accelerates the movement of the process towards setpoint and eliminates the residual steady-state error that occurs with a pure proportional controller. 
However, since the integral term is accumulated errors from the past, it can cause the present value to overshoot the setpoint value(target value)
* **D** : The derivative term is the rate of change of error and prevents overshoot. Derivative action predicts system behavior and thus improves settling time and 
stability of the system.

<img src="https://user-images.githubusercontent.com/67142421/148653901-3497dcdd-c0d1-4f1e-b59c-9adde9259e48.png">
<img src="https://user-images.githubusercontent.com/67142421/148654000-3df7e315-1842-421e-a38d-8ee38b75041f.png">
<img src="https://user-images.githubusercontent.com/67142421/148653837-4aafcccb-372d-4aa7-a78e-79ec43f2236f.png">

~~~C
double Kp = 1.7; // If this is small it takes more time to reach target, and if it's big overshoot the target.
double Ki = 0.5, Kd = 0.5;
double P, I, D, PID_control;
double desired_angle = 10, current_angle;
double error, error_prev = 0, de;
double time_prev = time.time(), dt; // dt is only needed for the theory with calculus, it's not needed actually in real life

void PID_control() {
  /*
  Code to update current angle here
  */ 

  dt = time.time() - time.prev; 
  error = desired_angle - current_angle;
  de = error - error_prev;

  P = Kp * error;
  I += Ki * error * dt;
  D = Kd * de / dt;

  PID_control = P + I + D; // Control variable(voltage)
  PID_control = constrain(PID_control, 0, 255); // The range of analogWrite() is 1 byte(255)
  analogWrite(6, PID_control);

  error_prev = error;
  time_prev = time.time();
}

void loop() {
  PID_control();
}
~~~

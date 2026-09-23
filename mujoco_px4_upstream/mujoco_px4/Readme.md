### Troubleshooting
1. **Tuning controller parameters**
- The overshooting causes  the collision to occur before the actual collision time in feedforward trajectory causing the dual collision at impact 1 & 2. 
Solution: Tune the parameters to avoid overshooting

- <image src="assets/images/image.png" alt="velocity overshoot screenshot" width="400">

2. **Unable to track the velocity profile (above 2m/s)**
- Is actuator saturation occuring? Max thrust from each motor 12N. The actuators have been clipped to max 12 N in MuJoCo, however, I need to clip it from the commander(geometeric_controller ros node) side as well.  

- <image src="assets/images/motor_output.png" alt="motor output screenshot" width="400"> out of bounds!!!
- <image src="assets/images/motor_output2.png" alt="motor output screenshot2" width="400"> fixed!!

- The tradeoff between rise time and overshoot.
# Face-Recognition
This is an implementation of face recogniser in OpenCV-310 with the face module added to the opencv-310.jar and opencv_java.dll files for javaFX operating on a windows x64 machines.

The purpose of this project was to assess the demands of face recognition on memory, cpu etc on various machines without cuda, tbb, or openGl. Devices ranged from an UDOO-Qaud with 1GB DDR to a Gigabyte ATX with i5 qaud core. The system is designed for deployment in aged care for patients with Alzheimer’s disease where nursing staff have been found to be resistent to operating Linux OS. It is intended the recognizer component allows the master-bot to determine who it communicating with and reference the individual in tts. 

There remain issues with lighting, accuracy,  and frame-rate when using inbuilt webcams. It may be the lower resolution cameras offer a better outcome. Futher assessment is required to optimize capture and recognition.

Original work on the GUI and video capture is by Luigi De Russis of the Politecnico di Torino and who's tutorial material can be found at https://github.com/opencv-java. Face training and face recognition where implimented by Igor who's Linux .so and .jar file can be found at https://github.com/heroinsoul. Compliling the face recognition system openCV for windows based machines and load assessments was undertaken by Jim O'Connorhorrill at Cuerobotics.  

Download opencv-310 and install. Replace the .dll and .jar file in C:/opencv/build/java/x64 and C:/opencv/build/ respectively. The .dll was build for 64bit machines. 



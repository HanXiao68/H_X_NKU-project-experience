%% Track Location of An Object
% Track the location of a physical object moving in one direction.
% 
% Generate synthetic data which mimics the 1-D location of a physical object 
% moving at a constant speed.

detectedLocations = num2cell(2*randn(1,40) + (1:40));
%% 
% Simulate missing detections by setting some elements to empty.
%%
detectedLocations{1} = [];
  for idx = 16: 25 
      detectedLocations{idx} = []; 
  end
%% 
% Create a figure to show the location of detections and the results of 
% using the Kalman filter for tracking.
%%
figure;
hold on;
ylabel('Location');
ylim([0,50]); 
xlabel('Time');
xlim([0,length(detectedLocations)]);
%% 
% Create a 1-D, constant speed Kalman filter when the physical object is 
% first detected. Predict the location of the object based on previous states. 
% If the object is detected at the current time step, use its location to correct 
% the states.
%%
kalman = []; 
for idx = 1: length(detectedLocations) 
   location = detectedLocations{idx}; 
   if isempty(kalman)
     if ~isempty(location) 
       
       stateModel = [1 1;0 1]; 
       measurementModel = [1 0]; 
       kalman = vision.KalmanFilter(stateModel,measurementModel,'ProcessNoise',1e-4,'MeasurementNoise',4);
      kalman.State = [location, 0]; 
     end 
   else
     trackedLocation = predict(kalman);
     if ~isempty(location) 
       plot(idx, location,'k+');
      d = distance(kalman,location); 
       title(sprintf('Distance:%f', d));
       trackedLocation = correct(kalman,location); 
     else 
       title('Missing detection'); 
     end 
     pause(0.2);
     plot(idx,trackedLocation,'ro'); 
   end 
 end 

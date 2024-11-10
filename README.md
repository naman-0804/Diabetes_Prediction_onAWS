**Working Link-https://www.youtube.com/watch?v=OqBdSu4jDhs**

Project is basically Diabetes prediction using ML 

AWS sagemaker is used to train the ml model to predict the diabetes by taking In the required inputs 

AWS amplify is used to host the frontend on our code which will communicate with the backend in ec2

AWS ec2 is used to host the backend to predict and send the data to and from model 

AWS IAM is used to give required permission to ec2, s3 to communicate with each other 

AWS S3 is user to store the ml model for prediction 

AWS API is used to set up communication between the frontend and backend as and when request is given from the user 

AWS SNS is a feature used to send emails to the users regarding their prediction so that they can store it 

AWS DYNAMODB is used to store the user data for future ml improvements

AWS budget is used to keep the cost in check

WINSCP is used for File transferring between ec2 and host device using authentication through .ppk file and  ssh command line

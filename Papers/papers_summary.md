# Short summaries of papers 
#
# Put in every paper you've read on the subject! So, if you considered it useless the others don't waste time
# on reading it again (that's what the importance scale is for)


## Layout
# Filename
Title (importance scale from 0: useless to 5: must-read)[read by initials, e.g. KK]
- Short summary
-- personal comments
##


# microsoft_image_rec_recent.pdf
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification ()[]
-
--

# adaptive_2014_3_30_50086.pdf
Driving Style Recognition for Co-operative Driving: A Survey ()[]
- Using a dataset of 6 drivers, they tried to identify the individual drivers 
--

# DriverIdentification.pdf
Driver Identification by Driving Style (4)[KK]
- Using a dataset of 6 drivers, they tried to identify the individual drivers using either SVMs or multinomial logistic regression with the following features:

speed (created  distribution of speed) / accuracy with SVM 70-75% 
acceleration & decelaration profile (search for points of high acceleration or decelaration -> fit curve to speed profile at those points -> create accelaration distribution) / accuracy with SVM 75-80%
gas mileage (unvailable to us so uninteresting); unfortunately best single feature with 85-90% (SVM)
turning speed vs radius (find turning points) / accuracy 45%

using a combination of speed, acceleration and gas mileage they achieved above 90% with SVMs

using this combination with multinomial logistic regression only achieved 84% (small dataset for log reg + Classification of multiple drivers instead of 1vs1 as with the SVM)
-- good/short paper to get into the topic!

# Driving Style Recognition Based.pdf
Review of Driving Conditions Prediction and Driving Style Recognition Based Control Algorithms for Hybrid Electric Vehicles ()[]
-
--

# NieWuYu_Driving Behavior Improvement and Drive… Information.pdf
Driving Behavior Improvement and Driver Recognition Based on Real-Time DrivingInformation ()[]
-
--

# PAKDD14-PYin.pdf
Mining GPS Data for TrajectoryRecommendation ()[]
- 
--

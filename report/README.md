


image:

inspectedvsref.png
in the image we can see on the right the inspected image versus the reference image

defectsvisualizationfromtxt.png
here is the defects by the locations given in the text (circle of size 10 around each location),
we can see clearly #1 a quite noticle but #2 and #3 are harder to detect. 


images_histograms.png
here is the histogram and some statistical inormation, we can see around 42 graylevel we have a pick in the defect that is not seen in the reference. over all the histogram and stasitsc are quite similar so we didnt apply any normalization or standartization or other operations. 

# Alignment 

althought the images are alike and have the same orientation and grayscale ditribution the images are not completely aligned and in order to get the defect we have to align them. 
alignment can be done using feature matching approaches. 
the most known one is SIFT, which is based on 3 step: intrests points detection, description of these intrest points using histogram of orientation graidents, and matching of the description by knn or other matcher. 
results with SIFT weren't good, so I used ORB. 

** write about ORB **
orb_keypoints.png 
here we can see the keypoints from orb, the size of the blob indicates the intrest point region (a corner in a big blob is not a neccassarley a corner in a smaller blob). and the line is the prinicipal orientation. 

after the feature matching, we can go over to the homography calculation. we use RANSAC which is an iterative method to refine the model and get model which is a consensus of most matches. 

orb_inliers_matches.png
here we can see the inlier matches, we can see quite good correspondence. 

we use the function wrapPerspceitve and wrap the images on top of eachother. 

overlayed.png
we can see in the images that there are parts that are not exactly overlayed perfectly, some of the edges are different.
this image hints about the future issues. 

we calculate the differance image, idially we would like only the defects to be shown in the differance image, 
cuz all the rest should be the same. but in fact we can see clearly some of the edges of the chip pattern are shown and noise. 

diff_image.png
the diff image show us some of the problems better. 

we leave the alignment as is for now and going over to other ways to detect the defects with the alignment we have at the moment.
for future reference it is possible to use ECC iterative refinment and use the homography as an initial point, 
other way is also using other feature matching techniques such as LoFTR and its procedings, which is a detector-free transformers based feature matcher. 

the approach i was thinking about is using Gaussian Mixture Model (GMM) when using the reference image as the background, 
my goal was to try to distinguish the defects as the foreground and to use this kind of separation. 
but unfortunatly the edges of the patterns were classified as foregorund too. 
i use 10 components in the gmm training, i used threshold that uses percentile of 20 for the binary mask creation. 
these two are hyperparameters and it is possible to tune them but since the method should be general as possible i leave it as is. 

** write about GMM **

gmm_10_20_res.png
on the right is the probability map where blue mean it is higly likely to be foreground and the red indicated is it likley to be background. 


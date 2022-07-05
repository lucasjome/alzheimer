This module can be used in order to compute run length texture feature maps of an input image. The computation of the run length features is based on the grey level run length matrix (GLRLM) computed with itk::itkRunLengthTextureFeaturesImageFilter for each pixel’s neighborhood.
The GLRLM matrix describes each neighborhood local texture, it is then used to compute the following run length texture features:

short run emphasis
long run emphasis
grey level non uniformity
run length non uniformity
low grey level run emphasis
high grey level run emphasis
short run low grey level emphasis
short run high grey level emphasis
long run low grey level emphasis
long run high grey level emphasis


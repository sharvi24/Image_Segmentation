# Image_Segmentation

- **Objective:** To produce a DCNN model that can segment seeps from SAR images and classify them, if possible. 

- **Model architecture:** I have selected a U-Net architecture, which is a popular model for image segmentation tasks. U-Net is a fully convolutional neural network that consists of an encoder and a decoder, with skip connections between them. The encoder extracts high-level features from the input image, while the decoder upsamples the features to produce the final segmentation map.

- **Model evaluation:** To evaluate the performance of the model, I have used the Intersection over Union (IoU) metric, which is commonly used for image segmentation tasks. IoU measures the overlap between the predicted and ground truth masks, and ranges from 0 to 1, where 1 indicates perfect overlap. I would compute the IoU score for each class of seeps separately, as well as for the overall segmentation map.




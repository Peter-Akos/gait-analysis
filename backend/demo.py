from mmpose.apis import MMPoseInferencer

img_path = 'demo/image-demo.jpeg'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer("body26")


# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)
result = next(result_generator)

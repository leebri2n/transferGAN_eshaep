Track by week:

Week 1:
- Started development from Google Colab
- Planned out the more rudimentary aspects of data pipeline
	- General objective: Take raw data and transform into a standard format: Accept good-quality, square images of same resolution, while throwing out those deemed unsuitable. Criteria:
	1. Must be of a valid image format: .jpg, .webp, .png, etc. (Later to be standardized to jpeg).
	2. Must have a high-enough resolution, as images will be scaled to same resolution later. In our case, no less than 80% of pixel height, width of desired size (1024x1024)
	3. Grayscale images are difficult to train on, so any grayscale images should be thrown out.
	4. Image must be sufficiently sharp. Utilizes opencv image process to compute blurriness, mathematical operation.
	5. Text must be removed from the images, as a significant amount of text is difficult for the generator to replicate.
	6. Remove human faces, as this can also create difficulty and instability during training phase, like the rest of these.

- At the end of week 1, the pipeline consisted of:
	- Format check
	- Resolution check
	- Facial recognition check
	- Grayscale check
	- Blurriness check
	- Walking entire directory
	- Image quality check (Later removed)
	- Log file writing functionality
	- Image cropping capabiiliity

- Not many issues encountered

Week 2:

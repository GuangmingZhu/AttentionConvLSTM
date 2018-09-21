### The original video files are decoded into images. <br/>
### Images of each video file are stored in one directory named according to the video file name. <br/>
 1) The datalist of each modality (i.e. RGB, Depth, Flow) for training/validation/testing subsets is listed below <br/>
    datalist = ('train_rgb_list.txt','train_depth_list.txt','train_flow_list.txt', 
		'valid_rgb_list.txt','valid_depth_list.txt','valid_flow_list.txt',
		'test_rgb_list.txt','test_depth_list.txt','test_flow_list.txt')
 2) Each item in the text of the datalist is in format like 'path framecount label'. <br/>
    The image name in each 'path' is encoded in format '%06d.jpg' (1...framecount). 'label' is in (0...248). 
 3) You can change the path in the files indicated by datalist below <br/>

#Distorting RGB images: Use linear interpolation to get smoother results
python apply_fisheye_distortion.py dir.input=samples/ file_ext.input=.rgb.png linear_interpolation=True dir.output=samples/output

#Distorting Masks: Use nearest neighbor interpolation to preserve correct values
#python apply_fisheye_distortion.py dir.input=samples/ file_ext.input=.segments.png

#Saving output files to a different directory. A new directory will be created if it does not exist.
#python apply_fisheye_distortion.py dir.output=samples/output



# TODO.....
# 生成鱼眼数据，相应的真值应该怎么处理？
# 用两个图进行辅助生成，一个图用来说明哪些像素属于同一个目标，另外一个图用来说明这些像素的顺序
# 如果存在类别的话，再添加一个新图，用来说明是哪个类别的。
from utils import*

tiles = r"C:\Users\User-1\Desktop\Data\detectron2\New folder\sakhawat_bottom\tiles"
out_file = r"C:\Users\User-1\Desktop\Data\detectron2\New folder\sakhawat_bottom\results"

generate_shapefiles(tiles,out_file)

merge_files(out_file)

# remove_overlapping_bbox(r"C:\Users\User-1\Desktop\Data\detectron2\New folder\sakhawat_bottom\results\merged.shp",out_file)
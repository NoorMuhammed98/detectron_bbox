from utils import*

tiles = r"C:\Users\User-1\Desktop\Data\labels_banana\Final Labels\clipped_tiles_momin2"
out_file = r"C:\Users\User-1\Desktop\Data\labels_banana\Final Labels\New folder"

generate_shapefiles(tiles,out_file)

merge_files(out_file)
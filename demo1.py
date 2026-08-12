import classes as cell

img = cell.Blob_Image("IMG/pic1.webp")
img.detect_blobs()
img.save_results("Result1/blobs.webp")

grid_img = cell.Make_Grids(img.path, 200, img.blobs)
grid_img.assign_blobs()
grid_img.save_region_blobs("Result1/grids_blobs")

grid_img.separate_cells()
grid_img.save_region_cells("Result1/grids_cells")

grid_img.save_region_combined("Result1/grids_combined")
grid_img.save_all_overview("Result1")

for region in grid_img.regions:
    print(f"Region {region.id}: {region.count_blobs()} blobs, {region.count_cells()} cells")

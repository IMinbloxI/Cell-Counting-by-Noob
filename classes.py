import cv2 as cv
import numpy as np
import os

class Blob_Image:
    def __init__(self, path):
        self.path = path #img path
        self.image = cv.imread(path) #img
        self.blobs = [] #all blob
        self.processed = None #img that already processed

    def detect_blobs(self):
        grayscale = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(grayscale, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        kernel = np.ones((5,5), np.uint8)
        closed = cv.morphologyEx(threshold, cv.MORPH_DILATE, kernel)
        self.processed = closed

        contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cnt.size > 0:
                self.blobs.append(cnt)
                x, y, w, h = cv.boundingRect(cnt)
                cv.drawContours(self.image, [cnt], -1, (0, 0, 255), 1)
                cv.rectangle(self.image, (x, y), (x+w, y+h), (255, 255, 0), 1)

    def save_results(self, out_path):
        cv.imwrite(out_path, self.image)

class Region:
    def __init__(self, id, x, y, w, h):
        self.id = id #id 0 - 8
        self.x = x # x mid grid
        self.y = y # y mid grid
        self.w = w # width
        self.h = h # height
        self.blobs = [] # raw blobs assigned
        self.cells = [] # separated cells assigned

    def add_blob(self, contour):
        self.blobs.append(contour)

    def add_cell(self, contour):
        self.cells.append(contour)

    def count_blobs(self):
        return len(self.blobs)

    def count_cells(self):
        return len(self.cells)

class Make_Grids:
    def __init__(self, image_path, n, blobs):
        self.image = cv.imread(image_path) 
        self.n = n # fixed grid size in pixels
        self.blobs = blobs # cotain all raw blobs
        self.regions = self._create_regions() # 0 - 8 of regions

    def _create_regions(self):
        h, w = self.image.shape[:2]
        cx, cy = w // 2, h // 2
        
        # grid boundaries (centered)
        x1 = cx - self.n // 2
        y1 = cy - self.n // 2
        x2 = cx + self.n // 2
        y2 = cy + self.n // 2

        step_x = self.n // 3
        step_y = self.n // 3

        regions = []
        id = 0
        for r in range(3):
            for c in range(3):
                rx = x1 + c * step_x
                ry = y1 + r * step_y
                regions.append(Region(id, rx, ry, step_x, step_y))
                id += 1
        return regions

    def assign_blobs(self):
        for cnt in self.blobs:
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv.drawContours(mask, [cnt], -1, 255, -1)

            for region in self.regions:
                region_mask = mask[region.y:region.y+region.h, region.x:region.x+region.w]
                if np.any(region_mask > 0):
                    region.add_blob(cnt)

    def separate_cells(self):
        for region in self.regions:
            cropped = self.image[region.y:region.y+region.h, region.x:region.x+region.w]

            for blob in region.blobs:
                # local mask
                local_blob = blob.copy()
                local_blob[:,0,0] -= region.x
                local_blob[:,0,1] -= region.y
                mask = np.zeros((region.h, region.w), dtype=np.uint8)
                cv.drawContours(mask, [local_blob], -1, 255, -1)

                dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
                _, sure_fg = cv.threshold(dist, 0.3*dist.max(), 255, 0)  # lower threshold
                sure_fg = np.uint8(sure_fg)
                unknown = cv.subtract(mask, sure_fg)

                _, markers = cv.connectedComponents(sure_fg)
                markers = markers+1
                markers[unknown==255] = 0

                markers = cv.watershed(cropped.copy(), markers)

                for label in np.unique(markers):
                    if label <= 1: continue
                    cell_mask = np.zeros(mask.shape, dtype="uint8")
                    cell_mask[markers==label] = 255
                    cnts, _ = cv.findContours(cell_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        # shift back to global coords
                        c[:,0,0] += region.x
                        c[:,0,1] += region.y
                        region.add_cell(c)

    def save_region_blobs(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        for region in self.regions:
            img_copy = self.image.copy()
            cv.rectangle(img_copy, (region.x, region.y),
                        (region.x+region.w, region.y+region.h),
                        (0,255,0), 2)

            # draw blobs (yellow)
            for cnt in region.blobs:
                cv.drawContours(img_copy, [cnt], -1, (0,255,255), 1)

            out_file = os.path.join(folder_path, f"region_{region.id}_blobs.webp")
            cv.imwrite(out_file, img_copy)

    def save_region_cells(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        for region in self.regions:
            img_copy = self.image.copy()
            cv.rectangle(img_copy, (region.x, region.y),
                        (region.x+region.w, region.y+region.h),
                        (0,255,0), 2)

            # draw cells (red)
            for cnt in region.cells:
                cv.drawContours(img_copy, [cnt], -1, (0,0,255), 1)

            out_file = os.path.join(folder_path, f"region_{region.id}_cells.webp")
            cv.imwrite(out_file, img_copy)

    def save_region_combined(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        for region in self.regions:
            img_copy = self.image.copy()

            # draw region box
            cv.rectangle(img_copy, (region.x, region.y),
                        (region.x+region.w, region.y+region.h),
                        (0,255,0), 2)

            # draw blobs (yellow)
            for cnt in region.blobs:
                cv.drawContours(img_copy, [cnt], -1, (0,255,255), 1)

            # draw cells (red)
            for cnt in region.cells:
                cv.drawContours(img_copy, [cnt], -1, (0,0,255), 1)

            # optional: overlay counts
            text = f"B:{region.count_blobs()} C:{region.count_cells()}"
            cv.putText(img_copy, text, (region.x+5, region.y+15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            out_file = os.path.join(folder_path, f"region_{region.id}_combined.webp")
            cv.imwrite(out_file, img_copy)

    def save_all_overview(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        # --- Blobs overview ---
        img_blobs = self.image.copy()
        for region in self.regions:
            # draw region box
            cv.rectangle(img_blobs, (region.x, region.y),
                        (region.x+region.w, region.y+region.h),
                        (0,255,0), 2)
            # draw blobs (yellow)
            for cnt in region.blobs:
                cv.drawContours(img_blobs, [cnt], -1, (0,255,255), 1)
        cv.imwrite(os.path.join(folder_path, "overview_blobs.webp"), img_blobs)

        # --- Cells overview ---
        img_cells = self.image.copy()
        for region in self.regions:
            # draw region box
            cv.rectangle(img_cells, (region.x, region.y),
                        (region.x+region.w, region.y+region.h),
                        (0,255,0), 2)
            # draw cells (red)
            for cnt in region.cells:
                cv.drawContours(img_cells, [cnt], -1, (0,0,255), 1)
        cv.imwrite(os.path.join(folder_path, "overview_cells.webp"), img_cells)
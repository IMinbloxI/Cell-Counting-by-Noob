import cv2 as cv
from matplotlib import pyplot as plt
import OpenMeUp as omu #Opencv แต่เขียนเอง
import numpy as np

'''
    imread อ่าน pathและ Flag example:cv.IMREAD_###
    return array เมทริกซ์[h,w] = [ค่าblue, ค่าgreen, ค่าred]
'''
img_bgr = cv.imread('png.png')
grey_img = omu.grey_scale(img_bgr) #เฉลี่ยค่า สี bgr Grayscale = 0.299R + 0.587G + 0.114B

# blur = cv.medianBlur(grey_img,3) kernel ขนาด 3x3 นำมาเรียงแล้ว เอาสีกลางเป็น pixel นั้น

threshold = omu.threshold(grey_img,200) # หากค่าความสว่าง > 200 ให้เป็น 255 ขาว ไม่เช่นนั้นเป็น 0 ดำ
# _, threshold = cv.threshold(grey_img,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU) 
# Otsu จะลอง threshold ทุกค่าที่เป็นไปได้
# แล้วเลือกค่าที่แยก object กับ background ได้ดีที่สุด
# โดยทำให้ความแปรปรวนระหว่างสองกลุ่มมีค่ามากที่สุด
# separated = cv.erode(threshold, kernel, iterations=1)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(threshold)
# หา connected components
# labels    = pixel แต่ละจุดอยู่ในกลุ่มใด
# stats     = area, width, height, bounding box
# centroids = จุดกึ่งกลางของแต่ละกลุ่ม

result_img = img_bgr.copy()
counted = 0
print("There are "+str(num_labels-2)+" groups")

areas = []
#สร้าง Areas เฉลี่ยหาค่า median เพื่อเช็คหากลุ่มที่มีโอกาส มีมากกว่า 1 cell
for i in range(2, num_labels):
    areas.append(stats[i,cv.CC_STAT_AREA])

median_area = np.median(areas)
print("median area = "+str(median_area))
kernel = np.ones((3,3), np.uint8)

for i in range(2, num_labels): # ข้าม label 0 background และ label 1 disk ขนาดใหญ่
    x, y = centroids[i]
    area = stats[i, cv.CC_STAT_AREA]
    if area < median_area*2.5: #หากพื้นที่ใกล้เคียง cell ปกติ ถือว่าเป็น 1 cell
        counted+=1
        cv.circle(result_img,(int(x), int(y)),5,(0, 0, 255),1)
        cv.putText(result_img,str(counted),(int(x), int(y)),cv.FONT_HERSHEY_SIMPLEX,0.3,(255, 0, 0),1,cv.LINE_AA)
        
    else: #กลุ่มมีขนาดใหญ่ผิดปกติ แยกเฉพาะกลุ่มนี้ด้วย erosion แล้วนับใหม่
        print(i, stats[i, cv.CC_STAT_AREA]) 
        blob = np.zeros_like(threshold) #สร้างภาพดำ ขนาดเท่า Threshold
        blob[labels == i] = 255 #เลือกกลุ่มของ pixel ทำให้ขาว

        blob_eroded = cv.erode(blob,kernel,iterations=1)
        num_labels2, labels2, stats2, centroids2= cv.connectedComponentsWithStats(blob_eroded) #นับจำนวนกลุ่มของ pixel สีขาวที่เชื่อมต่อกัน

        for j in range(1,num_labels2): # ข้าม label 0 background นับและแสดง cell ที่ถูกแยกออกมา
            x2,y2 = centroids2[j]
            counted+=1
            cv.circle(result_img,(int(x2), int(y2)),5,(0, 255, 0),1)
            cv.putText(result_img,str(counted),(int(x2), int(y2)),cv.FONT_HERSHEY_SIMPLEX,0.3,(0, 255, 0),1,cv.LINE_AA)

print("Detected " + str(counted) + " Cells")
fig, axes = plt.subplots(2, 2, figsize=(10, 5))

axes[0][0].imshow(img_bgr)
axes[0][0].set_title("Original (RGB)")
axes[0][0].axis('off')

axes[0][1].imshow(grey_img,cmap = 'grey')
axes[0][1].set_title("Grayscale")
axes[0][1].axis('off')

axes[1][0].imshow(threshold,cmap = 'grey')
axes[1][0].set_title("Threshold")
axes[1][0].axis('off')

axes[1][1].imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
axes[1][1].set_title("There are " + str(counted)+" cells")
axes[1][1].axis('off')

# plt.tight_layout()
plt.show()

plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
plt.title(f"Detected Cells: {counted}")
plt.axis('off')
plt.show()

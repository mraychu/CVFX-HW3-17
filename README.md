# CVFX-HW3-17
## Abstract
由於 Generative Adversarial Network (GAN) 近年來非常火紅，發展越來越好，但隨之而來衍生出一些問題，例如生成的品質不夠好，output 有失真等等。而此次作業的目的是探討 GAN 內部的 units 經過 dissection 後會產生什麼 artifacts。

## Content
### 1. Generate images with GANPaint
| Original | Draw Grass | Draw Dome | Draw Cloud | 
| ----------------- | --------------- |--------------- |--------------- |
|![](https://i.imgur.com/M7eo18Z.png)|![](https://i.imgur.com/uAnVe11.png)|![](https://i.imgur.com/MZ3sHIJ.png)|![](https://i.imgur.com/f4rcNf3.png)

| Original | Remove Grass | Remove Tree | Remove Door |
| ----------------- | --------------- |--------------- |--------------- |
|![](https://i.imgur.com/IbEdJrQ.png)|![](https://i.imgur.com/K1eL7uz.png)|![](https://i.imgur.com/wI21gqI.png)|![](https://i.imgur.com/dFl6wr4.png)

| Original | Remove Door | Remove Brick | Draw Sky |
| ----------------- | --------------- |--------------- |--------------- |
![](https://i.imgur.com/aNNBwz2.png)|![](https://i.imgur.com/6Al3TEJ.png)|![](https://i.imgur.com/K8fAU2t.png)|![](https://i.imgur.com/6MzWtDo.png)

| Original | Draw Door | Remove Cloud | Remove Dome |
| ----------------- | --------------- |--------------- |--------------- |
![](https://i.imgur.com/RMlpkyp.png)|![](https://i.imgur.com/bPUuAUK.png)|![](https://i.imgur.com/w2QVaZu.png)|![](https://i.imgur.com/F7VHRNr.png)

我們對範例提供的照片做了 draw/remove 的動作。從結果可以觀察到，在對於合理區域進行 draw/remove 物件時，效果都不錯。例如在天空區域把雲消除，或是畫出草叢，結果都蠻合理，只有 remove door 的效果不明顯，比較像把本來的窗戶或門進行變形。但如果在一個不合理的區域進行增減，例如在建築上畫出天空或是雲，都會變成奇怪的結果。如果對於並沒有該物件的區域進行移除，例如對沒有圓頂的房子進行 remove dome，或是對房子本身 remove tree，房子都會發生一些變化，但都在合理範圍。
根據 paper 裡面的內容，remove 應該就是讓針對某區域的 units U 強制為 0 (<img src="http://chart.googleapis.com/chart?cht=tx&chl=$r_U_P=0$" style="border:none;">),而 draw 則是讓該 unit 的值強制為 k(<img src="http://chart.googleapis.com/chart?cht=tx&chl=$r_U_P=k$" style="border:none;">)來進行對 units 關閉/開啟的動作，同一個物件(如 tree)也可能是由好幾個 units 一起控制。我們畫的筆刷範圍則是指定某區域，但由於關閉或開啟了數個 units，影響的範圍則不限於筆刷畫的範圍，可能讓周圍的影像也一起受到影響。

### 2. Dissect any GAN model and analyze what you find
我們測試的是範例中提供的 Karras 的 bedroom model，分別檢查 Layer 4 和 Layer 7，size 使用 1000。

| Layer 4 | Layer 7 |
| ----------------- | --------------- 
|![](https://i.imgur.com/E8wE8GW.png)|![](https://i.imgur.com/VyRwZMX.png)
|![](https://i.imgur.com/2tB5Fsu.png)|![](https://i.imgur.com/c2qSw3p.png)
|![](https://i.imgur.com/S6fOA3V.png)|![](https://i.imgur.com/77TDOIY.png)
|![](https://i.imgur.com/hI8YNZQ.png)|![](https://i.imgur.com/Da20KuZ.png)

**Layer 4**

![](https://i.imgur.com/K5QbbEX.png)

**Layer 7**

![](https://i.imgur.com/jXXkJNC.png)

由上述結果可以觀察到，在 mid layer (Layer 4) 中，objects 和 parts (部分的 object )的數量都比 deep layer (Layer 7) 中更多，與論文中的觀察相符合。Layer 4 和 Layer 7 中，大部分 unit 生成的特徵都還是 Semantic Segmantation 出來的具體物件為主，如天花板、床、枕頭、窗戶、地板等，少部分則是材質，我們認為可能要再檢查更深的 Layer 才會有更抽象的特徵，例如更精細的材質。另外，我們也發現同樣的 units 在不同的 Layer 中，負責生成的 object 或是 part 也有一定程度的變化，有的 IoU 會變好，而有些會變差，這些圈選出來的範圍都會影響後續 unit 的學習效果。

**Layer 4 Ceiling**
> unit 177 (object) iou 0.25

![](https://i.imgur.com/ER07cRi.png)
![](https://i.imgur.com/CKIsDn3.png)

**Layer 7 Ceiling**
> unit 23 (object) iou 0.33

![](https://i.imgur.com/dWUoo1O.png)
![](https://i.imgur.com/caeDK6u.png)

我們在兩個 Layer 中，將 Ceiling 表現最好的兩個 unit 產生出來的結果互相比較。在 IoU 的部分，可以觀察到在更深的 Layer 中，mask 與 Segmantation 有更高的重疊率(IoU)，mask 出來的結果更好，代表經過 越多層 Layer 之後生成的結果的確有改善，當然也有些例外。

### 3. Compare with other method
#### Method: [Image Inpainting](https://docs.opencv.org/3.4.1/df/d3d/tutorial_py_inpainting.html)
> 此方法的 Idea 是若一張影像存在某些物件是我們不想要的，就可以對圖片動一些手腳後能重建一張新的圖片。因此，其定義為在影像中對已丟失或損壞的部份進行重建的過程，實際應用有 Object Removal、Text Removal 以及 Image Restoration 等等。

```python
import cv2
import numpy as np

imagename='orangetree.jpg'
testdir='test3/'
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global needdraw
    if event == cv2.EVENT_LBUTTONDOWN:
        needdraw = True
    if event == cv2.EVENT_LBUTTONUP:
        needdraw = False
    if event == cv2.EVENT_MOUSEMOVE:
        if needdraw:
            cv2.rectangle(img,(x,y),(x+8,y+8),(0,0,0),-1)

img = cv2.imread(imagename)
cv2.namedWindow('image')
needdraw=False
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()

origin = cv2.imread(imagename)
originnp = np.array(origin,dtype=int)
after = np.array(img,dtype=int)

mask = originnp - after

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if not np.array_equal(mask[i][j],[0,0,0]):
            mask[i][j] = [255,255,255]
mask = mask.astype(np.uint8)

cv2.imwrite(testdir+imagename+'_after.jpeg', img)
cv2.imwrite(testdir+imagename+'_mask.jpeg',mask)

mask = mask[:,:,0]
tesla_inpaint = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
ns_inpaint = cv2.inpaint(img,mask,3,cv2.INPAINT_NS)

cv2.imwrite(testdir+imagename+'_tesla_inpaint.jpeg', tesla_inpaint)
cv2.imwrite(testdir+imagename+'_ns_inpaint.jpeg', ns_inpaint)
```

*上面是用 Opencv Python Library 所實作出來的兩種 image inpainting 的方法的 code，我們是用 Jupyter Notebook 來執行，能允許使用者用滑鼠在圖片上畫想要的 mask，再透過 inpaint API 查看效果*

#### 我們參考並使用以下兩種 paper 所提出的方法：
  
#### 1. [“An Image Inpainting Technique Based on the Fast Marching Method” by Alexandru Telea in 2004“](https://pdfs.semanticscholar.org/622d/5f432e515da69f8f220fb92b17c8426d0427.pdf)

 - 此演算法基於 Fast Marching Method (FMM)，主要是參考要被 inpaint 的周遭 pixel, 依照靠近程度給予 weight，越靠近修復點的 pixel 權重越大，並用 FMM 向內推進找下一個要 inpaint 的 pixel，直到修復完所有的 pixels。
   
#### 2. [“Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting” by Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro in 2001“](https://conservancy.umn.edu/bitstream/handle/11299/3607/1772.pdf?sequence=1&isAllowed=y)
 
 - 此演算法則是基於流體力學，並利用偏微分方程，先遍歷位置區塊的邊緣，並持續進行 isophotes (由灰度值相等的點連成的線)，最後透過填補顏色來使區域內的灰度值變化達到最小。

> 我們使用以下這五張圖來做三種不同的實驗

| Apple | Apple Tree | Face | Orange | Orange Tree |
| ----------------- | --------------- |--------------- |--------------- |--------------- |
|![](https://imgur.com/mDJec4b.png)|![](https://imgur.com/d1rOwRn.png)|![](https://imgur.com/aIsanvf.png)|![](https://imgur.com/djPiDXz.png)|![](https://imgur.com/IUCS9m8.png)

#### Experiment 1: random thin line

由下面結果可以發現不論我們怎麼畫，所形成的 mask 使各圖片或是方法 inpaint 過後的圖片看不出明顯差異，故品質皆不差。

| ns | telea | mask | after mask
| ----------------- | --------------- | --------------- | ---------------
|![](https://imgur.com/m1EhD9E.png)|![](https://imgur.com/gSxx53R.png)|![](https://imgur.com/N5jF5lL.png)|![](https://imgur.com/Qpf7qzl.png)
|![](https://imgur.com/MiAc8lo.png)|![](https://imgur.com/GxUbXo7.png)|![](https://imgur.com/XXHIufj.png)|![](https://imgur.com/Uka9woS.png)
|![](https://imgur.com/cQrRHci.png)|![](https://imgur.com/fxjJjOu.png)|![](https://imgur.com/uezQ3gZ.png)|![](https://imgur.com/ItLtD3H.png)
|![](https://imgur.com/lb6DpI8.png)|![](https://imgur.com/icEzxY6.png)|![](https://imgur.com/T4Hc8fd.png)|![](https://imgur.com/mLULb4l.png)
|![](https://imgur.com/banDVMs.png)|![](https://imgur.com/mx6rDKy.png)|![](https://imgur.com/Bhawi2N.png)|![](https://imgur.com/qUhtWVm.png)

#### Experiment 2: random thick line

由於未知的範圍增加了，相比於 Experiment 1，圖片就會有明顯類似馬賽克的 defect 或是 mask 波紋，如 face 圖片中的眼睛就不見了，但還是可以看出原本圖片的樣貌，若是周圍背景顏色較為單一，也不會與原圖相差太多。

| ns | telea | mask | after mask
| ----------------- | --------------- | --------------- | ---------------
|![](https://imgur.com/LUFhOyF.png)|![](https://imgur.com/09Znx9w.png)|![](https://imgur.com/dihdWzl.png)|![](https://imgur.com/ZfNbb1R.png)
|![](https://imgur.com/JdEOVtx.png)|![](https://imgur.com/nsETz4p.png)|![](https://imgur.com/Zi66ZqM.png)|![](https://imgur.com/AKD1MiA.png)
|![](https://imgur.com/6rSSZGv.png)|![](https://imgur.com/WLiV1Pn.png)|![](https://imgur.com/ua9iGlu.png)|![](https://imgur.com/JmHZkeG.png)
|![](https://imgur.com/3Uf72Ey.png)|![](https://imgur.com/OXTHMCg.png)|![](https://imgur.com/Hy9hyVo.png)|![](https://imgur.com/35TuwBn.png)
|![](https://imgur.com/e20nzQV.png)|![](https://imgur.com/jVagSiQ.png)|![](https://imgur.com/0Bw85fT.png)|![](https://imgur.com/RZlMgUj.png)

### Experiment 3: erase object

由於先前有看過 nvidia 的[影片](https://www.youtube.com/watch?v=tU484zM3pDY)以及自己動手玩過，因此我們想做與 nvidia 類似的實驗，但不出意料地，大部分的結果並不是很好，只有在被刪除的周圍顏色背景差不多時，才會有比較真實的結果，但如果是小部分不同的像素想要移除，這依然是個不錯的方法。

| ns | telea | mask | after mask
| ----------------- | --------------- | --------------- | ---------------
|![](https://imgur.com/2OvAQWh.png)|![](https://imgur.com/d2K6Rix.png)|![](https://imgur.com/G5FSWe5.png)|![](https://imgur.com/tgjByIC.png)
|![](https://imgur.com/nIOyHQe.png)|![](https://imgur.com/VrowgfI.png)|![](https://imgur.com/xobJEHH.png)|![](https://imgur.com/8Vgdc6O.png)
|![](https://imgur.com/dNQRKxy.png)|![](https://imgur.com/C31LDYB.png)|![](https://imgur.com/wylteSn.png)|![](https://imgur.com/aiLOeyS.png)
|![](https://imgur.com/hfDN9y1.png)|![](https://imgur.com/YiwDLms.png)|![](https://imgur.com/A6u8nWi.png)|![](https://imgur.com/yZ77Wi1.png)
|![](https://imgur.com/QHzk1ci.png)|![](https://imgur.com/Lgp31Uj.png)|![](https://imgur.com/HkiQb3B.png)|![](https://imgur.com/39RFN6Z.png)

## Conclusion

| | GANPaint | Random the line | Random thick line | Erase object |
| ----------------- | --------------- |--------------- |--------------- |--------------- |
| image quality | better | not bad | worse | not bad (depend on object)
| effectiveness | better | not bad | worse | not bad
| draw mask     | random | random | random | random
| object removal| better | X | X | worse
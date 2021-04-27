from torchvision.utils import save_image
import os
def save_feature(feature_map,basePath,layerName):
    basePath = os.path.join("feature",basePath)
    savePath = os.path.join(basePath,layerName)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    for i in range(feature_map.size(1)):
         save_image(feature_map[0][i], os.path.join(savePath,"image_{}.jpg".format(i)),
                               nrow=1, padding=0)
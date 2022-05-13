# Construction-Site-Monitoring
Construction workers have to be continuously monitored to make sure that they  adopt the usage of proper Personal Protective Equipment (PPE), such as hardhats, vests ,etc. But monitoring them manually through CCTV footages is a tedious task, hence there is a need for introducing an automate system.

## Dataset Preparation:
### PPE kit: 

CHV, Pictor v3 and other web mined images.
Identify people from the images using YOLO v4
Bounding box divided into 4 halves – Head : 1/4th part, Body – 2/4th and 3/4th part, Leg- final part.

### Construction Equipment / Tools :
ACID, MOCS, Google v6 tools and web mined tools images.
Custom labelled used CVAT tool. – ‘Equipment’ and ‘Tool’

## Architecture:
![image](https://user-images.githubusercontent.com/88308675/168221582-587c43fa-6017-4777-8c72-14ecacbcaeac.png)




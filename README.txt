How to run this app:

1. Install Python 3.7+ 
2. Install dependencies: pip install -r requirements.txt
3. Run the app: python appPackage.py
4. Open browser to http://localhost:5001

Changes Needed:
Currently excludes any occluded poses i.e. not all 17 keypoints --> No data for that person, that frame
-How to deal with occlusions in proper COCO way
Maybe make it dump finished video results if it has stopped partially through
-Write to dump folder which is added to repo?
--Also prevents data loss if front end is reloaded
Maybe add downsampling option
-If CPU and fps>30 -->downsample to 30 fps
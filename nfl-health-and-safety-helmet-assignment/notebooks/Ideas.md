#TODO:
- we can simply get the relative distances from tracking data and associate it baseline boxes already given!
    - kalman filter to fill in the gap in tracking data to make it 59.9 hz, then associate
- use deepsort? others are using it on already given baseline boxes taking one main advantage that is cosine distances of object features! (they use Dk and ignore Da, original authers claim is Da is multiple times more useful than Da!)
[reference](https://nanonets.com/blog/object-tracking-deepsort/)

- yolo v5 uses anchors at multiple resolution stages (to detect tiny, medium, and massive objects). While in NFL we only need to detect helmets (and possibly jersey numbers) and so we can keep 1 stage (or 2 at most), and the backbone can also be shrinked(to match that of in tesla ai day presentation) 
[reference](https://youtu.be/Grir6TZbc1M?t=543)
- also the anchor box themselves can be improved as we know what our object shapes will be (literally almost a square for the helmet), YOLO has predefined (found using k-means) anchor boxes so each grid cell only predicts 1 class but the anchor boxes may vary in size so each box specialized in variety of object types


- do whole end to end NN architecture (do away with deepsort etc)
- key pieces:
  - baselineboxes are provided but training our own yolo would give us access to the features that are used to generate the boxes. Use these features for Cosine distances like in deepsort.
  - deepsort works by tracking an object through similarity in features! so only helmet is not enough as all helmets look kinda similar so the person's jersey or some other characteristics has to be included in features(nn should automatically take care of this)
  - kalmanfilters can be applied to the bboxes but it's better for tracking data to make 10hz to 59.9 hz
  - sideline and endzone viws matter; our baseline features should also learn to identify this (we've already got labels so maybe we can directly feed this! although, forcing a network learn this on it's own might make it learn something fundamental that would come in handy when we have some kind of fusion of bboxes with NGS data)
  - some kind of transformation of bboxes to map it to NGS vector space; we have endzone and sideline both that can be laid onto NGS vectorspace; Now both the endzone and sideline view need different transformations so should we use 2 networks or 1 network with intrinsic knowledge about camera view or fed in the sideline/endzone bit.
  - it's better to transpose camera data on to NGS space (3d to 2d, because 2d to 3d there exist many possibilities)
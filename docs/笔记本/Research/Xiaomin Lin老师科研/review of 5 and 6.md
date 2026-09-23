This is my review of **Context-Aware Risk Assessment for Autonomous Underwater Robots in Marine Debris Management Tasks** and **Edge-Oriented Multi-Task Vision for Shrimp Detection and Residual-Feed Segmentation**.

## 1. **Context-Aware Risk Assessment for Autonomous Underwater Robots in Marine Debris Management Tasks**

This article point out that when we using underwater robot to do underwater tasks, we not only need to complete the tasks but also protect the underwater environment, people, fish and so on in the water from harm. The article's idea is that we can combine semantic segmentation with object detection.

There are several bright spots I found:

1. RGB images are used as input for YOLOv9c to detect marine debris, but RGB alone does not provide accurate 3D geometric relationships. Therefore, the authors use depth information from a stereo camera to estimate the 3D positions and distances between debris and surrounding semantic regions.
2. The Kalman filter predicts the debris motion based on its previous state, and then corrects the prediction using new observations. This allows the system to continuously update the estimated trajectory instead of relying on a fixed prediction.
3. Instead of training a new large unified model, the authors reuse and integrate existing models because underwater annotated data are scarce. This makes the system more practical and easier to deploy.
4. Because SUIMNet is not trained with a marine debris class, it may misclassify debris pixels as other semantic classes, such as reef, fish, or background. To correct this, the authors use the debris segmentation mask produced by YOLO to update the SUIMNet segmentation map. In regions identified as debris by YOLO, the SUIMNet predictions are overridden. The authors also apply dilation to the YOLO mask so that misclassified pixels around the debris boundaries can also be removed.

I also found some limitations that I think the article has:

1. One limitation is that the current method mainly uses a fixed-horizon future prediction. Even if the current scene is safe and the predicted state at time $T$ is unsafe, the risk threshold may actually be crossed at an intermediate time $t^*$ between $0$ and $T$. For safety-critical control, it would be better to estimate the entire future risk trajectory and identify the earliest predicted threshold-crossing time.
2. The framework mainly relies on predicted motion under previously observed dynamics and does not explicitly address sudden environmental disturbances. For example, an unexpected current could push the AUV toward a diver and invalidate the previous risk prediction. A more robust system should include online disturbance detection and an emergency safety mechanism that reacts when the observed motion deviates significantly from the predicted trajectory.
3. I noticed is that the current metric aggregates risks from all semantic classes into a single scalar. Although class-specific weights are used, all hazards are eventually compared using the same global threshold. I wonder whether it would be safer to maintain class-specific risk values and thresholds, especially for safety-critical classes such as humans. For example, a high human-specific risk could independently trigger an unsafe decision even if the overall aggregated risk remains below the global threshold. The aggregate metric could still be retained to capture cumulative risks from multiple surrounding objects.


## 2. **Edge-Oriented Multi-Task Vision for Shrimp Detection and Residual-Feed Segmentation**

This article shows that when shrimp detection and semantic segmentation are both required, sharing feature extraction in a single multi-task model can reduce computational cost, memory usage, and hardware pressure compared with running two separate models.

There are two bright spots I found:

1. From this article, I learned that when a network is too large for resource-constrained hardware, we can reduce the size of the network and use a larger model to guide the smaller model during training. This technique is called knowledge distillation.
2. If the training data for two tasks have disjoint annotations, training one task may update the shared features in a way that hurts the other task. One possible way to reduce this interference is to separate the training into different stages and freeze task-important components while training the other task.

I also found some limitations:

1. The paper directly removes YOLO’s original backbone and replaces it with PIDNet features. Since PIDNet is mainly designed for segmentation, these features may not be the best for detection, which may partly explain the drop in detection accuracy. One possible improvement is to use the full YOLO model as a teacher during training and let the fused detector learn from it through knowledge distillation. After training, the teacher can be removed, so we can improve detection performance without adding extra deployment cost.
2. The detection and segmentation annotations belong to separate data pools. One possible improvement is to train task-specific teacher models and use cross-task pseudo-labeling to generate the missing annotations for each pool. This could convert the disjoint datasets into approximately jointly annotated data and enable more direct multi-task training.